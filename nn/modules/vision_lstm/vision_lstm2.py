# This file is licensed under Apache-2.0
# Copyright (c) NXAI GmbH and its affiliates 2024
# Benedikt Alkin, Maximilian Beck, Korbinian Pöppel
import math
import warnings
from enum import Enum

import einops
import torch
import torch.nn.functional as F
from torch import nn

from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d, SequenceConv3d

class SequenceTraversal(Enum):
    ROWWISE_FROM_TOP_LEFT = "rowwise_from_top_left"
    ROWWISE_FROM_BOT_RIGHT = "rowwise_from_bot_right"


def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param


def small_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param

def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)

def parallel_stabilized_simple(
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        igate_preact: torch.Tensor,
        fgate_preact: torch.Tensor,
        lower_triangular_matrix: torch.Tensor = None,
        stabilize_rowwise: bool = True,
        eps: float = 1e-6,
) -> torch.Tensor:
    """
    This is the mLSTM cell in parallel form.
    This version is stabilized. We control the range of exp() arguments by
    ensuring that they are always smaller than 0.0 by subtracting the maximum.

    Args:
        :param queries: (torch.Tensor) (B, NH, S, DH)
        :param keys: (torch.Tensor) (B, NH, S, DH)
        :param values: (torch.Tensor) (B, NH, S, DH)
        :param igate_preact: (torch.Tensor) (B, NH, S, 1)
        :param fgate_preact: (torch.Tensor) (B, NH, S, 1)
        :param lower_triangular_matrix: (torch.Tensor) (S,S). Defaults to None.
        :param stabilize_rowwise: (bool) Wether to stabilize the combination matrix C rowwise (take maximum per row).
            Alternative: Subtract the maximum over all rows. Defaults to True.
        :param eps: (float) small constant to avoid division by 0. Defaults to 1e-6.

    Returns:
        torch.Tensor: (B, NH, S, DH), h_tilde_state
    """

    B, NH, S, DH = queries.shape
    _dtype, _device = queries.dtype, queries.device

    # forget gate matrix
    log_fgates = torch.nn.functional.logsigmoid(fgate_preact)  # (B, NH, S, 1)
    if lower_triangular_matrix is None or S < lower_triangular_matrix.size(-1):
        ltr = torch.tril(torch.ones((S, S), dtype=torch.bool, device=_device))
    else:
        ltr = lower_triangular_matrix
    assert ltr.dtype == torch.bool, f"lower_triangular_matrix must be of dtype bool, got {ltr.dtype}"

    log_fgates_cumsum = torch.cat(
        [
            torch.zeros((B, NH, 1, 1), dtype=_dtype, device=_device),
            torch.cumsum(log_fgates, dim=-2),
        ],
        dim=-2,
    )  # (B, NH, S+1, 1)
    # for each batch/head this is a matrix of shape (S+1, S+1) containing the cumsum of the log forget gate values
    # in the second dimension (colum dimension). Each row has the same is a copy of the first row.
    # First entry of each row is zero.
    rep_log_fgates_cumsum = log_fgates_cumsum.repeat(1, 1, 1, S + 1)  # (B, NH, S+1, S+1)
    # Now in each row cut off / subtract the forgetgate values of the later timesteps
    # where col j > row i
    _log_fg_matrix = rep_log_fgates_cumsum - rep_log_fgates_cumsum.transpose(-2, -1)  # (B, NH, S+1, S+1)
    # Causal masking & selection of the correct submatrix, such that forgetgate at timestep t is not applied
    # to the input at timestep t
    log_fg_matrix = torch.where(ltr, _log_fg_matrix[:, :, 1:, 1:], -float("inf"))  # (B, NH, S, S)

    # gate decay matrix D (combination of forget gate and input gate)
    log_D_matrix = log_fg_matrix + igate_preact.transpose(-2, -1)  # (B, NH, S, S)
    # D matrix stabilization
    if stabilize_rowwise:
        max_log_D, _ = torch.max(log_D_matrix, dim=-1, keepdim=True)  # (B, NH, S, 1)
    else:
        max_log_D = torch.max(log_D_matrix.view(B, NH, -1), dim=-1, keepdim=True)[0].unsqueeze(-1)
        # (B, NH, 1, 1)
    log_D_matrix_stabilized = log_D_matrix - max_log_D  # (B, NH, S, S)
    D_matrix = torch.exp(log_D_matrix_stabilized)  # (B, NH, S, S)

    keys_scaled = keys / math.sqrt(DH)

    # combination matrix C
    qk_matrix = queries @ keys_scaled.transpose(-2, -1)  # (B, NH, S, S)
    C_matrix = qk_matrix * D_matrix  # (B, NH, S, S)
    normalizer = torch.maximum(C_matrix.sum(dim=-1, keepdim=True).abs(), torch.exp(-max_log_D))  # (B, NH, S, 1)
    # (B, NH, S, S)
    C_matrix_normalized = C_matrix / (normalizer + eps)

    # retrieved values
    h_tilde_state = C_matrix_normalized @ values  # (B, NH, S, DH)

    return h_tilde_state


class FeedForward(nn.Module):
    def __init__(
        self,
        embedding_dim,
        ffn_proj_factor=2.6667,
        ffn_round_up_to_multiple_of=64,
        use_bias=True,
        weight_mode="fused",
        num_blocks=12,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.ffn_proj_factor = ffn_proj_factor
        self.ffn_round_up_to_multiple_of = ffn_round_up_to_multiple_of
        self.use_bias = use_bias
        self.weight_mode = weight_mode
        self.num_blocks = num_blocks

        self.up_proj_dim = round_up_to_next_multiple_of(
            embedding_dim * ffn_proj_factor,
            ffn_round_up_to_multiple_of,
        )

        if self.weight_mode == "single":
            self.proj_up_gate = nn.Linear(
                in_features=embedding_dim,
                out_features=self.up_proj_dim,
                bias=use_bias,
            )
            self.proj_up = nn.Linear(
                in_features=embedding_dim,
                out_features=self.up_proj_dim,
                bias=use_bias,
            )
        elif self.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(
                in_features=embedding_dim,
                out_features=2 * self.up_proj_dim,
                bias=use_bias,
            )

        self.proj_down = nn.Linear(
            in_features=self.up_proj_dim,
            out_features=embedding_dim,
            bias=use_bias,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y

    def reset_parameters(self):
        if self.weight_mode == "fused":
            small_init_(self.proj_up_gate_z.weight, dim=self.embedding_dim)
            if self.proj_up_gate_z.bias is not None:
                nn.init.zeros_(self.proj_up_gate_z.bias)
        elif self.weight_mode == "single":
            small_init_(self.proj_up_gate.weight, dim=self.embedding_dim)
            small_init_(self.proj_up.weight, dim=self.embedding_dim)
            if self.proj_up_gate.bias is not None:
                nn.init.zeros_(self.proj_up_gate.bias)
            if self.proj_up.bias is not None:
                nn.init.zeros_(self.proj_up.bias)

        wang_init_(
            self.proj_down.weight,
            dim=self.embedding_dim,
            num_blocks=self.num_blocks or 1,
        )
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

#gpt 4.5
class ViLLayer(nn.Module):
    def __init__(self,
            dim,
            direction,
            expansion=2,
            qkv_block_size=4,
            proj_bias=True,
            norm_bias=True,
            conv_bias=True,
            conv_kernel_size=3,
            conv_kind="2d",
            init_weights="original-fixed",
            seqlens=None,
            num_blocks=12,
            gate_soft_cap=15.0,
            ffn_proj_factor=2.6667,
            ffn_round_up_to_multiple_of=64,
            weight_mode="fused",
            chunk_size=64
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.gate_soft_cap = gate_soft_cap
        self.weight_mode = weight_mode
        self.num_blocks = num_blocks

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)
        self.qkv_proj = nn.Linear(inner_dim, 3 * inner_dim, bias=proj_bias)
        
        self.q_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.k_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )
        self.v_proj = LinearHeadwiseExpand(
            dim=inner_dim,
            num_heads=num_heads,
            bias=proj_bias,
        )

        if conv_kind == "causal1d":
            self.conv = CausalConv1d(inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1
            self.conv = SequenceConv2d(
                inner_dim, inner_dim, kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2, groups=inner_dim,
                bias=conv_bias, seqlens=seqlens
            )
        else:
            raise NotImplementedError

        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=num_heads,
            norm_bias=norm_bias,
            eps=1e-5
        )

        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)

        self.ffn = FeedForward(
            embedding_dim=dim,
            ffn_proj_factor=ffn_proj_factor,
            ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
            use_bias=proj_bias,
            weight_mode=weight_mode,
            num_blocks=num_blocks or 1,
        )

        self.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, 2, dim=-1)

        x_mlstm_conv_act = F.silu(self.conv(x_mlstm))

        # fused QKV projections
        qkv = self.qkv_proj(x_mlstm_conv_act)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # q = self.q_proj(x_mlstm_conv_act)
        # k = self.k_proj(x_mlstm_conv_act)
        # v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        h_state = h_tilde_state_skip * F.silu(z)

        x = self.proj_down(h_state)

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        x = residual + x

        # FFN with residual
        ffn_residual = x
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = ffn_residual + x_ffn

        return x

    def reset_parameters(self):
        small_init_(self.proj_up.weight, self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

        small_init_(self.qkv_proj.weight, self.dim)
        if self.qkv_proj.bias is not None:
            nn.init.zeros_(self.qkv_proj.bias)

        wang_init_(self.proj_down.weight, self.dim, num_blocks=self.num_blocks or 1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)
        self.mlstm_cell.reset_parameters()
        self.norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

#

#GPT 4.5 refactor
class ViLBlock(nn.Module):
    def __init__(self,
        dim,
        direction,
        drop_path=0.2,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=None,
        init_weights="original",
        chunk_size=256,
        qkv_block_size = 4):


        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.norm_bias = norm_bias

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = nn.RMSNorm(dim, eps=1e-3)
        self.layer = ViLLayer(dim,
                                direction,
                                qkv_block_size=qkv_block_size,
                                proj_bias=True,
                                norm_bias=True,
                                conv_bias=True,
                                conv_kernel_size=3,
                                conv_kind="2d",
                                init_weights="original",
                                seqlens=None,  # Initial seqlens, can be overridden in forward
                                num_blocks=None,
                                chunk_size=64,
                            )

        self.reset_parameters()

    def _forward_path(self, x):
        x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_path(x, self._forward_path)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()

# #original mlstm

# class MatrixLSTMCell(nn.Module):
#     def __init__(self, dim, num_heads, norm_bias=True):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads

#         self.igate = nn.Linear(3 * dim, num_heads)
#         self.fgate = nn.Linear(3 * dim, num_heads)
#         self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
#         self.causal_mask_cache = {}
#         self.reset_parameters()

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         B, S, _ = q.shape  # (B, S, H)

#         if_gate_input = torch.cat([q, k, v], dim=-1)
#         q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
#         k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
#         v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

#         q = q.transpose(1, 2)  # (B, NH, S, DH)
#         k = k.transpose(1, 2)  # (B, NH, S, DH)
#         v = v.transpose(1, 2)  # (B, NH, S, DH)

#         # compute input and forget gate pre-activations
#         igate_preact = self.igate(if_gate_input)  # (B, S, NH)
#         igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
#         fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
#         fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

#         # cache causal mask to avoid memory allocation in every iteration
#         if S in self.causal_mask_cache:
#             causal_mask = self.causal_mask_cache[(S, str(q.device))]
#         else:
#             causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
#             self.causal_mask_cache[(S, str(q.device))] = causal_mask

#         h_state = parallel_stabilized_simple(
#             queries=q,
#             keys=k,
#             values=v,
#             igate_preact=igate_preact,
#             fgate_preact=fgate_preact,
#             lower_triangular_matrix=causal_mask,
#         )  # (B, NH, S, DH)

#         h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
#         h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

#         return h_state_norm

#     def reset_parameters(self):
#         self.outnorm.reset_parameters()
#         # forget gate initialization
#         torch.nn.init.zeros_(self.fgate.weight)
#         bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
#         # input gate initialization
#         torch.nn.init.zeros_(self.igate.weight)
#         torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)

#grok backendsclass 
from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
class MatrixLSTMCell(nn.Module):
        def __init__(self, dim, num_heads, norm_bias=True, eps=1e-6, chunk_size=16, use_autocast=True, autocast_dtype=torch.float32):
            super().__init__()
            self.dim = dim
            self.num_heads = num_heads
            self.use_autocast = use_autocast  # Added to enable/disable autocasting
            self.autocast_dtype = autocast_dtype  # Added to specify autocast dtype

            self.igate = nn.Linear(3 * dim, num_heads)
            self.fgate = nn.Linear(3 * dim, num_heads)
            self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias, eps=1e-3)
            self.causal_mask_cache = {}
            chunk_size = 64



            # CPU-compatible backend configuration (remains float32)
            self.cpu_backend_config_infer = mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--native_autograd",
                sequence_kernel="native_sequence__native",
                step_kernel="native",
                chunk_size=int(chunk_size),
                autocast_kernel_dtype="float32",
                return_last_states=True,
                mode="inference",
                eps=1e-5
            )
            self.cpu_backend = mLSTMBackend(
                config=self.cpu_backend_config_infer,
            )

            # GPU-compatible (Triton) backend configuration
            self.gpu_backend_config_infer = mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
                sequence_kernel="native_sequence__triton",
                step_kernel="triton",
                chunk_size=int(chunk_size),
                autocast_kernel_dtype="float32",  # Autocast in forward pass can override this
                return_last_states=True,
                mode="inference",
                eps=1e-5
            )
            self.gpu_backend = mLSTMBackend(
                config=self.gpu_backend_config_infer,
            )


            # CPU-compatible backend configuration (remains float32)
            self.cpu_backend_config = mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--native_autograd",
                sequence_kernel="native_sequence__native",
                step_kernel="native",
                chunk_size=int(chunk_size),
                autocast_kernel_dtype="float32",
                return_last_states=False,
                mode="train",
                eps=1e-5
            )
            self.cpu_backend = mLSTMBackend(
                config=self.cpu_backend_config,
            )

            # GPU-compatible (Triton) backend configuration
            self.gpu_backend_config = mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
                sequence_kernel="native_sequence__triton",
                step_kernel="triton",
                chunk_size=int(chunk_size),
                autocast_kernel_dtype="float32",  # Autocast in forward pass can override this
                return_last_states=False,
                mode="train",
                eps=1e-5
            )
            self.gpu_backend = mLSTMBackend(
                config=self.gpu_backend_config,
            )

            self.reset_parameters()


        def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            B, S, H = q.shape  # (B, S, H)
            # print("Input type" + str(q.dtype))
            dtype = q.dtype
            device = q.device

            # All inputs must reside on the same device
            if not (q.device == k.device == v.device):
                raise ValueError("All input tensors (q, k, v) must be on the same device.")
            backend = self.gpu_backend if device.type == 'cuda' else self.cpu_backend

            # Prepare gate inputs
            if_gate_input = torch.cat([q, k, v], dim=-1)
            i = self.igate(if_gate_input).transpose(-1, -2)  # (B, NH, S)
            f = self.fgate(if_gate_input).transpose(-1, -2)  # (B, NH, S)

            # Reshape for backend
            q = q.view(B, S, self.num_heads, -1).transpose(1, 2)
            k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
            v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

            # Compute h_state with the selected backend, applying autocast if specified
            if device.type == 'cuda' and self.use_autocast and self.train:
                q = q.to(self.autocast_dtype).contiguous()
                k = k.to(self.autocast_dtype).contiguous()
                v = v.to(self.autocast_dtype).contiguous()
                i = i.to(self.autocast_dtype).contiguous()
                f = f.to(self.autocast_dtype).contiguous()
                # print("Autocast type" + str(q.dtype))
                h_state = backend(
                    q=q,
                    k=k,
                    v=v,
                    i=i,
                    f=f,
                )  
            elif device.type == 'cpu' and self.train:
                h_state = backend(
                    q=q,
                    k=k,
                    v=v,
                    i=i,
                    f=f,
                )  # (B, NH, S, DH)
                        # Compute h_state with the selected backend, applying autocast if specified
            if device.type == 'cuda' and self.use_autocast and not self.training:
                q = q.to(self.autocast_dtype).contiguous()
                k = k.to(self.autocast_dtype).contiguous()
                v = v.to(self.autocast_dtype).contiguous()
                i = i.to(self.autocast_dtype).contiguous()
                f = f.to(self.autocast_dtype).contiguous()
                h_state = backend(
                    q=q,
                    k=k,
                    v=v,
                    i=i,
                    f=f,
                )  # (B, NH, S, DH)
            elif device.type == 'cpu' and not self.training:
                h_state = backend(
                    q=q,
                    k=k,
                    v=v,
                    i=i,
                    f=f,
                )  # (B, NH, S, DH)

            h_state = h_state.to(dtype)
            h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
            h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

            # print("Output type" + str(h_state.dtype))
            # Force output to match input dtype
            return h_state_norm

        def reset_parameters(self):
            self.outnorm.reset_parameters()
            # Forget gate initialization
            torch.nn.init.zeros_(self.fgate.weight)
            bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
            # Input gate initialization
            torch.nn.init.zeros_(self.igate.weight)
            torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)



class MultiHeadRMSNorm(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.eps = eps
        self.rmsnorm = nn.RMSNorm(num_heads * head_dim, eps=eps, elementwise_affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, NH, S, DH) -> (B, S, NH, DH)
        x = x.transpose(1, 2)  # (B, S, NH, DH)
        B, S, NH, DH = x.shape
        x = x.reshape(B, S, -1)  # (B, S, NH * DH)
        x = self.rmsnorm(x)  # (B, S, NH * DH)
        x = x.view(B, S, NH, DH).transpose(1, 2)  # (B, NH, S, DH)
        return x

class LinearHeadwiseExpand(nn.Module):
    """
    This is a structured projection layer that projects the input to a higher dimension.
    It only allows integer up-projection factors, i.e. the output dimension is a multiple of the input dimension.
    """

    def __init__(self, dim, num_heads, bias=False):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads

        dim_per_head = dim // num_heads
        self.weight = nn.Parameter(torch.empty(num_heads, dim_per_head, dim_per_head))
        if bias:
            self.bias = nn.Parameter(torch.empty(dim))
        else:
            self.bias = None
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.weight.data, mean=0.0, std=math.sqrt(2 / 5 / self.weight.shape[-1]))
        if self.bias is not None:
            nn.init.zeros_(self.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = einops.rearrange(x, "... (nh d) -> ... nh d", nh=self.num_heads)
        x = einops.einsum(
            x,
            self.weight,
            "... nh d, nh out_d d -> ... nh out_d",
        )
        x = einops.rearrange(x, "... nh out_d -> ... (nh out_d)")
        if self.bias is not None:
            x = x + self.bias
        return x

    def extra_repr(self):
        return (
            f"dim={self.dim}, "
            f"num_heads={self.num_heads}, "
            f"bias={self.bias is not None}, "
        )


class CausalConv1d(nn.Module):
    """
    Implements causal depthwise convolution of a time series tensor.
    Input:  Tensor of shape (B,T,F), i.e. (batch, time, feature)
    Output: Tensor of shape (B,T,F)

    Args:
        feature_dim: number of features in the input tensor
        kernel_size: size of the kernel for the depthwise convolution
        causal_conv_bias: whether to use bias in the depthwise convolution
        channel_mixing: whether to use channel mixing (i.e. groups=1) or not (i.e. groups=feature_dim)
                        If True, it mixes the convolved features across channels.
                        If False, all the features are convolved independently.
    """

    def __init__(self, dim, kernel_size=4, bias=True):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.bias = bias
        # padding of this size assures temporal causality.
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=kernel_size,
            padding=self.pad,
            groups=dim,
            bias=bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # conv requires dim first
        x = einops.rearrange(x, "b l d -> b d l")
        # causal conv1d
        x = self.conv(x)
        x = x[:, :, :-self.pad]
        # back to dim last
        x = einops.rearrange(x, "b d l -> b l d")
        return x


class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False. """

    def __init__(
            self,
            ndim: int = -1,
            weight: bool = True,
            bias: bool = False,
            eps: float = 1e-5,
            residual_weight: bool = True,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(ndim)) if weight else None
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        self.eps = eps
        self.residual_weight = residual_weight
        self.ndim = ndim
        self.reset_parameters()

    @property
    def weight_proxy(self) -> torch.Tensor:
        if self.weight is None:
            return None
        if self.residual_weight:
            return 1.0 + self.weight
        else:
            return self.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            normalized_shape=(self.ndim,),
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )

    def reset_parameters(self):
        if self.weight_proxy is not None:
            if self.residual_weight:
                nn.init.zeros_(self.weight)
            else:
                nn.init.ones_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class MultiHeadLayerNorm(LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, "Input must be 4D tensor (B, NH, S, DH)"
        B, NH, S, DH = x.shape

        gn_in_1 = x.transpose(1, 2)  # (B, S, NH, DH)
        gn_in_2 = gn_in_1.reshape(B * S, NH * DH)  # (B * S, NH * DH)
        out = F.group_norm(
            gn_in_2,
            num_groups=NH,
            weight=self.weight_proxy,
            bias=self.bias,
            eps=self.eps,
        )  # .to(x.dtype)
        # (B * S), (NH * DH) -> (B, S, NH, DH) -> (B, NH, S, DH)
        out = out.view(B, S, NH, DH).transpose(1, 2)
        return out

from mlstm_kernels.torch.backend_module import mLSTMBackendConfig, mLSTMBackend
# # # # original
# class MatrixLSTMCell(nn.Module):
#     def __init__(self, dim, num_heads, norm_bias=True):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads

#         self.igate = nn.Linear(3 * dim, num_heads)
#         self.fgate = nn.Linear(3 * dim, num_heads)
#         self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
#         self.causal_mask_cache = {}
#         self.reset_parameters()

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         B, S, _ = q.shape  # (B, S, H)

#         if_gate_input = torch.cat([q, k, v], dim=-1)
#         q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
#         k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
#         v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

#         q = q.transpose(1, 2)  # (B, NH, S, DH)
#         k = k.transpose(1, 2)  # (B, NH, S, DH)
#         v = v.transpose(1, 2)  # (B, NH, S, DH)

#         # compute input and forget gate pre-activations
#         igate_preact = self.igate(if_gate_input)  # (B, S, NH)
#         igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
#         fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
#         fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

#         # cache causal mask to avoid memory allocation in every iteration
#         if S in self.causal_mask_cache:
#             causal_mask = self.causal_mask_cache[(S, str(q.device))]
#         else:
#             causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
#             self.causal_mask_cache[(S, str(q.device))] = causal_mask

#         h_state = parallel_stabilized_simple(
#             queries=q,
#             keys=k,
#             values=v,
#             igate_preact=igate_preact,
#             fgate_preact=fgate_preact,
#             lower_triangular_matrix=causal_mask,
#         )  # (B, NH, S, DH)

#         h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
#         h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

#         return h_state_norm

#     def reset_parameters(self):
#         self.outnorm.reset_parameters()
#         # forget gate initialization
#         torch.nn.init.zeros_(self.fgate.weight)
#         bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
#         # input gate initialization
#         torch.nn.init.zeros_(self.igate.weight)
#         torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


from .mlstm_large import mLSTMLayerVision
from .mlstm_large import VilLayerUpdated

class ViLBlockPair(nn.Module):
    def __init__(
        self,
        dim,
        drop_path=0.0,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=None,
        init_weights="original",
        chunk_size=256,
        qkv_block_size = 4
    ):
        super().__init__()
        self.rowwise_from_top_left = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
            chunk_size=chunk_size,
            qkv_block_size = qkv_block_size
        )
        self.rowwise_from_bot_right = ViLBlock(
            dim=dim,
            direction=SequenceTraversal.ROWWISE_FROM_BOT_RIGHT,
            drop_path=drop_path,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            seqlens=seqlens,
            num_blocks=num_blocks,
            init_weights=init_weights,
            chunk_size=chunk_size,
            qkv_block_size = qkv_block_size
        )

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        out1 = self.rowwise_from_top_left(x)
        out2 = self.rowwise_from_bot_right(out1)
        return out2


class VisionLSTM2(nn.Module):
    def __init__(
            self,
            dim=192,
            input_shape=(3, 224, 224),
            patch_size=16,
            depth=12,
            output_shape=(1000,),
            mode="classifier",
            pooling="bilateral_flatten",
            drop_path_rate=0.0,
            drop_path_decay=False,
            stride=None,
            legacy_norm=False,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            init_weights="original",
    ):
        if depth == 24 and dim < 1024:
            warnings.warn(
                "A single VisionLSTM2 block consists of two subblocks (one for each traversal direction). "
                "ViL-T, ViL-S and ViL-B therefore use depth=12 instead of depth=24, are you sure you want to use "
                "depth=24?"
            )
        super().__init__()
        self.input_shape = input_shape
        self.output_shape = output_shape
        ndim = len(self.input_shape) - 1
        self.patch_size = to_ntuple(patch_size, n=ndim)
        self.dim = dim
        self.depth = depth
        self.stride = stride
        self.mode = mode
        self.pooling = pooling
        self.drop_path_rate = drop_path_rate
        self.drop_path_decay = drop_path_decay
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.proj_bias = proj_bias
        self.norm_bias = norm_bias
        self.init_weights = init_weights

        # initialize patch_embed
        self.patch_embed = VitPatchEmbed(
            dim=dim,
            stride=stride,
            num_channels=self.input_shape[0],
            resolution=self.input_shape[1:],
            patch_size=self.patch_size,
        )

        # pos embed
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=dim)

        # calculate stochastic depth per block
        if drop_path_decay and drop_path_rate > 0.:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        else:
            dpr = [drop_path_rate] * depth

        # merge two blocks into a blockpair to keep depth equal to the depth of transformers
        # useful to keep layer-wise lr decay implementations consistent with transformers
        self.blocks = nn.ModuleList(
            [
                ViLBlockPair(
                    dim=dim,
                    drop_path=dpr[i],
                    conv_kind=conv_kind,
                    seqlens=self.patch_embed.seqlens,
                    proj_bias=proj_bias,
                    norm_bias=norm_bias,
                    num_blocks=depth * 2,
                    init_weights=init_weights,
                )
                for i in range(depth)
            ],
        )
        if pooling == "bilateral_flatten" and mode == "classifier":
            head_dim = dim * 2
        else:
            head_dim = dim
        self.norm = LayerNorm(dim, bias=norm_bias, eps=1e-6)
        # LEGACY: not needed but was used during training
        if legacy_norm:
            self.legacy_norm = nn.LayerNorm(head_dim)
        else:
            self.legacy_norm = nn.Identity()

        # head
        if mode == "features":
            if self.output_shape is not None:
                warnings.warn(f"passed mode=features -> output_shape is ignored ({self.output_shape})")
            self.head = None
            if self.pooling is None:
                self.output_shape = (self.patch_embed.num_patches, dim)
            elif self.pooling == "to_image":
                self.output_shape = (dim, *self.patch_embed.seqlens)
            else:
                warnings.warn(f"passed invalid pooling -> pooling is ignored ({self.pooling})")
                self.pooling = None
        elif mode == "classifier":
            # linear classification head
            assert self.output_shape is not None and len(self.output_shape) == 1, \
                f"define number of classes via output_shape=(num_classes,) (e.g. output_shape=(1000,) for ImageNet-1K"
            self.head = nn.Linear(head_dim, self.output_shape[0])
            # following MAE https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L257
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
        else:
            raise NotImplementedError

    def load_state_dict(self, state_dict, strict=True):
        # interpolate pos_embed for different resolution (e.g. for fine-tuning on higher-resolution)
        old_pos_embed = state_dict["pos_embed.embed"]
        if old_pos_embed.shape != self.pos_embed.embed.shape:
            state_dict["pos_embed.embed"] = interpolate_sincos(embed=old_pos_embed, seqlens=self.pos_embed.seqlens)
        # remove head and adapt layernorm for feature extraction
        if self.mode == "features":
            state_dict.pop("head.weight", None)
            state_dict.pop("head.bias", None)
            # legacy_norm uses head dim (is doubled for bilateral_concat) -> not usable for feature extraction
            cur_sd = self.state_dict()
            state_dict["legacy_norm.weight"] = cur_sd["legacy_norm.weight"]
            state_dict["legacy_norm.bias"] = cur_sd["legacy_norm.bias"]
        return super().load_state_dict(state_dict=state_dict, strict=strict)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed.embed"}

    def forward(self, x):
        # embed patches
        x = self.patch_embed(x)
        # add pos_embed
        x = self.pos_embed(x)

        # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")

        # apply blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)

        # pool
        if self.pooling is None:
            x = self.legacy_norm(x)
        elif self.pooling == "to_image":
            x = self.legacy_norm(x)
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x = einops.rearrange(
                x,
                "b (seqlen_h seqlen_w) dim -> b dim seqlen_h seqlen_w",
                seqlen_h=seqlen_h,
                seqlen_w=seqlen_w,
            )
        elif self.pooling == "bilateral_avg":
            # norm after pooling
            x = (x[:, 0] + x[:, -1]) / 2
            x = self.legacy_norm(x)
        elif self.pooling == "bilateral_flatten":
            # norm after pooling
            x = torch.concat([x[:, 0], x[:, -1]], dim=1)
            x = self.legacy_norm(x)
        else:
            raise NotImplementedError(f"pooling '{self.pooling}' is not implemented")

        # head
        if self.head is not None:
            x = self.head(x)

        return x



class FusionMLPBase(nn.Module):
    def __init__(self, dim, hidden_dim=None):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim or 4 * dim

    def forward(self, x):
        raise NotImplementedError

class MLPBaseline(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)

class GEGLU(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.fc = nn.Linear(dim, self.hidden_dim * 2)
        self.proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.gelu(x1) * x2)

class SwiGLU(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.fc = nn.Linear(dim, self.hidden_dim * 2)
        self.proj = nn.Linear(self.hidden_dim, dim)

    def forward(self, x):
        x1, x2 = self.fc(x).chunk(2, dim=-1)
        return self.proj(F.silu(x1) * x2)

class RGBlock(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        local_dim = self.hidden_dim * 2 // 3
        self.fc1 = nn.Conv2d(dim, local_dim * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(local_dim, local_dim, kernel_size=3, padding=1, groups=local_dim)
        self.fc2 = nn.Conv2d(local_dim, dim, kernel_size=1)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = F.gelu(self.dwconv(x) + x) * v
        return self.fc2(x)

class ConvMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, kernel_size=3, padding=1, groups=self.hidden_dim),
            nn.GELU(),
            nn.Conv2d(self.hidden_dim, dim, kernel_size=1)
        )

    def forward(self, x):
        return self.mlp(x)

class LoRAMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None, rank=16):
        super().__init__(dim, hidden_dim)
        self.rank = min(rank, self.hidden_dim)
        self.down = nn.Linear(dim, self.rank)
        self.up = nn.Linear(self.rank, dim)

    def forward(self, x):
        return self.up(F.relu(self.down(x)))

class MLPMixer(FusionMLPBase):
    def __init__(self, dim, seq_len, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.token_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(seq_len, seq_len),
        )
        self.channel_mlp = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # B, C, S
        x = self.token_mlp(x)
        x = x.transpose(1, 2)
        return self.channel_mlp(x)

class CrossAttentionMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, dim)

    def forward(self, x1, x2):
        q = self.q(x1)
        k = self.k(x2)
        v = self.v(x2)
        attn = F.softmax(q @ k.transpose(-2, -1) / (self.dim ** 0.5), dim=-1)
        return self.out(attn @ v)

class FiLMMLP(FusionMLPBase):
    def __init__(self, dim, hidden_dim=None):
        super().__init__(dim, hidden_dim)
        self.gamma = nn.Linear(dim, dim)
        self.beta = nn.Linear(dim, dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, dim)
        )

    def forward(self, x, modulator):
        gamma = self.gamma(modulator)
        beta = self.beta(modulator)
        return self.ffn(x) * gamma + beta


# -------------------------------------------------------
# Registry of MLP Blocks (use dictionary for swappable logic)
# -------------------------------------------------------
MLP_REGISTRY = {
    "baseline": lambda dim, **kwargs: MLPBaseline(dim, **kwargs),
    "geglu": lambda dim, **kwargs: GEGLU(dim, **kwargs),
    "swiglu": lambda dim, **kwargs: SwiGLU(dim, **kwargs),
    "rgblock": lambda dim, **kwargs: RGBlock(dim, **kwargs),
    "convmlp": lambda dim, **kwargs: ConvMLP(dim, **kwargs),
    "lora": lambda dim, **kwargs: LoRAMLP(dim, **kwargs),
    "mixer": lambda dim, seq_len=64, **kw: MLPMixer(dim, seq_len=seq_len, **kw),
    "crossattn": lambda dim, **kwargs: CrossAttentionMLP(dim, **kwargs),
    "film": lambda dim, **kwargs: FiLMMLP(dim, **kwargs),
}

# -------------------------------------------------------
# FusionViLLayer Class
# -------------------------------------------------------
# - [-1, 1, FusionViLLayerBlock, [256, {
#     "proj_type": "conv",
#     "mlp_type": "swiglu",
#     "seq_len": 64,
#     "use_mlp": true
# }]]

class FusionViLLayer(nn.Module):
    def __init__(
        self,
        dim,
        direction="rowwise_from_top_left",
        mlp_type="baseline",
        mlp_hidden_dim=None,
        use_skip=True,
        use_mlp=True,
        conv_kind="2d",
        conv_kernel_size=3,
        proj_bias=True,
        norm_bias=True,
        seqlens=None,
        num_blocks=1,
        init_weights="original",
        seq_len=None,
        proj_type="linear",  # 'linear', 'conv', or 'sequenceconv'
    ):
        super().__init__()
        self.use_skip = use_skip
        self.use_mlp = use_mlp
        self.seq_len = seq_len
        self.proj_type = proj_type

        # Project + Normalize: supports 3 projection types
        if proj_type == "linear":
            self.input_proj = nn.Linear(dim * 2, dim)
        elif proj_type == "conv":
            self.input_proj = nn.Sequential(
                nn.Conv2d(dim * 2, dim, kernel_size=1, bias=proj_bias),
                nn.BatchNorm2d(dim),
                nn.SiLU()
            )
        elif proj_type == "sequenceconv":
            self.input_proj = SequenceConv2d(
                in_channels=dim * 2,
                out_channels=dim,
                kernel_size=1,
                padding=0,
                bias=proj_bias,
                seqlens=seqlens
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

        self.vilayer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            proj_bias=proj_bias,
            norm_bias=norm_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
        )

        self.residual_proj = nn.Identity() if not use_skip else nn.Linear(dim, dim)

        if use_mlp:
            self.norm2 = LayerNorm(ndim=dim)
            self.post_mlp = MLP_REGISTRY[mlp_type](
                dim, hidden_dim=mlp_hidden_dim or dim * 4, seq_len=seq_len
            )
        else:
            self.post_mlp = None

    def forward(self, x1, x2):
        B, C, H, W = x1.shape
        S = H * W

        if self.proj_type == "conv":
            x = torch.cat([x1, x2], dim=1)           # [B, 2C, H, W]
            x = self.input_proj(x)                  # [B, C, H, W]
            x_seq = rearrange(x, "b c h w -> b (h w) c")
        else:
            x1_seq = rearrange(x1, "b c h w -> b (h w) c")
            x2_seq = rearrange(x2, "b c h w -> b (h w) c")
            x = torch.cat([x1_seq, x2_seq], dim=-1)  # [B, S, 2C]
            x_seq = self.input_proj(x) if self.proj_type == "linear" else self.input_proj(x)

        fused = self.norm(x_seq)
        fused_out = self.vilayer(fused)

        if self.use_skip:
            fused_out = fused_out + self.residual_proj(x1_seq)

        if self.use_mlp:
            fused_out = fused_out + self.post_mlp(self.norm2(fused_out))

        return rearrange(fused_out, "b (h w) c -> b c h w", h=H, w=W)

def small_init_init_(param: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Fills the input Tensor with values according to the method described in Transformers without Tears: Improving
    the Normalization of Self-Attention - Nguyen, T. & Salazar, J. (2019), using a normal distribution.
    Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py.
    """
    std = math.sqrt(2 / (5 * dim))
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param


def wang_init_(param: torch.Tensor, dim: int, num_blocks: int):
    """ Adopted from https://github.com/EleutherAI/gpt-neox/blob/main/megatron/model/init_functions.py. """
    std = 2 / num_blocks / math.sqrt(dim)
    torch.nn.init.normal_(param, mean=0.0, std=std)
    return param

