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

from .vision_lstm_util import interpolate_sincos, to_ntuple, VitPatchEmbed, VitPosEmbed2d, DropPath, SequenceConv2d


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
 

# # original
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


# Updated MatrixLSTMCell
class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True, chunk_size=256, return_last_states = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        
        self.return_last_states = True

        # CPU-compatible backend configuration
        self.cpu_backend_config = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            chunk_size=chunk_size,
            return_last_states=return_last_states,
        )
        self.cpu_backend = mLSTMBackend(self.cpu_backend_config)
        
        # GPU-compatible (Triton) backend configuration
        self.gpu_backend_config = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_xl_chunk",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            chunk_size=chunk_size,
            return_last_states=return_last_states,
        )
        self.gpu_backend = None  # Lazily initialize
        
        self.causal_mask_cache = {}  # Retained for potential fallback
        self.reset_parameters()

    def get_gpu_backend(self, device):
        """Lazily create the GPU backend and move it to the specified device."""
        if self.gpu_backend is None:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
            self.gpu_backend = mLSTMBackend(self.gpu_backend_config).to(device)
        return self.gpu_backend

    def forward(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor, 
        v: torch.Tensor,
        # optional hidden states we can pass in
        c_state: torch.Tensor = None,
        n_state: torch.Tensor = None,
        m_state: torch.Tensor = None,
        return_states: bool = False
            ) -> torch.Tensor:
        B, S, H = q.shape  # (B, S, H)

        # Ensure all inputs are on the same device
        if not (q.device == k.device == v.device):
            raise ValueError("All input tensors (q, k, v) must be on the same device.")
        
        device = q.device
        if device.type == 'cuda':
            #print("using cuda backend mlstm")
            backend = self.get_gpu_backend(device)
        else:
            #print("using cpu backend")
            backend = self.cpu_backend

        # Prepare gate inputs
        if_gate_input = torch.cat([q, k, v], dim=-1)  # (B, S, 3*H)
        i = self.igate(if_gate_input).transpose(-1, -2)  # (B, NH, S)
        f = self.fgate(if_gate_input).transpose(-1, -2)  # (B, NH, S)

        # Reshape q, k, v for backend
        q = q.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)
        k = k.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)
        v = v.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)

        # **Call the backend** – request last states if return_states=True
        # c_state, n_state, m_state are optional initial states
        if return_states:
            # pass return_last_states=True
            # The backend will return (h_state, (c,n,m))
            h_state, (c_out, n_out, m_out) = backend(
                q=q, k=k, v=v, i=i, f=f,
                c_initial=c_state,
                n_initial=n_state,
                m_initial=m_state,
                return_last_states=True,   # <– crucial
                mode="train"               # or "inference", whichever you want
            )
        else:
            # old usage, just get h_state
            h_state = backend(q=q, k=k, v=v, i=i, f=f)  # no states returned
            c_out, n_out, m_out = None, None, None

        # Apply normalization and reshape
        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, H)  # (B, S, H)
        
        if return_states:
            return h_state_norm, (c_out, n_out, m_out)
        else:
            return h_state_norm

    def reset_parameters(self):
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
        self.outnorm.reset_parameters()

class ViLLayer(nn.Module):
    def __init__(
        self,
        dim,
        direction,
        expansion=2,
        qkv_block_size=4,
        proj_bias=True,
        norm_bias=True,
        conv_bias=True,
        conv_kernel_size=4,
        conv_kind="2d",
        init_weights="original",
        seqlens=None,
        num_blocks=None,
        chunk_size=256,  # New parameter
        use_hidden_states=False,      # <--- KEY FLAG
    ):
        super().__init__()
        assert dim % qkv_block_size == 0
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.proj_bias = proj_bias
        self.conv_bias = conv_bias
        self.conv_kernel_size = conv_kernel_size
        self.conv_kind = conv_kind
        self.init_weights = init_weights
        self.num_blocks = num_blocks
        self.use_hidden_states = use_hidden_states  # <--- STORE FLAG

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size

        # Up-projection
        self.proj_up = nn.Linear(
            in_features=dim,
            out_features=2 * inner_dim,
            bias=proj_bias,
        )

        # Q/K/V expansions
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

        # Convolution (2D or causal1D)
        if conv_kind == "causal1d":
            self.conv = CausalConv1d(
                dim=inner_dim,
                kernel_size=conv_kernel_size,
                bias=conv_bias,
            )
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, \
                "For 2D, we need an odd kernel size for same spatial dims"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        else:
            raise NotImplementedError(f"conv_kind={conv_kind} not implemented.")

        # mLSTM cell
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
            chunk_size=chunk_size,
            return_last_states=use_hidden_states,  # let it return states if needed
            use_hidden_states=use_hidden_states    # pass the same flag
        )

        # Learnable skip
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

        # Down-projection
        self.proj_down = nn.Linear(
            in_features=inner_dim,
            out_features=dim,
            bias=proj_bias,
        )

        self.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        c_state: torch.Tensor = None,
        n_state: torch.Tensor = None,
        m_state: torch.Tensor = None,
    ):
        """
        If self.use_hidden_states=True, we accept c_state/n_state/m_state from the caller,
        pass them into mlstm_cell, and return (output, (c_out, n_out, m_out)).
        Otherwise, we do the original single-output logic.
        """
        B, S, _ = x.shape

        # If direction=BOT_RIGHT, flip along sequence dimension
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError(f"Unknown direction: {self.direction}")

        # Up-projection => shape (B, S, 2*inner_dim)
        x_inner = self.proj_up(x)
        # Split into two halves: x_mlstm & z
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)

        # Convolution => (B, S, inner_dim)
        x_mlstm_conv = self.conv(x_mlstm)
        x_mlstm_conv_act = F.silu(x_mlstm_conv)

        # Q/K/V from the convolved branch
        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        # ============================
        #  mLSTM: handle hidden state
        # ============================
        if self.use_hidden_states and any(s is not None for s in [c_state, n_state, m_state]):
            # pass states in, get updated states
            h_tilde_state, (c_out, n_out, m_out) = self.mlstm_cell(
                q=q, k=k, v=v,
                c_state=c_state,
                n_state=n_state,
                m_state=m_state,
            )
        else:
            # old stateless usage
            # returns just output, no states
            h_tilde_out = self.mlstm_cell(q=q, k=k, v=v)
            c_out, n_out, m_out = None, None, None
            h_tilde_state = h_tilde_out

        # Residual skip
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)

        # Activation on z
        h_state = h_tilde_state_skip * F.silu(z)

        # Down-projection => final shape (B, S, dim)
        out = self.proj_down(h_state)

        # If direction=BOT_RIGHT, flip back
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            out = out.flip(dims=[1])

        # Return states only if we're in hidden-states mode
        if self.use_hidden_states:
            return out, (c_out, n_out, m_out)
        else:
            return out

    def reset_parameters(self):
        # init inproj
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

        # init outproj
        if self.init_weights == "original":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
        else:
            raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        # skip param
        nn.init.ones_(self.learnable_skip)

        # Initialize q/k/v
        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        # Reset parameters for mLSTM cell
        self.mlstm_cell.reset_parameters()


# class ViLLayer(nn.Module):
#     """
#     A Vision-Language Layer (ViLLayer) refactored to use a post-MLP design.
#     This implementation operates directly on the embedding dimension (dim) and uses
#     normalization and residual connections around mLSTM and MLP sub-layers,
#     eliminating the explicit up-projection and down-projection of the original design.
#     """
#     def __init__(
#         self,
#         dim,
#         direction,
#         expansion=2,
#         qkv_block_size=4,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=4,
#         conv_kind="2d",
#         init_weights="original",
#         seqlens=None,
#         num_blocks=None,
#         chunk_size=256,
#     ):
#         """
#         Initialize the ViLLayer with a post-MLP structure.

#         Args:
#             dim (int): Embedding dimension (input and output dimension).
#             direction (SequenceTraversal): Direction of sequence processing (e.g., ROWWISE_FROM_TOP_LEFT).
#             expansion (int): Expansion factor for the MLP's internal dimension.
#             qkv_block_size (int): Block size for q/k/v projections in the mLSTM cell.
#             proj_bias (bool): Whether to include bias in q/k/v and MLP projections.
#             norm_bias (bool): Whether to include bias in normalization layers.
#             conv_bias (bool): Whether to include bias in the convolution layer.
#             conv_kernel_size (int): Kernel size for the convolution.
#             conv_kind (str): Type of convolution ("causal1d" or "2d").
#             init_weights (str): Weight initialization method ("original" or "original-fixed").
#             seqlens (optional): Sequence lengths for 2D convolution.
#             num_blocks (int, optional): Number of blocks for weight initialization.
#             chunk_size (int): Chunk size for the mLSTM cell processing.
#         """
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.proj_bias = proj_bias
#         self.conv_bias = conv_bias
#         self.conv_kernel_size = conv_kernel_size
#         self.conv_kind = conv_kind
#         self.init_weights = init_weights
#         self.num_blocks = num_blocks

#         # Number of heads based on the input dimension and block size
#         num_heads = dim // qkv_block_size

#         # Normalization layers for mLSTM and MLP sub-layers
#         self.norm1 = nn.LayerNorm(dim, bias=norm_bias)  # Before mLSTM
#         self.norm2 = nn.LayerNorm(dim, bias=norm_bias)  # Before MLP

#         # Convolution layer operates directly on dim (no up-projection)
#         if conv_kind == "causal1d":
#             self.conv = CausalConv1d(
#                 dim=dim,
#                 kernel_size=conv_kernel_size,
#                 bias=conv_bias,
#             )
#         elif conv_kind == "2d":
#             assert conv_kernel_size % 2 == 1, "Even kernel sizes not supported for same shape output"
#             self.conv = SequenceConv2d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens,
#             )
#         else:
#             raise NotImplementedError(f"Convolution kind '{conv_kind}' not implemented")

#         # Q, K, V projections operate on dim (not inner_dim)
#         self.q_proj = LinearHeadwiseExpand(
#             dim=dim,
#             num_heads=num_heads,
#             bias=proj_bias,
#         )
#         self.k_proj = LinearHeadwiseExpand(
#             dim=dim,
#             num_heads=num_heads,
#             bias=proj_bias,
#         )
#         self.v_proj = LinearHeadwiseExpand(
#             dim=dim,
#             num_heads=num_heads,
#             bias=proj_bias,
#         )

#         # mLSTM cell operates on dim
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=dim,
#             num_heads=qkv_block_size,
#             norm_bias=norm_bias,
#             chunk_size=chunk_size,
#         )

#         # Learnable skip connection for the mLSTM output
#         self.learnable_skip = nn.Parameter(torch.ones(dim))

#         # MLP components with internal dimension expansion
#         inner_dim = expansion * dim
#         self.mlp_gate = nn.Linear(dim, inner_dim, bias=proj_bias)  # Gate projection
#         self.mlp_up = nn.Linear(dim, inner_dim, bias=proj_bias)    # Value projection
#         self.mlp_down = nn.Linear(inner_dim, dim, bias=proj_bias)  # Down projection

#         # Initialize weights
#         self.reset_parameters()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Forward pass of the ViLLayer with post-MLP design.

#         Args:
#             x (torch.Tensor): Input tensor of shape [B, S, dim], where:
#                 - B is the batch size
#                 - S is the sequence length
#                 - dim is the feature dimension

#         Returns:
#             torch.Tensor: Output tensor of shape [B, S, dim]
#         """
#         B, S, _ = x.shape

#         # Adjust sequence direction if needed
#         if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
#             pass
#         elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])
#         else:
#             raise NotImplementedError(f"Direction '{self.direction}' not implemented")

#         # mLSTM sub-layer
#         # Normalize input
#         x_norm = self.norm1(x)  # Shape: [B, S, dim]

#         # Conditional convolution handling based on conv_kind
#         if self.conv_kind == "causal1d":
#             # For causal1d, transpose to [B, D, S] for CausalConv1d
#             x_conv = self.conv(x_norm.transpose(1, 2))  # Shape: [B, D, S] -> [B, D, S]
#             x_mlstm_conv = x_conv.transpose(1, 2)       # Shape: [B, S, D]
#         elif self.conv_kind == "2d":
#             # For 2d, pass directly to SequenceConv2d which expects [B, S, D]
#             x_mlstm_conv = self.conv(x_norm)            # Shape: [B, S, D]
#         else:
#             raise ValueError(f"Unknown conv_kind: {self.conv_kind}")

#         # Apply activation after convolution
#         x_mlstm_conv_act = F.silu(x_mlstm_conv)  # Shape: [B, S, D]

#         # Compute q, k, v projections
#         q = self.q_proj(x_mlstm_conv_act)  # Shape: [B, S, dim]
#         k = self.k_proj(x_mlstm_conv_act)  # Shape: [B, S, dim]
#         v = self.v_proj(x_norm)            # Shape: [B, S, dim], uses normalized input

#         # Process through mLSTM cell
#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)  # Shape: [B, S, dim]

#         # Add learnable skip connection
#         h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)  # Shape: [B, S, dim]

#         # Residual connection
#         h_mlstm = x + h_tilde_state_skip  # Shape: [B, S, dim]

#         # MLP sub-layer
#         # Normalize intermediate result
#         h_norm = self.norm2(h_mlstm)  # Shape: [B, S, dim]

#         # Compute gate and value projections
#         gate = self.mlp_gate(h_norm)  # Shape: [B, S, expansion * dim]
#         z = self.mlp_up(h_norm)       # Shape: [B, S, expansion * dim]

#         # Apply gating mechanism
#         mlp_output = F.silu(gate) * z  # Shape: [B, S, expansion * dim]

#         # Project back to original dimension
#         mlp_output = self.mlp_down(mlp_output)  # Shape: [B, S, dim]

#         # Residual connection
#         x_out = h_mlstm + mlp_output  # Shape: [B, S, dim]

#         # Reverse direction adjustment if needed
#         if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
#             pass
#         elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x_out = x_out.flip(dims=[1])
#         else:
#             raise NotImplementedError(f"Direction '{self.direction}' not implemented")

#         return x_out

#     def reset_parameters(self):
#         """
#         Initialize the layer's parameters according to the specified method.
#         """
#         # Initialize q/k/v projections
#         small_init_(self.q_proj.weight, dim=self.dim)
#         small_init_(self.k_proj.weight, dim=self.dim)
#         small_init_(self.v_proj.weight, dim=self.dim)
#         if self.q_proj.bias is not None:
#             nn.init.zeros_(self.q_proj.bias)
#         if self.k_proj.bias is not None:
#             nn.init.zeros_(self.k_proj.bias)
#         if self.v_proj.bias is not None:
#             nn.init.zeros_(self.v_proj.bias)

#         # Initialize MLP projections
#         small_init_(self.mlp_gate.weight, dim=self.dim)
#         small_init_(self.mlp_up.weight, dim=self.dim)
#         if self.init_weights == "original":
#             wang_init_(self.mlp_down.weight, dim=self.dim, num_blocks=1)
#         elif self.init_weights == "original-fixed":
#             wang_init_(self.mlp_down.weight, dim=self.dim, num_blocks=self.num_blocks)
#         else:
#             raise NotImplementedError(f"Init method '{self.init_weights}' not implemented")
#         if self.mlp_gate.bias is not None:
#             nn.init.zeros_(self.mlp_gate.bias)
#         if self.mlp_up.bias is not None:
#             nn.init.zeros_(self.mlp_up.bias)
#         if self.mlp_down.bias is not None:
#             nn.init.zeros_(self.mlp_down.bias)

#         # Initialize learnable skip parameter
#         nn.init.ones_(self.learnable_skip)

#         # Initialize mLSTM cell parameters
#         self.mlstm_cell.reset_parameters()

class RMSNorm(nn.Module):
    """
    Root Mean Square Normalization (RMSNorm) as used in the xLSTMLarge model.
    """
    def __init__(self, num_features, eps=1e-6, use_weight=True, use_bias=False):
        super().__init__()
        self.eps = eps
        self.use_weight = use_weight
        self.use_bias = use_bias
        if use_weight:
            self.weight = nn.Parameter(torch.ones(num_features))
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
        x = x / rms
        if self.use_weight:
            x = x * self.weight
        if self.use_bias:
            x = x + self.bias
        return x

# class ViLLayer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         direction,
#         expansion=2,
#         qkv_block_size=4,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=4,
#         conv_kind="2d",
#         init_weights="original",
#         seqlens=None,
#         num_blocks=None,
#         chunk_size=256,
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.proj_bias = proj_bias
#         self.conv_bias = conv_bias
#         self.conv_kernel_size = conv_kernel_size
#         self.conv_kind = conv_kind
#         self.init_weights = init_weights
#         self.num_blocks = num_blocks

#         # Normalization layers using RMSNorm
#         self.norm1 = RMSNorm(num_features=dim, eps=1e-6, use_weight=True, use_bias=norm_bias)
#         self.norm2 = RMSNorm(num_features=dim, eps=1e-6, use_weight=True, use_bias=norm_bias)

#         # Convolution for vision-specific feature extraction
#         if conv_kind == "2d":
#             self.conv = SequenceConv2d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens,
#             )
#         elif conv_kind == "causal1d":
#             self.conv = CausalConv1d(
#                 dim=dim,
#                 kernel_size=conv_kernel_size,
#                 bias=conv_bias,
#             )
#         else:
#             raise NotImplementedError(f"Convolution kind '{conv_kind}' not implemented")

#         # Q, K, V projections using nn.Linear
#         self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=proj_bias)

#         # mLSTM cell
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=dim,
#             num_heads=qkv_block_size,
#             norm_bias=norm_bias,
#             chunk_size=chunk_size,
#         )

#         # MLP components
#         inner_dim = expansion * dim
#         self.mlp_gate = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.mlp_up = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.mlp_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         self.reset_parameters()


# class ViLLayer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         direction,
#         expansion=2,
#         qkv_block_size=4,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=4,
#         conv_kind="2d",
#         init_weights="original",
#         seqlens=None,
#         num_blocks=None,
#         chunk_size=256,
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.proj_bias = proj_bias
#         self.conv_bias = conv_bias
#         self.conv_kernel_size = conv_kernel_size
#         self.conv_kind = conv_kind
#         self.init_weights = init_weights
#         self.num_blocks = num_blocks

#         # Normalization layers using RMSNorm
#         self.norm1 = RMSNorm(num_features=dim, eps=1e-6, use_weight=True, use_bias=norm_bias)
#         self.norm2 = RMSNorm(num_features=dim, eps=1e-6, use_weight=True, use_bias=norm_bias)

#         # Convolution for vision-specific feature extraction
#         if conv_kind == "2d":
#             self.conv = SequenceConv2d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens,
#             )
#         elif conv_kind == "causal1d":
#             self.conv = CausalConv1d(
#                 dim=dim,
#                 kernel_size=conv_kernel_size,
#                 bias=conv_bias,
#             )
#         else:
#             raise NotImplementedError(f"Convolution kind '{conv_kind}' not implemented")

#         # Q, K, V projections using nn.Linear
#         self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=proj_bias)

#         # mLSTM cell
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=dim,
#             num_heads=qkv_block_size,
#             norm_bias=norm_bias,
#             chunk_size=chunk_size,
#         )

#         # MLP components
#         inner_dim = expansion * dim
#         self.mlp_gate = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.mlp_up = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.mlp_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         self.reset_parameters()

# class ViLLayer(nn.Module):
#     def __init__(
#         self,
#         dim,
#         direction,
#         expansion=2,
#         qkv_block_size=4,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=4,
#         conv_kind="2d",
#         init_weights="original",
#         seqlens=None,
#         num_blocks=None,
#         chunk_size=256,
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.proj_bias = proj_bias
#         self.conv_bias = conv_bias
#         self.conv_kernel_size = conv_kernel_size
#         self.conv_kind = conv_kind
#         self.init_weights = init_weights
#         self.num_blocks = num_blocks

#         # Normalization layers using RMSNorm
#         self.norm1 = RMSNorm(num_features=dim, eps=1e-6, use_weight=True, use_bias=norm_bias)
#         self.norm2 = RMSNorm(num_features=dim, eps=1e-6, use_weight=True, use_bias=norm_bias)

#         # Convolution for vision-specific feature extraction
#         if conv_kind == "2d":
#             self.conv = SequenceConv2d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens,
#             )
#         elif conv_kind == "causal1d":
#             self.conv = CausalConv1d(
#                 dim=dim,
#                 kernel_size=conv_kernel_size,
#                 bias=conv_bias,
#             )
#         else:
#             raise NotImplementedError(f"Convolution kind '{conv_kind}' not implemented")

#         # Q, K, V projections using nn.Linear
#         self.q_proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.k_proj = nn.Linear(dim, dim, bias=proj_bias)
#         self.v_proj = nn.Linear(dim, dim, bias=proj_bias)

#         # mLSTM cell
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=dim,
#             num_heads=qkv_block_size,
#             norm_bias=norm_bias,
#             chunk_size=chunk_size,
#         )

#         # MLP components
#         inner_dim = expansion * dim
#         self.mlp_gate = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.mlp_up = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.mlp_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         self.reset_parameters()

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         B, S, _ = x.shape

#         # Direction handling
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # mLSTM sub-layer
#         x_norm = self.norm1(x)  # [B, S, dim]
#         if self.conv_kind == "2d":
#             x_conv = self.conv(x_norm)  # [B, S, dim]
#         elif self.conv_kind == "causal1d":
#             x_conv = self.conv(x_norm.transpose(1, 2)).transpose(1, 2)
#         else:
#             raise ValueError(f"Unknown conv_kind: {self.conv_kind}")
#         x_conv_act = F.silu(x_conv)  # [B, S, dim]

#         # Compute q, k, v from the same convolved input
#         q = self.q_proj(x_conv_act)  # [B, S, dim]
#         k = self.k_proj(x_conv_act)  # [B, S, dim]
#         v = self.v_proj(x_conv_act)  # [B, S, dim]
#         h_tilde_state = self.mlstm_cell(q, k, v)  # [B, S, dim]
#         h_mlstm = x + h_tilde_state  # Direct residual connection

#         # MLP sub-layer
#         h_norm = self.norm2(h_mlstm)  # [B, S, dim]
#         gate = self.mlp_gate(h_norm)  # [B, S, inner_dim]
#         z = self.mlp_up(h_norm)  # [B, S, inner_dim]
#         mlp_output = F.silu(gate) * z  # [B, S, inner_dim]
#         mlp_output = self.mlp_down(mlp_output)  # [B, S, dim]
#         x_out = h_mlstm + mlp_output  # Second residual connection

#         # Reverse direction if needed
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x_out = x_out.flip(dims=[1])

#         return x_out

#     def reset_parameters(self):
#         # Initialize q/k/v projections
#         small_init_(self.q_proj.weight, dim=self.dim)
#         small_init_(self.k_proj.weight, dim=self.dim)
#         small_init_(self.v_proj.weight, dim=self.dim)
#         if self.proj_bias:
#             nn.init.zeros_(self.q_proj.bias)
#             nn.init.zeros_(self.k_proj.bias)
#             nn.init.zeros_(self.v_proj.bias)

#         # Initialize MLP projections
#         small_init_(self.mlp_gate.weight, dim=self.dim)
#         small_init_(self.mlp_up.weight, dim=self.dim)
#         if self.init_weights == "original":
#             wang_init_(self.mlp_down.weight, dim=self.dim, num_blocks=1)
#         elif self.init_weights == "original-fixed":
#             wang_init_(self.mlp_down.weight, dim=self.dim, num_blocks=self.num_blocks)
#         else:
#             raise NotImplementedError
#         if self.proj_bias:
#             nn.init.zeros_(self.mlp_gate.bias)
#             nn.init.zeros_(self.mlp_up.bias)
#             nn.init.zeros_(self.mlp_down.bias)

#         # Initialize convolution and mLSTM cell
#         self.conv.reset_parameters()
#         self.mlstm_cell.reset_parameters()

class ViLBlock(nn.Module):
    def __init__(
            self,
            dim,
            direction,
            drop_path=0.0,
            conv_kind="2d",
            conv_kernel_size=3,
            proj_bias=True,
            norm_bias=True,
            seqlens=None,
            num_blocks=None,
            init_weights="original",
            chunk_size=256,
            use_hidden_states=False,  # <--- NEW FLAG
    ):
        """
        If use_hidden_states=True, this block can handle passing hidden states
        (c_state,n_state,m_state) to its internal ViLLayer. Otherwise, it behaves statelessly.
        """
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_prob = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights
        self.use_hidden_states = use_hidden_states  # <--- store the flag

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.layer = ViLLayer(
            dim=dim,
            direction=direction,
            conv_kind=conv_kind,
            conv_kernel_size=conv_kernel_size,
            seqlens=seqlens,
            norm_bias=norm_bias,
            proj_bias=proj_bias,
            num_blocks=num_blocks,
            init_weights=init_weights,
            chunk_size=chunk_size,
            use_hidden_states=use_hidden_states,  # propagate the flag
        )

        self.reset_parameters()

    def _forward_fn(self, x, c_state=None, n_state=None, m_state=None):
        """
        This internal function normalizes 'x' and then calls self.layer.
        If self.use_hidden_states=True, we pass states to the layer and return both output + updated states.
        Otherwise, we call it stateless and return only the output.
        """
        x = self.norm(x)
        if self.use_hidden_states:
            out, (c_out, n_out, m_out) = self.layer(x, c_state, n_state, m_state)
            return out, (c_out, n_out, m_out)
        else:
            out = self.layer(x)
            return out, (None, None, None)

    def forward(
        self,
        x: torch.Tensor,
        c_state: torch.Tensor = None,
        n_state: torch.Tensor = None,
        m_state: torch.Tensor = None,
    ):
        """
        Main forward pass. If use_hidden_states=True, we accept c_state/n_state/m_state
        and pass them down to the layer. We do drop_path in a function call so that
        we can retrieve both output + hidden states if needed.
        """
        if self.use_hidden_states:
            # We'll define a lambda that calls _forward_fn with states
            out, (c_new, n_new, m_new) = self.drop_path(
                x,
                forward_fn=lambda z: self._forward_fn(z, c_state, n_state, m_state)
            )
            return out, (c_new, n_new, m_new)
        else:
            # Old usage: no hidden states
            out, _ = self.drop_path(x, forward_fn=self._forward_fn)
            return out

    def reset_parameters(self):
        self.layer.reset_parameters()
        self.norm.reset_parameters()



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
            use_hidden_states=False,  # <--- new flag
    ):
        """
        If use_hidden_states=True, both sub-blocks can accept and return c_state/n_state/m_state.
        This class passes states to each one and merges the results.
        """
        super().__init__()

        self.use_hidden_states = use_hidden_states

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
            use_hidden_states=use_hidden_states,  # <--- propagate flag
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
            use_hidden_states=use_hidden_states,  # <--- propagate flag
        )

    def forward(
        self,
        x,
        c_tl=None, n_tl=None, m_tl=None,
        c_br=None, n_br=None, m_br=None,
    ):
        """
        If use_hidden_states=True, we accept:
          - c_tl,n_tl,m_tl for top-left sub-block
          - c_br,n_br,m_br for bottom-right sub-block
        Return updated states for each sub-block if needed.

        If use_hidden_states=False, we do the original stateless forward,
        ignoring c_*, n_*, m_*.
        """
        if self.use_hidden_states:
            # (1) Forward top-left block
            out1, (ct1, nt1, mt1) = self.rowwise_from_top_left(
                x,
                c_state=c_tl,
                n_state=n_tl,
                m_state=m_tl
            )
            # (2) Forward bottom-right block
            out2, (ct2, nt2, mt2) = self.rowwise_from_bot_right(
                out1,
                c_state=c_br,
                n_state=n_br,
                m_state=m_br
            )
            return out2, (ct1, nt1, mt1, ct2, nt2, mt2)
        else:
            # stateless usage
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
                )x = einops.rearrange(x, "b h w d -> b (h w) d")  # (B, S, D)
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
