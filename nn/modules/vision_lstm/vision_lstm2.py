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
        num_blocks=15,
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

#grok refactor no gated Z
@torch.compile
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
                 num_blocks=15,
                 gate_soft_cap=15.0,
                 ffn_proj_factor=2.6667,
                 ffn_round_up_to_multiple_of=64,
                 weight_mode="fused",
                 chunk_size=64
    ):
        super().__init__()
        assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.gate_soft_cap = gate_soft_cap
        self.weight_mode = weight_mode
        self.num_blocks = num_blocks

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.inner_dim = inner_dim
        self.num_heads = num_heads

        # Fused up projection for Q, K, V, adjusted to 2 * inner_dim
        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

        # Convolution applied only to x_qk
        if conv_kind == "causal1d":
            self.conv = CausalConv1d(inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd for 2d conv"
            self.conv = SequenceConv2d(
                inner_dim, inner_dim, kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2, groups=inner_dim,
                bias=conv_bias, seqlens=seqlens
            )
        else:
            raise NotImplementedError(f"conv_kind '{conv_kind}' is not supported")

        # Separate projections for Q and K (from convolved x_qk) and V (from x_v)
        self.qk_proj = nn.Linear(inner_dim, 2 * inner_dim, bias=proj_bias)
        self.v_proj = nn.Linear(inner_dim, inner_dim, bias=proj_bias)

        # MatrixLSTMCell with fused ifgate and soft capping
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=num_heads,
            norm_bias=norm_bias,
            eps=1e-5,
            chunk_size=chunk_size,
            gate_soft_cap=gate_soft_cap
        )

        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        # FFN with its own normalization
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

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        # Handle sequence direction
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        # Fused up projection for Q, K, V
        x_inner = self.proj_up(x)  # (B, S, 2 * inner_dim)
        x_qk, x_v = torch.chunk(x_inner, 2, dim=-1)  # Each (B, S, inner_dim)

        # Convolution on x_qk only
        x_qk_conv_act = F.silu(self.conv(x_qk))  # (B, S, inner_dim)

        # Project to Q and K, and V separately
        qk = self.qk_proj(x_qk_conv_act)  # (B, S, 2 * inner_dim)
        q, k = torch.chunk(qk, 2, dim=-1)  # Each (B, S, inner_dim)
        v = self.v_proj(x_v)  # (B, S, inner_dim)

        # Reshape for multi-head processing
        B, S, _ = q.shape
        # q = q.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, head_dim)
        # k = k.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, head_dim)
        # v = v.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, head_dim)

        # mLSTM processing
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)  # (B, S, inner_dim)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_qk_conv_act)  # (B, S, inner_dim)
        x_mlstm = self.proj_down(h_tilde_state_skip)  # (B, S, dim)

        # Revert sequence direction if needed
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x_mlstm = x_mlstm.flip(dims=[1])

        # Residual connection after mLSTM
        x = residual + x_mlstm

        # FFN with residual connection
        ffn_residual = x
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = ffn_residual + x_ffn

        return x

    def reset_parameters(self):
        # Initialize weights as per original conventions
        small_init_(self.proj_up.weight, self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

        small_init_(self.qk_proj.weight, self.dim)
        if self.qk_proj.bias is not None:
            nn.init.zeros_(self.qk_proj.bias)

        small_init_(self.v_proj.weight, self.dim)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

        wang_init_(self.proj_down.weight, self.dim, num_blocks=self.num_blocks or 1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)
        self.mlstm_cell.reset_parameters()
        self.norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()



# ------------------------------------------------------------
# 2) Modify ViLLayer to accept a list of seqlens instead of a single one
# ------------------------------------------------------------
@torch.compile
class ViLLayerMultiScale(nn.Module):
    def __init__(
        self,
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
        seqlens=None,               # ← now expects a LIST-of-[Hi,Wi] pairs
        num_blocks=15,
        gate_soft_cap=15.0,
        ffn_proj_factor=2.6667,
        ffn_round_up_to_multiple_of=64,
        weight_mode="fused",
        chunk_size=64,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
        self.dim = dim
        self.direction = direction
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.gate_soft_cap = gate_soft_cap
        self.weight_mode = weight_mode
        self.num_blocks = num_blocks

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.inner_dim = inner_dim
        self.num_heads = num_heads

        # Fused up-projection for Q, K, V
        self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

        # --- Replace the single‐scale SequenceConv2d with the multi‐scale version ---
        if conv_kind == "causal1d":
            raise NotImplementedError("This example focuses on 2d only")
        elif conv_kind == "2d":
            # seqlens is now a list of [Hi, Wi], e.g. [[H1,W1], [H2,W2], …]
            # We'll pass that directly into our MultiScaleSequenceConv2d.
            # Note: in_channels = inner_dim, out_channels = inner_dim, groups=inner_dim for depthwise
            self.conv = MultiScaleSequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                multi_seqlens=seqlens,      # e.g. [[H1, W1], [H2, W2], ...]
            )
        else:
            raise NotImplementedError(f"conv_kind '{conv_kind}' is not supported")

        # Then project x_qk_conv_act → Q/K and x_v separately as before
        self.qk_proj = nn.Linear(inner_dim, 2 * inner_dim, bias=proj_bias)
        self.v_proj = nn.Linear(inner_dim, inner_dim, bias=proj_bias)

        # The rest (MatrixLSTMCell, skip, proj_down, norms, FFN, etc.) is unchanged:
        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=num_heads,
            norm_bias=norm_bias,
            eps=1e-5,
            chunk_size=chunk_size,
            gate_soft_cap=gate_soft_cap,
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

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of shape (B, S_total, dim)
        where S_total = sum_i (H_i * W_i) across all scales.
        """
        residual = x
        x = self.norm(x)

        # If we're traversing bottom-right→top-left, flip now:
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        # Up‐project to get Q/K/V
        x_inner = self.proj_up(x)                  # (B, S_total, 2*inner_dim)
        x_qk, x_v = torch.chunk(x_inner, 2, dim=-1)  # each is (B, S_total, inner_dim)

        # **Convolution on x_qk** (now multi‐scale)
        # This applies a separate depthwise 2D‐conv on each [Hi×Wi] slice inside x_qk:
        x_qk_conv_act = F.silu(self.conv(x_qk))     # (B, S_total, inner_dim)

        # Project to Q/K from the conv’d feature, and V from the “x_v” branch
        qk = self.qk_proj(x_qk_conv_act)            # (B, S_total, 2*inner_dim)
        q, k = torch.chunk(qk, 2, dim=-1)            # each (B, S_total, inner_dim)
        v = self.v_proj(x_v)                        # (B, S_total, inner_dim)

        # (You previously reshaped to multi‐head; we keep the same shape for mLSTMCell)
        # Now pass through the MatrixLSTMCell
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)  # (B, S_total, inner_dim)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_qk_conv_act)  # (B, S_total, inner_dim)

        # Project back down to dim
        x_mlstm = self.proj_down(h_tilde_state_skip)     # (B, S_total, dim)

        # Flip back if needed
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x_mlstm = x_mlstm.flip(dims=[1])

        # Residual
        x = residual + x_mlstm

        # FFN block
        ffn_residual = x
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = ffn_residual + x_ffn

        return x

    def reset_parameters(self):
        # Initialize all Linear/Conv/LayerNorm/FFN layers exactly as before
        small_init_(self.proj_up.weight, self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

        small_init_(self.qk_proj.weight, self.dim)
        if self.qk_proj.bias is not None:
            nn.init.zeros_(self.qk_proj.bias)

        small_init_(self.v_proj.weight, self.dim)
        if self.v_proj.bias is not None:
            nn.init.zeros_(self.v_proj.bias)

        wang_init_(self.proj_down.weight, self.dim, num_blocks=self.num_blocks or 1)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)
        self.mlstm_cell.reset_parameters()
        self.norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

# ViLLayer with directionality
@torch.compile
class ViLLayerFusedQKV(nn.Module):
    def __init__(self,
                 dim,
                 expansion=2,
                 qkv_block_size=4,
                 proj_bias=True,
                 norm_bias=True,
                 num_blocks=1,
                 gate_soft_cap=15.0,
                 ffn_proj_factor=2.6667,
                 ffn_round_up_to_multiple_of=64,
                 weight_mode="fused",
                 chunk_size=64,
                 direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
    ):
        super().__init__()
        assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
        self.dim = dim
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.gate_soft_cap = gate_soft_cap
        self.weight_mode = weight_mode
        self.num_blocks = num_blocks
        self.direction = direction

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.inner_dim = inner_dim
        self.num_heads = num_heads

        # Fused linear QKV projection (input -> [Q | K | V])
        self.qkv_proj = nn.Linear(dim, 3 * inner_dim, bias=proj_bias)

        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=num_heads,
            norm_bias=norm_bias,
            eps=1e-5,
            chunk_size=chunk_size,
            gate_soft_cap=gate_soft_cap
        )

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

    @torch.compile
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply directionality: reverse if specified
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(1)  # reverse along sequence axis
        residual = x
        x = self.norm(x)
        qkv = self.qkv_proj(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
        x_mlstm = self.proj_down(h_tilde_state)
        x = residual + x_mlstm
        ffn_residual = x
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = ffn_residual + x_ffn
        # Restore original order if reversed
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(1)
        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight); nn.init.zeros_(self.qkv_proj.bias)
        nn.init.xavier_uniform_(self.proj_down.weight); nn.init.zeros_(self.proj_down.bias)
        self.mlstm_cell.reset_parameters()
        self.norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()


# class RefactoredxLSTMCrossReader(nn.Module):
#     """
#     Cross-attention xLSTM Reader: FusedQKV-like structure, with directionality and strict masking.
#     """
#     def __init__(self,
#                  dim,
#                  expansion=2,
#                  qkv_block_size=32,
#                  proj_bias=True,
#                  norm_bias=True,
#                  num_blocks=1,  # kept for API compatibility, unused
#                  gate_soft_cap=15.0,
#                  chunk_size=64,
#                  direction=SequenceTraversal.ROWWISE_FROM_TOP_LEFT
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.gate_soft_cap = gate_soft_cap
#         self.num_blocks = num_blocks
#         self.direction = direction

#         inner_dim = expansion * dim
#         num_heads = inner_dim // qkv_block_size
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads

#         # Q, K, V projections
#         self.q_proj = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.k_proj = nn.Linear(dim, inner_dim, bias=proj_bias)
#         self.v_proj = nn.Linear(dim, inner_dim, bias=proj_bias)

#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=num_heads,
#             norm_bias=norm_bias,
#             eps=1e-5,
#             chunk_size=chunk_size,
#             gate_soft_cap=gate_soft_cap
#         )

#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
#         self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=2.6667,  # or make this an argument, if desired
#             ffn_round_up_to_multiple_of=64,
#             use_bias=proj_bias,
#             weight_mode="fused",
#             num_blocks=1,
#         )
#         self.reset_parameters()

#     def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
#         # Apply directionality: reverse memory if specified
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             memory = memory.flip(1)
#         x = torch.cat([memory, queries], dim=1)  # (B, N_mem + N_q, D)

#         residual = x
#         x = self.norm(x)

#         q = self.q_proj(x)
#         k = self.k_proj(x)
#         v = self.v_proj(x)

#         N_mem = memory.size(1)
#         # Mask out query positions in k/v (strict cross-attention, only memory writes)
#         if N_mem < x.shape[1]:
#             k[:, N_mem:, :] = 0.0
#             v[:, N_mem:, :] = 0.0

#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
#         x_mlstm = self.proj_down(h_tilde_state)

#         x = residual + x_mlstm
#         ffn_residual = x
#         x_ffn = self.ffn_norm(x)
#         x_ffn = self.ffn(x_ffn)
#         x = ffn_residual + x_ffn

#         # Output only the query portion, without flipping
#         out = x[:, N_mem:, :]
#         return out

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.q_proj.weight); nn.init.zeros_(self.q_proj.bias)
#         nn.init.xavier_uniform_(self.k_proj.weight); nn.init.zeros_(self.k_proj.bias)
#         nn.init.xavier_uniform_(self.v_proj.weight); nn.init.zeros_(self.v_proj.bias)
#         nn.init.xavier_uniform_(self.proj_down.weight); nn.init.zeros_(self.proj_down.bias)
#         self.mlstm_cell.reset_parameters()
#         self.norm.reset_parameters()
#         self.ffn_norm.reset_parameters()
#         self.ffn.reset_parameters()

@torch.compile
class DeformableCrossReader(nn.Module):
    """
    Deformable cross-attention reader for mLSTM head (grid-sampled, per-level sequence conv, multi-scale).
    """
    def __init__(
        self,
        *,
        dim: int,
        expansion: int = 2,
        qkv_block_size: int = 32,
        seqlens_list: list[tuple[int, int]],
        n_points: int = 4,
        proj_bias: bool = True,
        norm_bias: bool = True,
        conv_bias: bool = True,
        conv_kernel_size: int = 3,
        gate_soft_cap: float = 15.0,
        chunk_size: int = 64,
        mlstm_kwargs=None,
    ):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.n_points = n_points
        self.seqlens_list = [tuple(map(int, s)) for s in seqlens_list]
        self.n_levels = len(self.seqlens_list)
        self.inner_dim = expansion * dim
        self.num_heads = self.inner_dim // qkv_block_size

        # Learn offsets per-query per-level
        self.offsets = nn.Linear(dim, self.n_levels * self.n_points * 2)
        # Project context samples (sampled_feats) and query
        self.sampling_proj = nn.Linear(dim, self.inner_dim, bias=proj_bias)
        self.query_proj = nn.Linear(dim, self.inner_dim, bias=proj_bias)
        # QKV projections for sampled context
        self.context_q_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=proj_bias)
        self.context_k_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=proj_bias)
        self.context_v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=proj_bias)

        # Per-level conv: one SequenceConv2d per scale/grid
        self.context_convs = nn.ModuleList([
            SequenceConv2d(
                self.inner_dim,
                self.inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=self.inner_dim,
                bias=conv_bias,
                seqlens=sl,
            )
            for sl in self.seqlens_list
        ])


        # mLSTM cell --------------------------------------------------------------------------
        self.mlstm_cell = MatrixLSTMCell(
            dim=self.inner_dim,
            num_heads=self.num_heads,
            norm_bias=norm_bias,
            eps=1e-3,
            chunk_size=chunk_size,
            gate_soft_cap=gate_soft_cap,
        )

        self.proj_down = nn.Linear(self.inner_dim, dim)
        self.ffn_norm = nn.LayerNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn = FeedForward(
            embedding_dim=dim,
            ffn_proj_factor=2.6667,
            ffn_round_up_to_multiple_of=64,
            use_bias=proj_bias,
            weight_mode="fused",
            num_blocks=1,
        )

    def forward(self, memory, queries, memory_spatial_shapes, memory_level_start_index=None):
        """
        memory: [B, total_mem, dim]
        queries: [B, Nq, dim]
        memory_spatial_shapes: [(H1, W1), (H2, W2), ...]
        memory_level_start_index: [n_levels] (optional)
        """
        B, Nq, D = queries.shape
        n_levels, n_points = self.n_levels, self.n_points

        # Sample offsets
        reference_points = queries.new_full((B, Nq, n_levels, 2), 0.5)
        offsets = self.offsets(queries).view(B, Nq, n_levels, n_points, 2)
        sampling_locations = reference_points.unsqueeze(3) + offsets  # [B, Nq, n_levels, n_points, 2]

        sampled_feats_list = []
        mem_idx = 0
        # Sample grids for each level
        for lvl, (H, W) in enumerate(memory_spatial_shapes):
            num_tokens = H * W
            mem = memory[:, mem_idx:mem_idx+num_tokens, :].view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
            points = sampling_locations[:, :, lvl, :, :]  # [B, Nq, n_points, 2]
            grid = points.clone()
            grid[..., 0] = grid[..., 0] * 2 - 1
            grid[..., 1] = grid[..., 1] * 2 - 1
            grid = grid.view(B * Nq, n_points, 2).unsqueeze(2)
            mem_flat = mem.unsqueeze(1).repeat(1, Nq, 1, 1, 1).view(B * Nq, D, H, W)
            sampled = F.grid_sample(mem_flat, grid, mode='bilinear', align_corners=True)
            sampled = sampled.squeeze(-1).permute(0, 2, 1)
            sampled = sampled.view(B, Nq, n_points, D)
            sampled_feats_list.append(sampled)
            mem_idx += num_tokens

        # Concatenate per-level samples: [B, Nq, total_points, D]
        sampled_feats = torch.cat(sampled_feats_list, dim=2)
        # Project up to inner_dim
        sampled_feats = self.sampling_proj(sampled_feats)

        # --------- PER-LEVEL SEQUENCE CONVOLUTION -----------
        conv_feats_list = []
        start = 0
        for lvl, conv in enumerate(self.context_convs):
            pts = n_points
            lvl_feats = sampled_feats[:, :, start:start+pts, :]
            B_, Nq_, npts_, C_ = lvl_feats.shape
            lvl_feats = lvl_feats.view(B_*Nq_, npts_, C_)  # (B*Nq, pts, C)
            lvl_feats = F.silu(conv(lvl_feats))            # (B*Nq, pts, C)
            lvl_feats = lvl_feats.view(B, Nq, npts_, C_)
            conv_feats_list.append(lvl_feats)
            start += pts
        context = torch.cat(conv_feats_list, dim=2)        # [B, Nq, total_points, inner_dim]

        # --------- QKV PROJECTIONS -----------
        context_q = self.context_q_proj(context)
        context_k = self.context_k_proj(context)
        context_v = self.context_v_proj(context)
        query_q = self.query_proj(queries)                 # [B, Nq, inner_dim]

        # --------- FORM SEQUENCE: [context, query] ---------
        seq_q = torch.cat([context_q, query_q.unsqueeze(2)], dim=2)
        seq_k = torch.cat([context_k, torch.zeros_like(query_q).unsqueeze(2)], dim=2)
        seq_v = torch.cat([context_v, torch.zeros_like(query_q).unsqueeze(2)], dim=2)
        # Reshape for mLSTM: (B*Nq, seq_len, inner_dim)
        seq_q = seq_q.view(B*Nq, -1, self.inner_dim)
        seq_k = seq_k.view(B*Nq, -1, self.inner_dim)
        seq_v = seq_v.view(B*Nq, -1, self.inner_dim)

        # --------- mLSTM ---------
        out = self.mlstm_cell(seq_q, seq_k, seq_v)   # [B*Nq, seq_len, inner_dim]
        out = out[:, -1, :]                          # use last position (the query slot)
        out = self.proj_down(out).view(B, Nq, -1)
        out = self.ffn_norm(out)
        out = self.ffn(out)
        return out



@torch.compile
class DeformableCrossReaderNoConv(nn.Module):
    """
    Deformable cross-attention reader for mLSTM head, using fused QKV projection.
    Library-consistent, set-mixer style (no conv over sampled points).
    """
    def __init__(
        self,
        dim,
        expansion=2,
        qkv_block_size=32,
        n_points=4,
        seqlens_list=None,  # list of (H, W)
        proj_bias=True,
        norm_bias=True,
        gate_soft_cap=15.0,
        ffn_proj_factor=2.6667,
        ffn_round_up_to_multiple_of=64,
        weight_mode="fused",
        chunk_size=64,
    ):
        super().__init__()
        self.dim = dim
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.n_points = n_points
        self.seqlens_list = seqlens_list
        self.n_levels = len(seqlens_list) if seqlens_list is not None else 0
        self.inner_dim = expansion * dim
        self.num_heads = self.inner_dim // qkv_block_size

        # Deformable offsets per-query
        self.offsets = nn.Linear(dim, self.n_levels * n_points * 2)

        # Up-project sampled context
        self.proj_up = nn.Linear(dim, self.inner_dim, bias=proj_bias)

        # Fused QKV projection for context
        self.context_qkv_proj = nn.Linear(self.inner_dim, 3 * self.inner_dim, bias=proj_bias)
        # Q projection for query token
        self.query_proj = nn.Linear(dim, self.inner_dim, bias=proj_bias)

        # mLSTM cell
        self.mlstm_cell = MatrixLSTMCell(
            dim=self.inner_dim,
            num_heads=self.num_heads,
            norm_bias=norm_bias,
            eps=1e-5,
            chunk_size=chunk_size,
            gate_soft_cap=gate_soft_cap
        )

        self.learnable_skip = nn.Parameter(torch.ones(self.inner_dim))
        self.proj_down = nn.Linear(self.inner_dim, dim, bias=proj_bias)

        self.norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn = FeedForward(
            embedding_dim=dim,
            ffn_proj_factor=ffn_proj_factor,
            ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
            use_bias=proj_bias,
            weight_mode=weight_mode,
            num_blocks=1,
        )
        self.reset_parameters()

    @torch.compile
    def forward(self, memory, queries, memory_spatial_shapes, memory_level_start_index=None):
        """
        memory: [B, total_tokens, dim]
        queries: [B, Nq, dim]
        memory_spatial_shapes: [(H1, W1), ...]
        """
        residual = queries
        x = self.norm(queries)

        B, Nq, D = x.shape
        n_levels, n_points = self.n_levels, self.n_points

        # ---- Deformable Sampling ----
        reference_points = x.new_full((B, Nq, n_levels, 2), 0.5)
        offsets = self.offsets(x).view(B, Nq, n_levels, n_points, 2)
        sampling_locations = reference_points.unsqueeze(3) + offsets  # [B, Nq, n_levels, n_points, 2]

        sampled_feats_list = []
        mem_idx = 0
        for lvl, (H, W) in enumerate(memory_spatial_shapes):
            num_tokens = H * W
            mem = memory[:, mem_idx:mem_idx+num_tokens, :].view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
            points = sampling_locations[:, :, lvl, :, :]  # [B, Nq, n_points, 2]
            grid = points.clone()
            grid[..., 0] = grid[..., 0] * 2 - 1
            grid[..., 1] = grid[..., 1] * 2 - 1
            grid = grid.view(B * Nq, n_points, 2).unsqueeze(2)
            mem_flat = mem.unsqueeze(1).expand(-1, Nq, -1, -1, -1).reshape(B * Nq, D, H, W)
            sampled = F.grid_sample(mem_flat, grid, mode='bilinear', align_corners=True)
            sampled = sampled.squeeze(-1).permute(0, 2, 1)        # [B*Nq, n_points, D]
            sampled = sampled.view(B, Nq, n_points, D)
            sampled_feats_list.append(sampled)
            mem_idx += num_tokens

        # Concatenate all sampled points: [B, Nq, n_levels*n_points, D]
        sampled_feats = torch.cat(sampled_feats_list, dim=2)
        x_context = F.silu(self.proj_up(sampled_feats))  # [B, Nq, total_points, inner_dim]

        # --- Fused QKV projection for context ---
        context_qkv = self.context_qkv_proj(x_context)   # [B, Nq, total_points, 3*inner_dim]
        context_q, context_k, context_v = torch.chunk(context_qkv, 3, dim=-1)

        # --- Query Q projection ---
        query_q = self.query_proj(x)                     # [B, Nq, inner_dim]

        # --- mLSTM input sequence: [context, query] ---
        seq_q = torch.cat([context_q, query_q.unsqueeze(2)], dim=2)
        seq_k = torch.cat([context_k, torch.zeros_like(query_q).unsqueeze(2)], dim=2)
        seq_v = torch.cat([context_v, torch.zeros_like(query_q).unsqueeze(2)], dim=2)
        seq_q = seq_q.view(B*Nq, -1, self.inner_dim)
        seq_k = seq_k.view(B*Nq, -1, self.inner_dim)
        seq_v = seq_v.view(B*Nq, -1, self.inner_dim)

        # --- mLSTM ---
        h_tilde_state = self.mlstm_cell(q=seq_q, k=seq_k, v=seq_v)
        h_tilde_state = h_tilde_state[:, -1, :]  # take last token (query slot)
        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * query_q.view(B*Nq, -1))
        x_mlstm = self.proj_down(h_tilde_state_skip).view(B, Nq, -1)

        # Residual & FFN (ViLLayer style)
        x = residual + x_mlstm
        ffn_residual = x
        x_ffn = self.ffn_norm(x)
        x_ffn = self.ffn(x_ffn)
        x = ffn_residual + x_ffn

        return x

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.proj_up.weight)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)
        nn.init.xavier_uniform_(self.context_qkv_proj.weight)
        if self.context_qkv_proj.bias is not None:
            nn.init.zeros_(self.context_qkv_proj.bias)
        nn.init.xavier_uniform_(self.query_proj.weight)
        if self.query_proj.bias is not None:
            nn.init.zeros_(self.query_proj.bias)
        nn.init.xavier_uniform_(self.proj_down.weight)
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)
        nn.init.ones_(self.learnable_skip)
        self.mlstm_cell.reset_parameters()
        self.norm.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()



# =============================================================================
#                        MULTI‑SCALE CROSS‑READER
# =============================================================================
@torch.compile
class RefactoredxLSTMCrossReader(nn.Module):
    """Same xLSTM reader, *now* able to ingest a **concatenated** multi‑scale
    memory and internally apply an independent depth‑wise convolution to each
    FPN level.
    """

    def __init__(
        self,
        *,
        dim: int,
        expansion: int = 2,
        qkv_block_size: int = 32,
        proj_bias: bool = True,
        norm_bias: bool = True,
        conv_bias: bool = True,
        conv_kernel_size: int = 3,
        seqlens_list: list[tuple[int, int]],  # NEW: [(H,W), ...] per memory slice
        gate_soft_cap: float = 15.0,
        chunk_size: int = 64,
        memory_direction=None,
        query_direction=None,
    ) -> None:
        super().__init__()
        assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"

        self.dim = dim
        self.expansion = expansion
        self.qkv_block_size = qkv_block_size
        self.memory_direction = memory_direction
        self.query_direction = query_direction
        self.seqlens_list = [tuple(map(int, s)) for s in seqlens_list]
        self.mem_lengths = [h * w for h, w in self.seqlens_list]

        self.inner_dim = self.expansion * self.dim
        self.num_heads = self.inner_dim // self.qkv_block_size

        # projections -------------------------------------------------------------------------
        self.memory_proj_up = nn.Linear(dim, 2 * self.inner_dim, bias=proj_bias)
        self.memory_qk_proj = nn.Linear(self.inner_dim, 2 * self.inner_dim, bias=proj_bias)
        self.memory_v_proj = nn.Linear(self.inner_dim, self.inner_dim, bias=proj_bias)
        self.query_q_proj = nn.Linear(dim, self.inner_dim, bias=proj_bias)

        # per‑scale convolution list ----------------------------------------------------------
        self.memory_convs = nn.ModuleList([
            SequenceConv2d(
                self.inner_dim,
                self.inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=self.inner_dim,
                bias=conv_bias,
                seqlens=sl,
            )
            for sl in self.seqlens_list
        ])

        # mLSTM cell --------------------------------------------------------------------------
        self.mlstm_cell = MatrixLSTMCell(
            dim=self.inner_dim,
            num_heads=self.num_heads,
            norm_bias=norm_bias,
            eps=1e-3,
            chunk_size=chunk_size,
            gate_soft_cap=gate_soft_cap,
        )

        # output & ffn ------------------------------------------------------------------------
        self.proj_down = nn.Linear(self.inner_dim, dim, bias=proj_bias)
        self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
        self.ffn = FeedForward(
            embedding_dim=dim,
            ffn_proj_factor=2.6667,
            ffn_round_up_to_multiple_of=64,
            use_bias=proj_bias,
            weight_mode="fused",
            num_blocks=1,
        )
        self.reset_parameters()

    @torch.compile
    # ----------------------------------- FORWARD ----------------------------------
    def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
        B, N_mem, D = memory.shape
        Bq, N_q,  _ = queries.shape
        assert B == Bq and D == self.dim

        flip_mem = self.memory_direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT
        flip_qry = self.query_direction  == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT

        # ---- query branch ---------------------------------------------------------
        if flip_qry:
            queries = queries.flip(1)                     # reverse order before Q-projection
        query_q = self.query_q_proj(queries)              # (B, N_q, inner_dim)

        # ---- memory branch --------------------------------------------------------
        mem_up   = self.memory_proj_up(memory)            # (B, N_mem, 2*inner_dim)
        mem_qk, mem_v = torch.chunk(mem_up, 2, dim=-1)    # both still (B, N_mem, inner_dim)

        # per-scale DW-conv (expects canonical order!) ------------------------------
        conv_parts, start = [], 0
        for conv, seg_len in zip(self.memory_convs, self.mem_lengths):
            seg = mem_qk[:, start : start + seg_len]      # (B, seg_len, C)
            conv_parts.append(F.silu(conv(seg)))          # depth-wise 2-D conv
            start += seg_len
        mem_qk_conv = torch.cat(conv_parts, 1)            # (B, N_mem, inner_dim)

        # ---- now apply *optional* flip -------------------------------------------
        if flip_mem:
            mem_qk_conv = mem_qk_conv.flip(1)
            mem_v       = mem_v.flip(1)

        mem_qk_proj = self.memory_qk_proj(mem_qk_conv)    # (B, N_mem, 2*inner_dim)
        mem_q, mem_k = torch.chunk(mem_qk_proj, 2, dim=-1)
        mem_v = self.memory_v_proj(mem_v)                 # (B, N_mem, inner_dim)

        # ---- concat & mLSTM -------------------------------------------------------
        q = torch.cat([mem_q, query_q], 1)
        k = torch.cat([mem_k, torch.zeros_like(query_q)], 1)
        v = torch.cat([mem_v, torch.zeros_like(query_q)], 1)

        h_tilde = self.mlstm_cell(q=q, k=k, v=v)          # (B, N_mem+N_q, inner_dim)
        x = self.proj_down(h_tilde)                       # (B, N_mem+N_q, dim)
        out_q = x[:, N_mem:]                              # keep queries only

        # flip *back* if we flipped the queries in ----------------------------------
        if flip_qry:
            out_q = out_q.flip(1)

        # FFN -----------------------------------------------------------------------
        out_q = self.ffn_norm(out_q)
        out_q = self.ffn(out_q)
        return out_q
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.memory_proj_up.weight); nn.init.zeros_(self.memory_proj_up.bias)
        nn.init.xavier_uniform_(self.memory_qk_proj.weight); nn.init.zeros_(self.memory_qk_proj.bias)
        nn.init.xavier_uniform_(self.memory_v_proj.weight); nn.init.zeros_(self.memory_v_proj.bias)
        nn.init.xavier_uniform_(self.query_q_proj.weight); nn.init.zeros_(self.query_q_proj.bias)
        nn.init.xavier_uniform_(self.proj_down.weight); nn.init.zeros_(self.proj_down.bias)
        self.mlstm_cell.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()




# #zeroing K,V on queries for "read only"
# #10 MAP @ 50
# class RefactoredxLSTMCrossReader(nn.Module):
#     """
#     Cross-attention xLSTM Reader with:
#     - Convolutional processing of memory tokens (spatial grid, e.g. image patches)
#     - Strict cross-attention: queries read from memory, but not each other
#     - FFN is applied only to the query outputs
#     - Supports independent direction (traversal order) for both memory and query
#     """
#     def __init__(
#         self,
#         dim,
#         expansion=2,
#         qkv_block_size=32,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=3,
#         conv_kind="2d",
#         seqlens=None,  # Needed for SequenceConv2d
#         gate_soft_cap=15.0,
#         chunk_size=64,
#         memory_direction=None,
#         query_direction=None,
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.gate_soft_cap = gate_soft_cap
#         self.memory_direction = memory_direction
#         self.query_direction = query_direction

#         inner_dim = expansion * dim
#         num_heads = inner_dim // qkv_block_size
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads

#         # Projections for memory
#         self.memory_proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

#         # Convolution for memory (Q/K only, ViL style)
#         assert conv_kind == "2d", "Only 2D convolution is implemented in this example."
#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd for 2d conv"
#         self.memory_conv = SequenceConv2d(
#             inner_dim, inner_dim, kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2, groups=inner_dim,
#             bias=conv_bias, seqlens=seqlens
#         )

#         self.memory_qk_proj = nn.Linear(inner_dim, 2 * inner_dim, bias=proj_bias)
#         self.memory_v_proj = nn.Linear(inner_dim, inner_dim, bias=proj_bias)

#         # Query projections (NO convolution for queries, since they're not spatial)
#         self.query_q_proj = nn.Linear(dim, inner_dim, bias=proj_bias)
        

#         # mLSTM cell (fused QKV logic)
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=num_heads,
#             norm_bias=norm_bias,
#             eps=1e-3,
#             chunk_size=chunk_size,
#             gate_soft_cap=gate_soft_cap,
#         )

#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
#         self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=2.6667,
#             ffn_round_up_to_multiple_of=64,
#             use_bias=proj_bias,
#             weight_mode="fused",
#             num_blocks=1,
#         )
#         self.reset_parameters()

#     def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
#         """
#         memory: (B, N_mem, D)  # Spatial grid, e.g. flattened H*W patches
#         queries: (B, N_q, D)   # Query features (no spatial structure)
#         Returns:
#             out: (B, N_q, D)
#         """
#         B, N_mem, D = memory.shape
#         Bq, N_q, Dq = queries.shape
#         assert B == Bq and D == Dq

#         # ---- DIRECTIONALITY ----
#         # Flip memory if specified
#         if self.memory_direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             memory = memory.flip(1)
#         # Flip queries if specified
#         if self.query_direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             queries = queries.flip(1)

#         # 1. Project up and split memory into QK/V branches
#         memory_proj = self.memory_proj_up(memory)  # (B, N_mem, 2*inner_dim)
#         memory_qk, memory_v = torch.chunk(memory_proj, 2, dim=-1)  # (B, N_mem, inner_dim)

#         # 2. Convolutional activation (ViL style) on memory QK only
#         memory_qk_conv = F.silu(self.memory_conv(memory_qk))  # (B, N_mem, inner_dim)

#         # 3. Final QK/V projections for memory
#         mem_qk_proj = self.memory_qk_proj(memory_qk_conv)  # (B, N_mem, 2*inner_dim)
#         mem_q, mem_k = torch.chunk(mem_qk_proj, 2, dim=-1)
#         mem_v = self.memory_v_proj(memory_v)

#         # 4. Prepare queries: only Q projection, no convolution
#         query_q = self.query_q_proj(queries)  # (B, N_q, inner_dim)

#         # 5. Concatenate memory and queries for joint mLSTM processing
#         q = torch.cat([mem_q, query_q], dim=1)  # (B, N_mem + N_q, inner_dim)
#         k = torch.cat([mem_k, torch.zeros_like(query_q)], dim=1)  # Zero-out K for queries
#         v = torch.cat([mem_v, torch.zeros_like(query_q)], dim=1)  # Zero-out V for queries

#         # 6. Run through mLSTM cell
#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
#         x_mlstm = self.proj_down(h_tilde_state)

#         # 7. Output only the query portion (last N_q tokens)
#         query_out = x_mlstm[:, N_mem:, :]  # (B, N_q, D)

#         # Flip query output back if needed
#         if self.query_direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             query_out = query_out.flip(1)

#         # 8. Apply FFN and normalization only to query tokens
#         query_out = self.ffn_norm(query_out)
#         query_out = self.ffn(query_out)

#         return query_out

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.memory_proj_up.weight); nn.init.zeros_(self.memory_proj_up.bias)
#         nn.init.xavier_uniform_(self.memory_qk_proj.weight); nn.init.zeros_(self.memory_qk_proj.bias)
#         nn.init.xavier_uniform_(self.memory_v_proj.weight); nn.init.zeros_(self.memory_v_proj.bias)
#         nn.init.xavier_uniform_(self.query_q_proj.weight); nn.init.zeros_(self.query_q_proj.bias)
#         nn.init.xavier_uniform_(self.proj_down.weight); nn.init.zeros_(self.proj_down.bias)
#         self.mlstm_cell.reset_parameters()
#         self.ffn_norm.reset_parameters()
#         self.ffn.reset_parameters()


# class RefactoredxLSTMCrossReader(nn.Module):
#     """
#     Cross-attention xLSTM Reader with convolutional processing of memory tokens (spatial grid)
#     and strict cross-attention from query tokens to memory tokens only. FFN is applied only to
#     the query outputs for efficiency.
#     """
#     def __init__(
#         self,
#         dim,
#         expansion=2,
#         qkv_block_size=32,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=3,
#         conv_kind="2d",
#         seqlens=None,  # Needed for SequenceConv2d
#         gate_soft_cap=15.0,
#         chunk_size=64,
#         direction=None,
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.expansion = expansion
#         self.qkv_block_size = qkv_block_size
#         self.gate_soft_cap = gate_soft_cap
#         self.direction = direction

#         inner_dim = expansion * dim
#         num_heads = inner_dim // qkv_block_size
#         self.inner_dim = inner_dim
#         self.num_heads = num_heads

#         # Projections for memory
#         self.memory_proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

#         # Convolution for memory (Q/K only, ViL style)
#         assert conv_kind == "2d", "Only 2D convolution is implemented in this example."
#         assert conv_kernel_size % 2 == 1, "conv_kernel_size must be odd for 2d conv"
#         self.memory_conv = SequenceConv2d(
#             inner_dim, inner_dim, kernel_size=conv_kernel_size,
#             padding=conv_kernel_size // 2, groups=inner_dim,
#             bias=conv_bias, seqlens=seqlens
#         )

#         self.memory_qk_proj = nn.Linear(inner_dim, 2 * inner_dim, bias=proj_bias)
#         self.memory_v_proj = nn.Linear(inner_dim, inner_dim, bias=proj_bias)

#         # Query projections (NO convolution for queries, since they're not spatial)
#         self.query_q_proj = nn.Linear(dim, inner_dim, bias=proj_bias)

#         # mLSTM cell (fused QKV logic)
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=num_heads,
#             norm_bias=norm_bias,
#             eps=1e-5,
#             chunk_size=chunk_size,
#             gate_soft_cap=gate_soft_cap,
#         )

#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)
#         self.ffn_norm = nn.RMSNorm(dim, eps=1e-6, elementwise_affine=norm_bias)
#         self.ffn = FeedForward(
#             embedding_dim=dim,
#             ffn_proj_factor=2.6667,
#             ffn_round_up_to_multiple_of=64,
#             use_bias=proj_bias,
#             weight_mode="fused",
#             num_blocks=1,
#         )
#         self.reset_parameters()

#     def forward(self, memory: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
#         """
#         memory: (B, N_mem, D)  # Spatial grid, e.g. flattened H*W patches, with known arrangement
#         queries: (B, N_q, D)   # Query features (no spatial structure)
#         Returns:
#             out: (B, N_q, D)
#         """
#         B, N_mem, D = memory.shape
#         Bq, N_q, Dq = queries.shape
#         assert B == Bq and D == Dq

#         # Handle directionality for memory if needed
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             memory = memory.flip(1)

#         # 1. Project up and split memory into QK/V branches
#         memory_proj = self.memory_proj_up(memory)  # (B, N_mem, 2*inner_dim)
#         memory_qk, memory_v = torch.chunk(memory_proj, 2, dim=-1)  # (B, N_mem, inner_dim)

#         # 2. Convolutional activation (ViL style) **on memory QK only**
#         memory_qk_conv = F.silu(self.memory_conv(memory_qk))  # (B, N_mem, inner_dim)

#         # 3. Final QK/V projections for memory
#         mem_qk_proj = self.memory_qk_proj(memory_qk_conv)  # (B, N_mem, 2*inner_dim)
#         mem_q, mem_k = torch.chunk(mem_qk_proj, 2, dim=-1)
#         mem_v = self.memory_v_proj(memory_v)

#         # 4. Prepare queries: only Q projection, no convolution
#         query_q = self.query_q_proj(queries)  # (B, N_q, inner_dim)

#         # 5. Concatenate memory and queries for joint mLSTM processing
#         q = torch.cat([mem_q, query_q], dim=1)  # (B, N_mem + N_q, inner_dim)
#         k = torch.cat([mem_k, torch.zeros_like(query_q)], dim=1)  # Zero-out K for queries
#         v = torch.cat([mem_v, torch.zeros_like(query_q)], dim=1)  # Zero-out V for queries

#         # 6. Run through mLSTM cell
#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)
#         x_mlstm = self.proj_down(h_tilde_state)

#         # 7. Output only the query portion (last N_q tokens)
#         query_out = x_mlstm[:, N_mem:, :]  # (B, N_q, D)

#         # 8. Apply FFN and normalization only to query tokens
#         query_out = self.ffn_norm(query_out)
#         query_out = self.ffn(query_out)

#         return query_out

#     def reset_parameters(self):
#         nn.init.xavier_uniform_(self.memory_proj_up.weight); nn.init.zeros_(self.memory_proj_up.bias)
#         nn.init.xavier_uniform_(self.memory_qk_proj.weight); nn.init.zeros_(self.memory_qk_proj.bias)
#         nn.init.xavier_uniform_(self.memory_v_proj.weight); nn.init.zeros_(self.memory_v_proj.bias)
#         nn.init.xavier_uniform_(self.query_q_proj.weight); nn.init.zeros_(self.query_q_proj.bias)
#         nn.init.xavier_uniform_(self.proj_down.weight); nn.init.zeros_(self.proj_down.bias)
#         self.mlstm_cell.reset_parameters()
#         self.ffn_norm.reset_parameters()
#         self.ffn.reset_parameters()



#GPT 4.5 refactor
class ViLBlock(nn.Module):
    def __init__(self,
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
        qkv_block_size = 4):


        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.norm_bias = norm_bias

        #self.drop_path = DropPath(drop_prob=0.0)
        #self.norm = nn.RMSNorm(dim, eps=1e-3)
        self.layer = ViLLayer(dim,
                                direction,
                                qkv_block_size=qkv_block_size,
                                proj_bias=True,
                                norm_bias=True,
                                conv_bias=True,
                                conv_kernel_size=3,
                                conv_kind=conv_kind,
                                init_weights="original",
                                seqlens=seqlens,  # Initial seqlens, can be overridden in forward
                                num_blocks=None,
                                chunk_size=chunk_size,
                            )

        self.reset_parameters()

    def _forward_path(self, x):
        #x = self.norm(x)
        x = self.layer(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x = self.drop_path(x, self._forward_path)
        x = self.layer(x)
        return x

    def reset_parameters(self):
        self.layer.reset_parameters()
        #self.norm.reset_parameters()

#grok backendsclass 
from mlstm_kernels.torch.chunkwise.triton_xl_chunk import mlstm_chunkwise__xl_chunk
class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True, eps=1e-6, chunk_size=16, use_autocast=True, autocast_dtype=torch.float16, gate_soft_cap=15.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_autocast = use_autocast
        self.autocast_dtype = autocast_dtype
        self.gate_soft_cap = gate_soft_cap  # Cap value for soft capping

        # Fused ifgate projection
        self.ifgate = nn.Linear(3 * dim, 2 * num_heads)

        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias, eps=eps)
        self.causal_mask_cache = {}
        self.chunk_size = chunk_size

        # CPU-compatible backend configuration (remains float16)
        self.cpu_backend_config_infer = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            chunk_size=int(chunk_size),
            autocast_kernel_dtype="float16",
            return_last_states=False,
            mode="inference",
            eps=5e-5
        )
        self.cpu_backend_infer = mLSTMBackend(
            config=self.cpu_backend_config_infer,
        )

        # GPU-compatible (Triton) backend configuration
        self.gpu_backend_config_infer = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
            sequence_kernel="native_sequence__triton",
            step_kernel="triton",
            chunk_size=int(chunk_size),
            autocast_kernel_dtype="float16",
            return_last_states=False,
            mode="inference",
            eps=5e-5
        )
        self.gpu_backend_infer = mLSTMBackend(
            config=self.gpu_backend_config_infer,
        )

        # CPU-compatible backend configuration (remains float16)
        self.cpu_backend_config = mLSTMBackendConfig(
            chunkwise_kernel="chunkwise--native_autograd",
            sequence_kernel="native_sequence__native",
            step_kernel="native",
            chunk_size=int(chunk_size),
            autocast_kernel_dtype="float16",
            return_last_states=False,
            mode="train_with_padding",
            eps=5e-5
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
            autocast_kernel_dtype="float16",
            return_last_states=False,
            mode="train_with_padding",
            eps=5e-5
        )
        self.gpu_backend = mLSTMBackend(
            config=self.gpu_backend_config,
        )

        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, H = q.shape  # (B, S, H)
        dtype = q.dtype
        device = q.device

        if not (q.device == k.device == v.device):
            raise ValueError("All input tensors (q, k, v) must be on the same device.")
        backend = self.gpu_backend if device.type == 'cuda' else self.cpu_backend

        # Prepare gate inputs
        if_gate_input = torch.cat([q, k, v], dim=-1)  # (B, S, 3 * H)
        
        # Compute fused ifgate preactivations and apply soft_cap
        if_preact = self.ifgate(if_gate_input)  # (B, S, 2 * NH)
        if_preact_capped = self.soft_cap(if_preact, self.gate_soft_cap)  # Cap the preactivations
        
        # Split into input and forget gate preactivations
        i_preact, f_preact = torch.chunk(if_preact_capped, 2, dim=-1)  # Each (B, S, NH)
        
        # Transpose for backend
        i = i_preact.transpose(-1, -2)  # (B, NH, S)
        f = f_preact.transpose(-1, -2)  # (B, NH, S)

        # Reshape for backend
        q = q.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, head_dim)
        k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
        v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

        # Compute h_state with the selected backend, applying autocast if specified
        if device.type == 'cuda' and self.use_autocast and self.training:
            q = q.to(self.autocast_dtype)
            k = k.to(self.autocast_dtype)
            v = v.to(self.autocast_dtype)
            i = i.to(self.autocast_dtype)
            f = f.to(self.autocast_dtype)
            h_state = backend(q=q, k=k, v=v, i=i, f=f)
        elif device.type == 'cpu' and self.training:
            h_state = backend(q=q, k=k, v=v, i=i, f=f)
        elif device.type == 'cuda' and self.use_autocast and not self.training:
            q = q.to(self.autocast_dtype)
            k = k.to(self.autocast_dtype)
            v = v.to(self.autocast_dtype)
            i = i.to(self.autocast_dtype)
            f = f.to(self.autocast_dtype)
            h_state = backend(q=q, k=k, v=v, i=i, f=f)
        elif device.type == 'cpu' and not self.training:
            h_state = backend(q=q, k=k, v=v, i=i, f=f)

        h_state = h_state.to(dtype)
        h_state_norm = self.outnorm(h_state)  # (B, NH, S, head_dim)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, S, H)

        return h_state_norm

    def soft_cap(self, x, cap_value):
        return cap_value * torch.tanh(x / cap_value)

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # Initialize fused ifgate
        torch.nn.init.zeros_(self.ifgate.weight)  # Shape: (2 * num_heads, 3 * dim)
        
        # Create the bias tensor
        i_bias = torch.full((self.num_heads,), -10, dtype=self.ifgate.bias.dtype, device=self.ifgate.bias.device)
        f_bias = torch.linspace(3.0, 6.0, steps=self.num_heads, dtype=self.ifgate.bias.dtype, device=self.ifgate.bias.device)
        bias = torch.cat([i_bias, f_bias], dim=0)
        
        # Assign to bias.data
        self.ifgate.bias.data = bias



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
from .mlstm_large import mLSTMLayerVision
from .mlstm_large import VilLayerUpdated

import torch.utils.checkpoint as cp

class ViLBlockPair(nn.Module):
    def __init__(self,
                 dim,
                 drop_path=0.0,
                 conv_kind="2d",
                 conv_kernel_size=3,
                 proj_bias=True,
                 norm_bias=True,
                 seqlens=None,
                 num_blocks=15,
                 init_weights="original",
                 chunk_size=256,
                 qkv_block_size=4):
        super().__init__()

        self.seqlens = seqlens                    # save for the forward test
        self.ckpt_thresh = 100 * 100             # 25 600 tokens
        # -------------------------------------------------------------------
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
            qkv_block_size=qkv_block_size,
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
            qkv_block_size=qkv_block_size,
        )

    # -----------------------------------------------------------------------
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rowwise_from_top_left(x)
        x = self.rowwise_from_bot_right(x)
        return x

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        """
        Runs the pair normally *unless* the token count exceeds 160²
        (≈25 600).  In that case, and only while training, we wrap the
        computation in torch.checkpoint to trade compute for memory.
        """
        # decide sequence length
        S = (self.seqlens[0] * self.seqlens[1]) if self.seqlens \
            else (seqlens[0] * seqlens[1]) if seqlens \
            else x.shape[1]

        need_ckpt = (
            self.training                 # only in train mode
            and x.requires_grad           # grads needed
            and S >= self.ckpt_thresh      # token count threshold
        )

        if need_ckpt:
            return cp.checkpoint(self._forward_impl, x, use_reentrant=False)

        return self._forward_impl(x)


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
