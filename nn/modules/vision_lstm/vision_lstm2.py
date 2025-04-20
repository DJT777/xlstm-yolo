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
 

# # # original
class MatrixLSTMCell(nn.Module):
    def __init__(self, dim, num_heads, norm_bias=True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        self.igate = nn.Linear(3 * dim, num_heads)
        self.fgate = nn.Linear(3 * dim, num_heads)
        self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)
        self.causal_mask_cache = {}
        self.reset_parameters()

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        B, S, _ = q.shape  # (B, S, H)

        if_gate_input = torch.cat([q, k, v], dim=-1)
        q = q.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        k = k.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)
        v = v.view(B, S, self.num_heads, -1)  # (B, S, NH, DH)

        q = q.transpose(1, 2)  # (B, NH, S, DH)
        k = k.transpose(1, 2)  # (B, NH, S, DH)
        v = v.transpose(1, 2)  # (B, NH, S, DH)

        # compute input and forget gate pre-activations
        igate_preact = self.igate(if_gate_input)  # (B, S, NH)
        igate_preact = igate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)
        fgate_preact = self.fgate(if_gate_input)  # (B, S, NH)
        fgate_preact = fgate_preact.transpose(-1, -2).unsqueeze(-1)  # (B, NH, S, 1)#

        # cache causal mask to avoid memory allocation in every iteration
        if S in self.causal_mask_cache:
            causal_mask = self.causal_mask_cache[(S, str(q.device))]
        else:
            causal_mask = torch.tril(torch.ones(S, S, dtype=torch.bool, device=q.device))
            self.causal_mask_cache[(S, str(q.device))] = causal_mask

        h_state = parallel_stabilized_simple(
            queries=q,
            keys=k,
            values=v,
            igate_preact=igate_preact,
            fgate_preact=fgate_preact,
            lower_triangular_matrix=causal_mask,
        )  # (B, NH, S, DH)

        h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
        h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, -1)  # (B, NH, S, DH) -> (B, S, NH, DH) -> (B, S, H)

        return h_state_norm

    def reset_parameters(self):
        self.outnorm.reset_parameters()
        # forget gate initialization
        torch.nn.init.zeros_(self.fgate.weight)
        bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.igate.weight)
        torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)


# class MatrixLSTMCell(nn.Module):
#     def __init__(self, dim, num_heads, norm_bias=True, chunk_size=64):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads

#         # Gate projections
#         self.igate = nn.Linear(3 * dim, num_heads)
#         self.fgate = nn.Linear(3 * dim, num_heads)
#         self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)

#         # CPU-compatible backend configuration (remains float32)
#         self.cpu_backend_config = mLSTMBackendConfig(
#             chunkwise_kernel="chunkwise--native_autograd",
#             sequence_kernel="native_sequence__native",
#             step_kernel="native",
#             chunk_size=chunk_size,
#             autocast_kernel_dtype="float32",
#             return_last_states=False,
#             mode="train"
#         )
#         self.cpu_backend = mLSTMBackend(self.cpu_backend_config)

#         # GPU-compatible (Triton) backend configuration — use float16 for AMP
#         self.gpu_backend_config = mLSTMBackendConfig(
#             chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
#             sequence_kernel="native_sequence__triton",
#             step_kernel="triton",
#             chunk_size=chunk_size,
#             autocast_kernel_dtype="float32",  # changed from float32 to float16
#             return_last_states=False,
#             mode="train"
#         )
#         self.gpu_backend = mLSTMBackend(self.gpu_backend_config)

#         # Internal states
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#         self.reset_parameters()

#     def get_gpu_backend(self, device):
#         if self.gpu_backend is None:
#             if not torch.cuda.is_available():
#                 raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
#             self.gpu_backend = mLSTMBackend(self.gpu_backend_config).to(device)
#         return self.gpu_backend

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         B, S, H = q.shape  # (B, S, H)

#         # All inputs must reside on same device
#         if not (q.device == k.device == v.device):
#             raise ValueError("All input tensors (q, k, v) must be on the same device.")
#         device = q.device
#         backend = self.get_gpu_backend(device) if device.type == 'cuda' else self.cpu_backend
#         backend = self.gpu_backend

#         # Prepare gate inputs
#         if_gate_input = torch.cat([q, k, v], dim=-1)
#         i = self.igate(if_gate_input).transpose(-1, -2)  # (B, NH, S)
#         f = self.fgate(if_gate_input).transpose(-1, -2)  # (B, NH, S)

#         # Reshape for backend
#         q = q.view(B, S, self.num_heads, -1).transpose(1, 2)
#         k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
#         v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

#         # Execute backend kernel
#         h_state = backend(
#             q=q, k=k, v=v, i=i, f=f,
#             return_last_states=False,
#             mode="train"
#         )

#         # Reset states
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#         # Normalize and reshape output
#         h_norm = self.outnorm(h_state)
#         return h_norm.transpose(1, 2).reshape(B, S, H)

#     def reset_states(self):
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.fgate.weight)
#         bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
#         torch.nn.init.zeros_(self.igate.weight)
#         torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
#         self.outnorm.reset_parameters()


# #mixed precision MLSTM
# class MatrixLSTMCell(nn.Module):
#     def __init__(self, dim, num_heads, norm_bias=True, chunk_size=256):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads

#         # Gate projections
#         self.igate = nn.Linear(3 * dim, num_heads)
#         self.fgate = nn.Linear(3 * dim, num_heads)
#         self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)

#         # CPU backend config (float32 precision)
#         self.cpu_backend_config = mLSTMBackendConfig(
#             chunkwise_kernel="chunkwise--native_autograd",
#             sequence_kernel="native_sequence__native",
#             step_kernel="native",
#             chunk_size=64,
#             autocast_kernel_dtype="float32",
#             return_last_states=False,
#             mode="train"
#         )
#         self.cpu_backend = mLSTMBackend(self.cpu_backend_config)

#         # GPU backend config (dtype will be set at runtime)
#         self.gpu_backend_config = mLSTMBackendConfig(
#             chunkwise_kernel="chunkwise--triton_xl_chunk",
#             sequence_kernel="native_sequence__triton",
#             step_kernel="triton",
#             chunk_size=64,
#             autocast_kernel_dtype="float32",  # placeholder
#             return_last_states=False,
#             mode="train"
#         )
#         self.gpu_backend = None
#         self.reset_parameters()

#     def get_gpu_backend(self, device):
#         # (Re)instantiate backend to pick up updated dtype
#         if not torch.cuda.is_available():
#             raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
#         self.gpu_backend = mLSTMBackend(self.gpu_backend_config).to(device)
#         return self.gpu_backend

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         B, S, H = q.shape  # (B, S, H)

#         # All inputs must be on the same device
#         device = q.device
#         if not (q.device == k.device == v.device):
#             raise ValueError("All inputs must be on the same device.")

#         # 1) Determine kernel dtype from actual tensor dtype
#         kernel_dtype = "float16" if q.dtype == torch.float16 else "float32"
#         self.gpu_backend_config.autocast_kernel_dtype = kernel_dtype

#         # 2) Select proper backend
#         if device.type == 'cuda':
#             backend = self.get_gpu_backend(device)
#         else:
#             backend = self.cpu_backend

#         # Prepare gate inputs
#         if_gate_input = torch.cat([q, k, v], dim=-1)
#         i = self.igate(if_gate_input).transpose(-1, -2)  # (B, NH, S)
#         f = self.fgate(if_gate_input).transpose(-1, -2)  # (B, NH, S)

#         # Reshape for backend
#         q = q.view(B, S, self.num_heads, -1).transpose(1, 2)
#         k = k.view(B, S, self.num_heads, -1).transpose(1, 2)
#         v = v.view(B, S, self.num_heads, -1).transpose(1, 2)

#         # Execute backend kernel (dtype-aligned)
#         h_state = backend(
#             q=q, k=k, v=v, i=i, f=f,
#             return_last_states=False,
#             mode="train"
#         )

#         # Reset states
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#         # Normalize and reshape output
#         h_norm = self.outnorm(h_state)
#         return h_norm.transpose(1, 2).reshape(B, S, H)

#     def reset_states(self):
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.fgate.weight)
#         bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
#         torch.nn.init.zeros_(self.igate.weight)
#         torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
#         self.outnorm.reset_parameters()



# Updated MatrixLSTMCell
# class MatrixLSTMCell(nn.Module):
#     def __init__(self, dim, num_heads, norm_bias=True, chunk_size=256):
#         super().__init__()
#         self.dim = dim
#         self.num_heads = num_heads

#         # print("CHUNK SIZE: " + str(chunk_size))
#         self.igate = nn.Linear(3 * dim, num_heads)
#         self.fgate = nn.Linear(3 * dim, num_heads)
#         self.outnorm = MultiHeadLayerNorm(ndim=dim, weight=True, bias=norm_bias)

#         # CPU-compatible backend configuration
#         self.cpu_backend_config = mLSTMBackendConfig(
#             chunkwise_kernel="chunkwise--native_autograd",
#             sequence_kernel="native_sequence__native",
#             step_kernel="native",
#             chunk_size=chunk_size,
#             autocast_kernel_dtype="float32",
#             return_last_states=False,
#             mode="train"
#         )
#         self.cpu_backend = mLSTMBackend(self.cpu_backend_config)

#         # GPU-compatible (Triton) backend configuration
#         self.gpu_backend_config = mLSTMBackendConfig(
#             chunkwise_kernel="chunkwise--triton_xl_chunk_siging",
#             sequence_kernel="native_sequence__triton",
#             step_kernel="triton",
#             chunk_size=chunk_size,
#             autocast_kernel_dtype="float32",
#             return_last_states=False,
#             mode="train"
#         )
#         self.gpu_backend = mLSTMBackend(self.gpu_backend_config) # Lazily initialize

#         self.causal_mask_cache = {}  # Retained for potential fallback

#         # Internal states
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#         self.reset_parameters()

#     def get_gpu_backend(self, device):
#         if self.gpu_backend is None:
#             if not torch.cuda.is_available():
#                 raise RuntimeError("CUDA is not available, but a CUDA device was requested.")
#             self.gpu_backend = mLSTMBackend(self.gpu_backend_config).to(device)
#         return self.gpu_backend

#     def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
#         B, S, H = q.shape  # (B, S, H)

#         # Ensure all inputs are on the same device
#         if not (q.device == k.device == v.device):
#             raise ValueError("All input tensors (q, k, v) must be on the same device.")

#         device = q.device
#         # print(device)
#         backend = self.get_gpu_backend(device) if device.type == 'cuda' else self.cpu_backend

#         # Prepare gate inputs
#         if_gate_input = torch.cat([q, k, v], dim=-1)  # (B, S, 3*H)
#         i = self.igate(if_gate_input).transpose(-1, -2)  # (B, NH, S)
#         f = self.fgate(if_gate_input).transpose(-1, -2)  # (B, NH, S)

#         # Reshape q, k, v for backend
#         q = q.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)
#         k = k.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)
#         v = v.view(B, S, self.num_heads, -1).transpose(1, 2)  # (B, NH, S, DH)

#         # Call backend with current internal states
#         h_state = backend(
#             q=q, k=k, v=v, i=i, f=f,
#             # c_initial=self.c_state,
#             # n_initial=self.n_state,
#             # m_initial=self.m_state,
#             return_last_states=False,
#             mode="train"  # or "inference" based on context
#         )

#         # Update internal states
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None
#         # self.c_state = c_out
#         # self.n_state = n_out
#         # self.m_state = m_out

#         # Apply normalization and reshape
#         h_state_norm = self.outnorm(h_state)  # (B, NH, S, DH)
#         h_state_norm = h_state_norm.transpose(1, 2).reshape(B, S, H)  # (B, S, H)

#         return h_state_norm

#     def reset_states(self):
#         """Reset internal states to None."""
#         self.c_state = None
#         self.n_state = None
#         self.m_state = None

#     def reset_parameters(self):
#         torch.nn.init.zeros_(self.fgate.weight)
#         bias_linspace_init_(self.fgate.bias, start=3.0, end=6.0)
#         torch.nn.init.zeros_(self.igate.weight)
#         torch.nn.init.normal_(self.igate.bias, mean=0.0, std=0.1)
#         self.outnorm.reset_parameters()


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
#         seqlens=None,  # Initial seqlens, can be overridden in forward
#         num_blocks=None,
#         chunk_size=64,
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

#         inner_dim = expansion * dim
#         num_heads = inner_dim // qkv_block_size
#         self.proj_up = nn.Linear(in_features=dim, out_features=2 * inner_dim, bias=proj_bias)
#         self.q_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
#         self.k_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
#         self.v_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)

#         # Convolution selection with 3D support
#         if conv_kind == "causal1d":
#             self.conv = CausalConv1d(dim=inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
#         elif conv_kind == "2d":
#             assert conv_kernel_size % 2 == 1, "For 2D, kernel size must be odd for same spatial dims"
#             self.conv = SequenceConv2d(
#                 in_channels=inner_dim,
#                 out_channels=inner_dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=inner_dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 2 else None,
#             )
#         elif conv_kind == "3d":
#             assert conv_kernel_size % 2 == 1, "For 3D, kernel size must be odd for same spatial dims"
#             self.conv = SequenceConv3d(
#                 in_channels=inner_dim,
#                 out_channels=inner_dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=inner_dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 3 else None,
#             )
#         else:
#             raise NotImplementedError(f"conv_kind={conv_kind} not implemented.")


#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=qkv_block_size,
#             norm_bias=norm_bias,
#             chunk_size=chunk_size,
#         )
#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
#         self.proj_down = nn.Linear(in_features=inner_dim, out_features=dim, bias=proj_bias)

#         self.reset_parameters()

#     def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
#         B, S, _ = x.shape  # S = T * H * W for 3D sequences

#         # Update conv seqlens dynamically if provided
#         if seqlens and hasattr(self.conv, 'seqlens'):
#             if self.conv_kind == "2d" and len(seqlens) == 2:
#                 self.conv.seqlens = seqlens
#             elif self.conv_kind == "3d" and len(seqlens) == 3:
#                 self.conv.seqlens = seqlens
#             elif self.conv_kind != "causal1d":
#                 raise ValueError(f"seqlens {seqlens} incompatible with conv_kind={self.conv_kind}")

#         # Direction handling
#         if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
#             pass
#         elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])
#         else:
#             raise NotImplementedError(f"Unknown direction: {self.direction}")

#         x_inner = self.proj_up(x)
#         x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)
#         x_mlstm_conv = self.conv(x_mlstm)  # Convolution handles 2D or 3D internally
#         x_mlstm_conv_act = F.silu(x_mlstm_conv)

#         q = self.q_proj(x_mlstm_conv_act)
#         k = self.k_proj(x_mlstm_conv_act)
#         v = self.v_proj(x_mlstm)

#         h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

#         h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)
#         h_state = h_tilde_state_skip * F.silu(z)
#         out = self.proj_down(h_state)

#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])

# #         return out

#     def reset_parameters(self):
#         small_init_(self.proj_up.weight, dim=self.dim)
#         if self.proj_up.bias is not None:
#             nn.init.zeros_(self.proj_up.bias)

#         if self.init_weights == "original":
#             wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
#         elif self.init_weights == "original-fixed":
#             wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
#         else:
#             raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
#         if self.proj_down.bias is not None:
#             nn.init.zeros_(self.proj_down.bias)

#         nn.init.ones_(self.learnable_skip)

#         def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
#             small_init_(qkv_proj.weight, dim=self.dim)
#             if qkv_proj.bias is not None:
#                 nn.init.zeros_(qkv_proj.bias)

#         _init_qkv_proj(self.q_proj)
#         _init_qkv_proj(self.k_proj)
#         _init_qkv_proj(self.v_proj)

#         self.mlstm_cell.reset_parameters()

from .mlstm_large import mLSTMLayerVision


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
        init_weights="original-fixed",
        seqlens=None,  # Initial seqlens, can be overridden in forward
        num_blocks=12,
        chunk_size=64,
    ):
        super().__init__()
        assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
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

        inner_dim = expansion * dim
        num_heads = inner_dim // qkv_block_size
        self.proj_up = nn.Linear(in_features=dim, out_features=2 * inner_dim, bias=proj_bias)
        self.q_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
        self.k_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
        self.v_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)

        # Convolution selection with 3D support
        if conv_kind == "causal1d":
            self.conv = CausalConv1d(dim=inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, "For 2D, kernel size must be odd for same spatial dims"
            self.conv = SequenceConv2d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens if seqlens and len(seqlens) == 2 else None,
            )
        elif conv_kind == "3d":
            assert conv_kernel_size % 2 == 1, "For 3D, kernel size must be odd for same spatial dims"
            self.conv = SequenceConv3d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens if seqlens and len(seqlens) == 3 else None,
            )
        else:
            raise NotImplementedError(f"conv_kind={conv_kind} not implemented.")


        self.mlstm_cell = MatrixLSTMCell(
            dim=inner_dim,
            num_heads=qkv_block_size,
            norm_bias=norm_bias,
            #chunk_size=chunk_size,
        )
        self.learnable_skip = nn.Parameter(torch.ones(inner_dim))
        self.proj_down = nn.Linear(in_features=inner_dim, out_features=dim, bias=proj_bias)

        self.reset_parameters()

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        B, S, _ = x.shape  # S = T * H * W for 3D sequences

        # Update conv seqlens dynamically if provided
        if seqlens and hasattr(self.conv, 'seqlens'):
            if self.conv_kind == "2d" and len(seqlens) == 2:
                self.conv.seqlens = seqlens
            elif self.conv_kind == "3d" and len(seqlens) == 3:
                self.conv.seqlens = seqlens
            elif self.conv_kind != "causal1d":
                raise ValueError(f"seqlens {seqlens} incompatible with conv_kind={self.conv_kind}")

        # Direction handling
        if self.direction == SequenceTraversal.ROWWISE_FROM_TOP_LEFT:
            pass
        elif self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        else:
            raise NotImplementedError(f"Unknown direction: {self.direction}")

        x_inner = self.proj_up(x)
        x_mlstm, z = torch.chunk(x_inner, chunks=2, dim=-1)
        x_mlstm_conv = self.conv(x_mlstm)  # Convolution handles 2D or 3D internally
        x_mlstm_conv_act = F.silu(x_mlstm_conv)

        q = self.q_proj(x_mlstm_conv_act)
        k = self.k_proj(x_mlstm_conv_act)
        v = self.v_proj(x_mlstm)

        h_tilde_state = self.mlstm_cell(q=q, k=k, v=v)

        h_tilde_state_skip = h_tilde_state + (self.learnable_skip * x_mlstm_conv_act)
        h_state = h_tilde_state_skip * F.silu(z)
        out = self.proj_down(h_state)

        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            out = out.flip(dims=[1])

        return out

    def reset_parameters(self):
        small_init_(self.proj_up.weight, dim=self.dim)
        if self.proj_up.bias is not None:
            nn.init.zeros_(self.proj_up.bias)

        if self.init_weights == "original":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=1)
        elif self.init_weights == "original-fixed":
            wang_init_(self.proj_down.weight, dim=self.dim, num_blocks=self.num_blocks)
        else:
            raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
        if self.proj_down.bias is not None:
            nn.init.zeros_(self.proj_down.bias)

        nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj: LinearHeadwiseExpand):
            small_init_(qkv_proj.weight, dim=self.dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.q_proj)
        _init_qkv_proj(self.k_proj)
        _init_qkv_proj(self.v_proj)

        self.mlstm_cell.reset_parameters()
    

from .mlstm_large import VilLayerUpdated

class ViLBlock(nn.Module):
    def __init__(
        self,
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
        qkv_block_size = 4
    ):
        super().__init__()
        self.dim = dim
        self.direction = direction
        self.drop_path = drop_path
        self.conv_kind = conv_kind
        self.conv_kernel_size = conv_kernel_size
        self.init_weights = init_weights

        self.drop_path = DropPath(drop_prob=drop_path)
        self.norm = LayerNorm(ndim=dim, weight=True, bias=norm_bias)
        # self.layer = VilLayerUpdated(
        #     embedding_dim=dim,
        #     num_heads=4,
        #     use_bias=True,
        #     norm_eps = 1e-6,
        #     norm_reduction_force_float32=True,
        #     qk_dim_factor=0.5,
        #     v_dim_factor=1.0,
        #     gate_soft_cap=15.0,
        #     weight_mode="single",
        #     ffn_proj_factor=2.6667,
        #     ffn_round_up_to_multiple_of=64,
        #     chunkwise_kernel = "chunkwise--triton_limit_chunk",
        #     sequence_kernel = "native_sequence__triton",
        #     step_kernel = "triton",
        #     mode = "train",
        #     chunk_size=chunk_size,            
        #     return_last_states=False,
        #     autocast_kernel_dtype="float32",
        #     eps = 1e-6,
        #     inference_state_dtype="float32",
        #     seqlens=seqlens,
        #     direction=direction
        # )
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
            chunk_size=chunk_size,)
        # )
        # self.layer = ViLLayerLite(
        #     dim=dim,
        #     direction=direction,
        #     conv_kind=conv_kind,
        #     conv_kernel_size=conv_kernel_size,
        #     seqlens=seqlens,
        #     norm_bias=norm_bias,
        #     proj_bias=proj_bias,
        #     num_blocks=num_blocks,
        #     init_weights=init_weights,
        #     chunk_size=chunk_size,
        #     qkv_block_size = 4,
        #     mlp_type="baseline"
        # )

        self.reset_parameters()

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        x = self.norm(x)
        residual = self.layer(x)
        out = self.drop_path(x, lambda _: residual)
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
            qkv_block_size = 4
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
            qkv_block_size = 4
        )

    def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
        out1 = self.rowwise_from_top_left(x, seqlens=seqlens)
        out2 = self.rowwise_from_bot_right(out1, seqlens=seqlens)
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

# class ViLLayerLite(nn.Module):
#     def __init__(
#         self,
#         dim,
#         direction,
#         expansion=1,  # Fixed to 1 to retain ViLLayerLite's logic
#         qkv_block_size=4,
#         proj_bias=True,
#         norm_bias=True,
#         conv_bias=True,
#         conv_kernel_size=4,  # Default from ViLLayerLite
#         conv_kind="2d",
#         init_weights="original",
#         seqlens=None,
#         num_blocks=None,
#         chunk_size=256,
#         mlp_type="baseline",
#         mlp_kwargs=None,
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.expansion = expansion  # Always 1 for ViLLayerLite
#         self.qkv_block_size = qkv_block_size
#         self.proj_bias = proj_bias
#         self.norm_bias = norm_bias
#         self.conv_bias = conv_bias
#         self.conv_kernel_size = conv_kernel_size
#         self.conv_kind = conv_kind
#         self.init_weights = init_weights
#         self.seqlens = seqlens
#         self.num_blocks = num_blocks

#         # Since expansion=1, inner_dim equals dim
#         inner_dim = expansion * dim  # inner_dim = dim
#         num_heads = 2 * inner_dim // qkv_block_size

#         # Convolution, mirroring ViLLayer's structure
#         if conv_kind == "causal1d":
#             self.conv = CausalConv1d(dim=inner_dim, kernel_size=conv_kernel_size, bias=conv_bias)
#         elif conv_kind == "2d":
#             assert conv_kernel_size % 2 == 1, "For 2D, kernel size must be odd for same spatial dims"
#             self.conv = SequenceConv2d(
#                 in_channels=inner_dim,
#                 out_channels=inner_dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=inner_dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 2 else None,
#             )
#         elif conv_kind == "3d":
#             assert conv_kernel_size % 2 == 1, "For 3D, kernel size must be odd for same spatial dims"
#             self.conv = SequenceConv3d(
#                 in_channels=inner_dim,
#                 out_channels=inner_dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=inner_dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 3 else None,
#             )
#         else:
#             raise NotImplementedError(f"conv_kind={conv_kind} not implemented.")

#         # Q/K/V projections using ViLLayer's LinearHeadwiseExpand
#         self.q_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
#         self.k_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)
#         self.v_proj = LinearHeadwiseExpand(dim=inner_dim, num_heads=num_heads, bias=proj_bias)

#         # Gating projection, unique to ViLLayerLite's logic
#         self.gate_proj = nn.Linear(inner_dim, inner_dim, bias=proj_bias)

#         # mLSTM cell, matching ViLLayer's convention
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=qkv_block_size,  # Consistent with ViLLayer
#             norm_bias=norm_bias,
#         )

#         # Learnable skip connection, as in ViLLayer
#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

#         # Normalization before MLP
#         self.norm = LayerNorm(ndim=inner_dim, weight=True, bias=norm_bias)

#         # MLP, preserving ViLLayerLite's logic
#         mlp_kwargs = mlp_kwargs or {}
#         if mlp_type == "mixer":
#             seq_len = math.prod(seqlens) if seqlens else 196  # Default 14*14
#             mlp_kwargs.setdefault("seq_len", seq_len)
#         self.mlp = MLP_REGISTRY[mlp_type](dim=inner_dim, **mlp_kwargs)

#         self.reset_parameters()

#     def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
#         B, S, _ = x.shape

#         # Update conv seqlens dynamically, as in ViLLayer
#         if seqlens and hasattr(self.conv, 'seqlens'):
#             if self.conv_kind == "2d" and len(seqlens) == 2:
#                 self.conv.seqlens = seqlens
#             elif self.conv_kind == "3d" and len(seqlens) == 3:
#                 self.conv.seqlens = seqlens
#             elif self.conv_kind != "causal1d":
#                 raise ValueError(f"seqlens {seqlens} incompatible with conv_kind={self.conv_kind}")

#         # Direction handling, mirroring ViLLayer
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # Convolution and activation
#         x_conv = self.conv(x)  # Operates at dim since inner_dim = dim
#         x_conv_act = F.silu(x_conv)

#         # Gating vector from input x (ViLLayerLite logic)
#         z = self.gate_proj(x)

#         # Q/K/V projections
#         q = self.q_proj(x_conv_act)
#         k = self.k_proj(x_conv_act)
#         v = self.v_proj(x)

#         # mLSTM processing
#         h_tilde = self.mlstm_cell(q=q, k=k, v=v)
#         h_tilde_skip = h_tilde + self.learnable_skip * x_conv_act
#         h_state = h_tilde_skip * F.silu(z)

#         # Flip back if reversed
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             h_state = h_state.flip(dims=[1])

#         # Normalization and MLP (ViLLayerLite logic)
#         x_norm = self.norm(h_state)
        
#         # MLP forward with shape handling
#         if isinstance(self.mlp, (ConvMLP, RGBlock)):
#             H, W = self.seqlens if self.seqlens and len(self.seqlens) == 2 else (int(S**0.5), int(S**0.5))
#             x_img = rearrange(x_norm, "b (h w) d -> b d h w", h=H)
#             x_out = self.mlp(x_img)
#             x_out = rearrange(x_out, "b d h w -> b (h w) d")
#         else:
#             x_out = self.mlp(x_norm)

#         # Residual connection
#         out = h_state + x_out

#         return out

#     def reset_parameters(self):
#         # Convolution initialization (default PyTorch init assumed)
#         # Q/K/V projections, matching ViLLayer
#         def _init_qkv_proj(qkv_proj):
#             small_init_(qkv_proj.weight, dim=self.dim)
#             if qkv_proj.bias is not None:
#                 nn.init.zeros_(qkv_proj.bias)
#         _init_qkv_proj(self.q_proj)
#         _init_qkv_proj(self.k_proj)
#         _init_qkv_proj(self.v_proj)

#         # Gating projection
#         small_init_(self.gate_proj.weight, dim=self.dim)
#         if self.gate_proj.bias is not None:
#             nn.init.zeros_(self.gate_proj.bias)

#         # mLSTM cell
#         self.mlstm_cell.reset_parameters()

#         # Learnable skip
#         nn.init.ones_(self.learnable_skip)

#         # Normalization
#         self.norm.reset_parameters()

#         # MLP initialization (example for baseline)
#         if isinstance(self.mlp, MLPBaseline):
#             small_init_(self.mlp.net[0].weight, dim=self.dim)
#             if self.mlp.net[0].bias is not None:
#                 nn.init.zeros_(self.mlp.net[0].bias)
#             wang_init_(self.mlp.net[2].weight, dim=self.dim, num_blocks=1 if self.init_weights == "original" else self.num_blocks)
#             if self.mlp.net[2].bias is not None:
#                 nn.init.zeros_(self.mlp.net[2].bias)
#         # Add other MLP types as needed

#grok refactor lamba traiing runs 0-44
# class ViLLayerLite(nn.Module):
#     """
#     A Vision-Language Layer (ViLLayer) that integrates a convolutional mechanism, 
#     matrix LSTM cell, and a flexible MLP selected from MLP_REGISTRY.
#     """
#     def __init__(
#         self,
#         dim,
#         direction,
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
#         drop_path=0.0,
#         mlp_type="baseline",  # Select MLP from MLP_REGISTRY
#         mlp_hidden_dim=None,  # Hidden dimension for MLP
#         mlp_kwargs=None,      # Additional MLP parameters (e.g., seq_len, rank)
#     ):
#         """
#         Initialize the ViLLayer.

#         Args:
#             dim (int): Dimension of the input and output tensors.
#             direction (SequenceTraversal): Direction of sequence traversal.
#             qkv_block_size (int): Block size for Q/K/V projections.
#             proj_bias (bool): Whether to use bias in projection layers.
#             norm_bias (bool): Whether to use bias in normalization layers.
#             conv_bias (bool): Whether to use bias in convolution layers.
#             conv_kernel_size (int): Kernel size for convolution.
#             conv_kind (str): Type of convolution ('causal1d', '2d', '3d').
#             init_weights (str): Weight initialization method.
#             seqlens (tuple): Sequence lengths (e.g., (H, W) for 2D).
#             num_blocks (int): Number of blocks for initialization.
#             chunk_size (int): Chunk size for MatrixLSTMCell (if applicable).
#             drop_path (float): Drop path probability.
#             mlp_type (str): Type of MLP to use from MLP_REGISTRY.
#             mlp_hidden_dim (int, optional): Hidden dimension for MLP.
#             mlp_kwargs (dict, optional): Additional MLP parameters.
#         """
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.qkv_block_size = qkv_block_size
#         self.conv_kind = conv_kind
#         self.init_weights = init_weights
#         self.seqlens = seqlens

#         num_heads = dim // qkv_block_size

#         # First normalization layer (pre-attention)
#         self.norm1 = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

#         # Convolution layer based on conv_kind
#         if conv_kind == "causal1d":
#             self.conv = CausalConv1d(dim=dim, kernel_size=conv_kernel_size, bias=conv_bias)
#         elif conv_kind == "2d":
#             assert conv_kernel_size % 2 == 1, "For 2D, kernel size must be odd"
#             self.conv = SequenceConv2d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 2 else None,
#             )
#         elif conv_kind == "3d":
#             assert conv_kernel_size % 2 == 1, "For 3D, kernel size must be odd"
#             self.conv = SequenceConv3d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 3 else None,
#             )
#         else:
#             raise NotImplementedError(f"conv_kind={conv_kind} not implemented.")

#         # Q/K/V projection layers
#         self.q_proj = LinearHeadwiseExpand(dim=dim, num_heads=num_heads, bias=proj_bias)
#         self.k_proj = LinearHeadwiseExpand(dim=dim, num_heads=num_heads, bias=proj_bias)
#         self.v_proj = LinearHeadwiseExpand(dim=dim, num_heads=num_heads, bias=proj_bias)

#         # Gate projection
#         self.gate_proj = nn.Linear(dim, dim, bias=proj_bias)

#         # Matrix LSTM cell
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=dim,
#             num_heads=qkv_block_size,
#             norm_bias=norm_bias,
#         )

#         # Learnable skip connection parameter
#         self.learnable_skip = nn.Parameter(torch.ones(dim))

#         # Drop path after mLSTM
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         # Second normalization layer (pre-MLP)
#         self.norm2 = LayerNorm(ndim=dim, weight=True, bias=norm_bias)

#         # MLP setup using MLP_REGISTRY
#         if mlp_hidden_dim is None:
#             mlp_hidden_dim = 4 * dim  # Default MLP expansion factor
#         mlp_kwargs = mlp_kwargs or {}
#         mlp_kwargs["hidden_dim"] = mlp_hidden_dim
#         self.mlp = MLP_REGISTRY[mlp_type](dim=dim, **mlp_kwargs)

#         # Drop path after MLP
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.reset_parameters()

#     def forward(self, x: torch.Tensor, seqlens=None) -> torch.Tensor:
#         """
#         Forward pass of the ViLLayer.

#         Args:
#             x (torch.Tensor): Input tensor of shape (B, S, dim).
#             seqlens (tuple, optional): Sequence lengths for reshaping.

#         Returns:
#             torch.Tensor: Output tensor of shape (B, S, dim).
#         """
#         B, S, _ = x.shape

#         # Update convolution seqlens if provided
#         if seqlens and hasattr(self.conv, 'seqlens'):
#             if self.conv_kind == "2d" and len(seqlens) == 2:
#                 self.conv.seqlens = seqlens
#             elif self.conv_kind == "3d" and len(seqlens) == 3:
#                 self.conv.seqlens = seqlens
#             elif self.conv_kind != "causal1d":
#                 raise ValueError(f"seqlens {seqlens} incompatible with conv_kind={self.conv_kind}")

#         # First residual connection
#         shortcut = x
#         x = self.norm1(x)

#         # Handle sequence direction
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # Convolution and activation
#         x_conv = self.conv(x)
#         x_conv_act = F.silu(x_conv)

#         # Q/K/V computation
#         q = self.q_proj(x_conv_act)
#         k = self.k_proj(x_conv_act)
#         v = self.v_proj(x)

#         # Gating vector
#         z = self.gate_proj(x)

#         # Matrix LSTM processing
#         h_tilde = self.mlstm_cell(q=q, k=k, v=v)
#         h_tilde_skip = h_tilde + (self.learnable_skip * x_conv_act)
#         h_state = h_tilde_skip * F.silu(z)

#         # Reverse direction if needed
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             h_state = h_state.flip(dims=[1])

#         # First residual connection with drop path
#         out = shortcut + self.drop_path1(h_state)

#         # Second normalization
#         out_norm = self.norm2(out)

#         # MLP processing with shape handling
#         if getattr(self.mlp, "input_shape_type", "sequence") == "image":
#             if self.conv_kind != "2d" or not self.seqlens:
#                 raise ValueError("Image-based MLPs require conv_kind='2d' and seqlens (H, W)")
#             H, W = self.seqlens
#             x_mlp = einops.rearrange(out_norm, "b (h w) c -> b c h w", h=H, w=W)
#             x_mlp = self.mlp(x_mlp)
#             x_mlp = einops.rearrange(x_mlp, "b c h w -> b (h w) c")
#         else:
#             x_mlp = self.mlp(out_norm)

#         # Second residual connection with drop path
#         out = out + self.drop_path2(x_mlp)

#         return out

#     def reset_parameters(self):
#         """Initialize layer parameters."""
#         self.norm1.reset_parameters()
#         self.norm2.reset_parameters()

#         # Initialize Q/K/V projections
#         small_init_(self.q_proj.weight, dim=self.dim)
#         if self.q_proj.bias is not None:
#             nn.init.zeros_(self.q_proj.bias)
#         small_init_(self.k_proj.weight, dim=self.dim)
#         if self.k_proj.bias is not None:
#             nn.init.zeros_(self.k_proj.bias)
#         small_init_(self.v_proj.weight, dim=self.dim)
#         if self.v_proj.bias is not None:
#             nn.init.zeros_(self.v_proj.bias)

#         # Initialize gate projection
#         small_init_(self.gate_proj.weight, dim=self.dim)
#         if self.gate_proj.bias is not None:
#             nn.init.zeros_(self.gate_proj.bias)

#         # Initialize mLSTM cell and skip connection
#         self.mlstm_cell.reset_parameters()
#         nn.init.ones_(self.learnable_skip)

#         # MLP initialization (example for baseline, extend as needed)
#         if hasattr(self.mlp, 'net') and len(self.mlp.net) >= 3:
#             small_init_(self.mlp.net[0].weight, dim=self.dim)
#             if self.mlp.net[0].bias is not None:
#                 nn.init.zeros_(self.mlp.net[0].bias)
#             wang_init_(self.mlp.net[2].weight, dim=self.dim, num_blocks=1 if self.init_weights == "original" else self.num_blocks)
#             if self.mlp.net[2].bias is not None:
#                 nn.init.zeros_(self.mlp.net[2].bias)


# #GPT 01 Refactor
# class ViLLayerLite(nn.Module):
#     """
#     ViL‑style mLSTM block that runs at native width and adds a post‑MLP.
#     Public interface is unchanged from the original ViLLayer.
#     """
#     def __init__(
#         self,
#         dim,
#         direction,
#         *,
#         expansion: int = 1,               # fixed at 1 (no inner expansion)
#         qkv_block_size: int = 2,
#         mlstm_num_heads: int | None = None,
#         proj_bias: bool = True,
#         norm_bias: bool = True,
#         conv_bias: bool = True,
#         conv_kernel_size: int = 3,
#         conv_kind: str = "causal1d",
#         init_weights: str = "original",
#         seqlens=None,
#         num_blocks=None,
#         drop_path: float = 0.0,
#         mlp_hidden_dim: int | None = None,
#         mlp_type: str = "baseline",
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0
#         self.dim = dim
#         self.direction = direction
#         self.init_weights = init_weights
#         self.num_blocks = num_blocks

#         # -------- sizes -------------------------------------------------------
#         inner_dim = expansion * dim          # = dim (expansion is 1)
#         self.num_lin_heads = inner_dim // qkv_block_size
#         self.mlstm_num_heads = mlstm_num_heads or qkv_block_size
#         assert inner_dim % self.mlstm_num_heads == 0, "dim % mlstm_heads must be 0"
#         # ----------------------------------------------------------------------

#         # up‑projection (still split into mlstm path + z gate)
#         self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

#         # block‑diagonal Q/K/V
#         self.q_proj = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)
#         self.k_proj = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)
#         self.v_proj = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)

#         # depth‑wise conv
#         if conv_kind == "causal1d":
#             self.conv = CausalConv1d(inner_dim, conv_kernel_size, bias=conv_bias)
#         elif conv_kind == "2d":
#             assert conv_kernel_size % 2 == 1
#             self.conv = SequenceConv2d(
#                 inner_dim, inner_dim, conv_kernel_size,
#                 padding=conv_kernel_size // 2, groups=inner_dim,
#                 bias=conv_bias, seqlens=seqlens)
#         else:
#             raise NotImplementedError(conv_kind)

#         # mLSTM
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=self.mlstm_num_heads,
#             norm_bias=norm_bias,
#         )

#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

#         # down‑projection (back to dim)
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         # post‑MLP branch
#         self.norm2 = LayerNorm(dim, bias=norm_bias)
#         hidden = mlp_hidden_dim or 4 * dim
#         self.mlp = MLP_REGISTRY[mlp_type](dim=dim, hidden_dim=hidden)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

#         self.reset_parameters()

#     # ---------------------------------------------------------------------- #
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         x : (B, S, dim)  →  (B, S, dim)
#         """
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # up‑projection and split
#         x_inner = self.proj_up(x)                      # (B,S,2*dim)
#         x_mlstm, z = torch.chunk(x_inner, 2, dim=-1)   # both (B,S,dim)

#         # conv + activation
#         x_conv = F.silu(self.conv(x_mlstm))

#         # Q/K/V
#         q = self.q_proj(x_conv)
#         k = self.k_proj(x_conv)
#         v = self.v_proj(x_mlstm)

#         # mLSTM
#         h = self.mlstm_cell(q=q, k=k, v=v)
#         h = h + self.learnable_skip * x_conv
#         h = h * F.silu(z)

#         # down‑projection
#         out = self.proj_down(h)

#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])

#         # post‑MLP enrichment (residual)
#         out = out + self.drop_path2(self.mlp(self.norm2(out)))
#         return out

#     # ---------------------------------------------------------------------- #
#     def reset_parameters(self):
#         small_init_(self.proj_up.weight, dim=self.dim)
#         nn.init.zeros_(self.proj_up.bias)

#         wang_init_(self.proj_down.weight, dim=self.dim,
#                    num_blocks=1 if self.init_weights == "original" else self.num_blocks)
#         nn.init.zeros_(self.proj_down.bias)

#         for proj in (self.q_proj, self.k_proj, self.v_proj):
#             small_init_(proj.weight, dim=self.dim)
#             if proj.bias is not None: nn.init.zeros_(proj.bias)

#         self.mlstm_cell.reset_parameters()
#         nn.init.ones_(self.learnable_skip)

# #final GPT refactor
# class ViLLayerLite(nn.Module):
#     """
#     ViL‑style mLSTM block that runs at native width (expansion=1) and adds a post‑MLP.
#     Public interface remains the same as the original ViLLayer.
#     """
#     def __init__(
#         self,
#         dim,
#         direction,
#         *,
#         expansion: int = 1,               # fixed at 1 (no inner expansion)
#         qkv_block_size: int = 4,          # set to 2 to reproduce original head count (e.g. 96 heads for dim=192)
#         mlstm_num_heads: int | None = None,
#         proj_bias: bool = True,
#         norm_bias: bool = True,
#         conv_bias: bool = True,
#         conv_kernel_size: int = 3,
#         conv_kind: str = "causal1d",
#         init_weights: str = "original",
#         seqlens=None,
#         num_blocks=None,
#         chunk_size = 256,
#         drop_path: float = 0.0,
#         mlp_hidden_dim: int | None = None,
#         mlp_type: str = "baseline",
#     ):
#         super().__init__()
#         assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
#         self.dim = dim
#         self.direction = direction
#         self.init_weights = init_weights
#         self.num_blocks = num_blocks
#         self.seqlens = seqlens

#         # -------- sizes -------------------------------------------------------
#         # With expansion=1, inner_dim equals dim.
#         inner_dim = expansion * dim  # = dim
#         # Recompute fine-head count as if expansion had been 2:
#         self.num_lin_heads = (dim) // qkv_block_size
#         assert dim % self.num_lin_heads == 0, "dim must be divisible by computed num_lin_heads"
#         # Coarse head count is set to qkv_block_size by default (you can override via mlstm_num_heads)
#         self.mlstm_num_heads = mlstm_num_heads or qkv_block_size
#         assert inner_dim % self.mlstm_num_heads == 0, "inner_dim must be divisible by mlstm_num_heads"
#         # ----------------------------------------------------------------------

#         # up‑projection: maps from dim to 2 * inner_dim
#         self.proj_up = nn.Linear(dim, 2 * inner_dim, bias=proj_bias)

#         # block‑diagonal Q/K/V projections
#         self.q_proj = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)
#         self.k_proj = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)
#         self.v_proj = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)

#         # Depth‑wise convolution
#         if conv_kind == "causal1d":
#             self.conv = CausalConv1d(inner_dim, conv_kernel_size, bias=conv_bias)
#         elif conv_kind == "2d":
#             assert conv_kernel_size % 2 == 1, "Kernel size must be odd for 2D conv."
#             self.conv = SequenceConv2d(
#                 inner_dim, inner_dim, conv_kernel_size,
#                 padding=conv_kernel_size // 2, groups=inner_dim,
#                 bias=conv_bias, seqlens=seqlens)
#         elif conv_kind == "3d":
#             assert conv_kernel_size % 2 == 1, "For 3D, kernel size must be odd"
#             self.conv = SequenceConv3d(
#                 in_channels=dim,
#                 out_channels=dim,
#                 kernel_size=conv_kernel_size,
#                 padding=conv_kernel_size // 2,
#                 groups=dim,
#                 bias=conv_bias,
#                 seqlens=seqlens if seqlens and len(seqlens) == 3 else None,
#             )
#         else:
#             raise NotImplementedError(conv_kind)

#         # mLSTM cell processing on native inner_dim with specified number of coarse heads
#         self.mlstm_cell = MatrixLSTMCell(
#             dim=inner_dim,
#             num_heads=self.mlstm_num_heads,
#             norm_bias=norm_bias,
#         )

#         self.learnable_skip = nn.Parameter(torch.ones(inner_dim))

#         # Down‑projection: maps back from inner_dim to dim.
#         self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

#         # ---------------- Post‑MLP branch ---------------------------------------
#         self.norm2 = LayerNorm(dim, bias=norm_bias)
#         hidden = 4 * dim
#         # Instantiate the MLP from our registry; no extra kwargs are passed here.
#         self.mlp = MLP_REGISTRY[mlp_type](dim, hidden_dim=hidden)
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         # ----------------------------------------------------------------------

#         self.reset_parameters()

#     # ---------------------------------------------------------------------- #
#     def forward(self, x: torch.Tensor, seqlens = None) -> torch.Tensor:
#         """
#         Input and output shapes: (B, S, dim)
#         """
#         B, S, _ = x.shape
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             x = x.flip(dims=[1])

#         # Up‑projection and splitting: produces x_mlstm and gating vector z
#         x_inner = self.proj_up(x)  # shape: (B, S, 2 * inner_dim)
#         x_mlstm, z = torch.chunk(x_inner, 2, dim=-1)  # each: (B, S, inner_dim)

#         # Depth‑wise convolution followed by activation
#         x_conv = F.silu(self.conv(x_mlstm))

#         # Q/K/V projections based on conv output (for Q and K) and original mlstm path (for V)
#         q = self.q_proj(x_conv)
#         k = self.k_proj(x_conv)
#         v = self.v_proj(x_mlstm)

#         # mLSTM processing: apply MatrixLSTMCell to q, k, v
#         h = self.mlstm_cell(q=q, k=k, v=v)
#         h = h + self.learnable_skip * x_conv
#         # Apply external gate (via SiLU) from z
#         h = h * F.silu(z)

#         # Down‑projection: project back to dim
#         out = self.proj_down(h)
#         if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
#             out = out.flip(dims=[1])

#         # Post‑MLP enrichment (residual branch)
#         out = out + self.drop_path2(self.mlp(self.norm2(out)))
#         return out

#     # ---------------------------------------------------------------------- #
#     def reset_parameters(self):
#         small_init_(self.proj_up.weight, dim=self.dim)
#         nn.init.zeros_(self.proj_up.bias)

#         wang_init_(self.proj_down.weight, dim=self.dim,
#                    num_blocks=1 if self.init_weights == "original" else self.num_blocks)
#         nn.init.zeros_(self.proj_down.bias)

#         for proj in (self.q_proj, self.k_proj, self.v_proj):
#             small_init_(proj.weight, dim=self.dim)
#             if proj.bias is not None:
#                 nn.init.zeros_(proj.bias)

#         self.mlstm_cell.reset_parameters()
#         nn.init.ones_(self.learnable_skip)

#o4 mini high refactor, trying to make params more configurable

class ViLLayerLite(nn.Module):
    """
    ViL‑style mLSTM block with uniform parameterization via control variables.
    """
    def __init__(
        self,
        dim: int,
        direction,
        *,
        expansion: int = 1,            # multiplier for inner_dim = expansion * dim
        proj_factor: int = 2,          # how many inner_dims to project up to
        num_proj_splits: int = 2,      # must equal proj_factor
        mlp_factor: int = 4,           # hidden size multiplier for post-MLP (mlp_factor * dim)
        qkv_block_size: int = 4,       # for block-diagonal Q/K/V
        mlstm_num_heads: int | None = None,
        proj_bias: bool = True,
        norm_bias: bool = True,
        conv_bias: bool = True,
        conv_kernel_size: int = 3,
        conv_kind: str = "causal1d",
        init_weights: str = "original",
        seqlens=None,
        num_blocks=None,
        drop_path: float = 0.0,
        mlp_hidden_dim: int | None = None,
        mlp_type: str = "baseline",
        chunk_size = 16
    ):
        super().__init__()
        # Core dims
        self.dim = dim
        self.exp = expansion
        self.proj_f = proj_factor
        self.splits = num_proj_splits
        self.mlp_f = mlp_factor
        

        # inner and projected dimensions
        inner_dim    = self.exp * self.dim
        proj_out_dim = inner_dim * self.proj_f
        # assert proj_out_dim == inner_dim * self.splits, \
        #     "proj_factor must equal num_proj_splits"

        # heads
        assert dim % qkv_block_size == 0, "dim must be divisible by qkv_block_size"
        self.num_lin_heads = inner_dim // qkv_block_size
        self.mlstm_num_heads = mlstm_num_heads or qkv_block_size
        assert inner_dim % self.mlstm_num_heads == 0, \
            "inner_dim must be divisible by mlstm_num_heads"

        # post-MLP hidden
        hidden_dim = mlp_hidden_dim if mlp_hidden_dim is not None else (self.mlp_f * self.dim)

        # modules
        self.proj_up  = nn.Linear(dim,      proj_out_dim, bias=proj_bias)
        # Q/K/V block-diagonal projections
        self.q_proj   = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)
        self.k_proj   = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)
        self.v_proj   = LinearHeadwiseExpand(inner_dim, self.num_lin_heads, bias=proj_bias)

        # convolution
        if conv_kind == "causal1d":
            self.conv = CausalConv1d(inner_dim, conv_kernel_size, bias=conv_bias)
        elif conv_kind == "2d":
            assert conv_kernel_size % 2 == 1, "Kernel size must be odd for 2D conv."
            self.conv = SequenceConv2d(
                inner_dim, inner_dim, conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens,
            )
        elif conv_kind == "3d":
            assert conv_kernel_size % 2 == 1, "Kernel size must be odd for 3D conv."
            self.conv = SequenceConv3d(
                in_channels=inner_dim,
                out_channels=inner_dim,
                kernel_size=conv_kernel_size,
                padding=conv_kernel_size // 2,
                groups=inner_dim,
                bias=conv_bias,
                seqlens=seqlens if seqlens and len(seqlens) == 3 else None,
            )
        else:
            raise NotImplementedError(f"conv_kind={conv_kind}")

        # mLSTM and skip
        chunk_size = (lambda seq: (seq if isinstance(seq, int) else seq[0]*seq[1]) // 16 * 16)(seqlens or self.seqlens)
        # print("CHUNK SIZE COMPUTED!" + str(chunk_size))
        self.mlstm_cell      = MatrixLSTMCell(dim=inner_dim,
                                              num_heads=self.mlstm_num_heads,
                                              norm_bias=norm_bias,
                                              #chunk_size=chunk_size
                                            )
        self.learnable_skip  = nn.Parameter(torch.ones(inner_dim))

        # down-projection
        self.proj_down = nn.Linear(inner_dim, dim, bias=proj_bias)

        # post-MLP
        self.norm2      = LayerNorm(dim, bias=norm_bias)
        self.mlp        = MLP_REGISTRY[mlp_type](dim, hidden_dim=hidden_dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.direction   = direction
        self.init_weights = init_weights
        self.num_blocks   = num_blocks
        self.seqlens      = seqlens
        self.reset_parameters()

    def forward(self, x: torch.Tensor, seqlens = None) -> torch.Tensor:
        B, S, _ = x.shape
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])

        # up-projection and split
        x_inner = self.proj_up(x)                       # (B,S,proj_out_dim)
        parts   = x_inner.chunk(self.splits, dim=-1)    # tuple of `splits` tensors
        x_mlstm, z = parts[0], parts[1]                 # each (B,S,inner_dim)

        # conv + activation
        x_conv = F.silu(self.conv(x_mlstm))             # (B,S,inner_dim)

        # Q/K/V
        q = self.q_proj(x_conv)
        k = self.k_proj(x_conv)
        v = self.v_proj(x_mlstm)

        # mLSTM
        h = self.mlstm_cell(q=q, k=k, v=v)
        h = h + self.learnable_skip * x_conv
        h = h * F.silu(z)

        # down-projection
        out = self.proj_down(h)
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            out = out.flip(dims=[1])

        # post-MLP residual
        out = out + self.drop_path2(self.mlp(self.norm2(out)))
        return out

    def reset_parameters(self):
        # init proj_up
        small_init_(self.proj_up.weight, dim=self.dim)
        nn.init.zeros_(self.proj_up.bias)
        # init proj_down
        wang_init_(self.proj_down.weight,
                   dim=self.dim,
                   num_blocks=1 if self.init_weights == "original" else self.num_blocks)
        nn.init.zeros_(self.proj_down.bias)
        # init QKV
        for proj in (self.q_proj, self.k_proj, self.v_proj):
            small_init_(proj.weight, dim=self.dim)
            if proj.bias is not None:
                nn.init.zeros_(proj.bias)
        # init mLSTM
        self.mlstm_cell.reset_parameters()
        # init skip
        nn.init.ones_(self.learnable_skip)



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

