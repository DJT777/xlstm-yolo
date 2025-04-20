import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .vision_lstm_util import SequenceConv2d
from typing import Optional, Literal, Tuple
from .vision_lstm2 import SequenceTraversal
from dataclasses import dataclass, replace

from .xlstm.xlstm_large.model import (
    mLSTMLayerStateType,
    MultiHeadLayerNorm,
    soft_cap,
    RMSNorm,


)

from mlstm_kernels.torch.backend_module import (
        mLSTMBackendConfig,
        mLSTMBackend,
        ChunkwiseKernelType,
        SequenceKernelType,
        StepKernelType,
        DtypeType,
        BackendModeType,
    )

mLSTMLayerStateType = tuple[torch.Tensor, torch.Tensor, torch.Tensor]
mLSTMStateType = dict[int, mLSTMLayerStateType]
WeightModeType = Literal["single", "fused"]



def round_up_to_next_multiple_of(x: int, multiple_of: int) -> int:
    """Rounds up x to the next multiple of multiple_of."""
    return int(((x + multiple_of - 1) // multiple_of) * multiple_of)


@dataclass
class mLSTMVisionBlockConfig:
    # core dimensions
    embedding_dim: int
    num_heads: int

    # biases & norms
    use_bias: bool = False
    norm_eps: float = 1e-6
    norm_reduction_force_float32: bool = False

    # projection factors
    qk_dim_factor: float = 0.5
    v_dim_factor: float = 1.0
    gate_soft_cap: float = 15.0
    weight_mode: WeightModeType = "single"

    #state param
    return_last_states: bool = False

    # feedforward parameters
    ffn_proj_factor: float = 2.6667
    ffn_round_up_to_multiple_of: int = 64

    # backend kernels & settings
    chunkwise_kernel: ChunkwiseKernelType = "chunkwise--triton_xl_chunk"
    sequence_kernel: SequenceKernelType = "native_sequence__triton"
    step_kernel: StepKernelType = "triton"
    mode: BackendModeType = "train"
    chunk_size: int = 64
    autocast_kernel_dtype: DtypeType = "bfloat16"
    eps: float = 1e-6
    inference_state_dtype: DtypeType = "bfloat16"
    seqlens: Optional[list[int]] = None

    mode = "train"

    # vision-specific
    seqlens: Optional[list[int]] = None

    # traversal
    direction: SequenceTraversal = SequenceTraversal.ROWWISE_FROM_TOP_LEFT

    num_blocks: int = 12


class FeedForward(nn.Module):
    def __init__(self, config: mLSTMVisionBlockConfig):
        super().__init__()
        self.config = config

        self.up_proj_dim = round_up_to_next_multiple_of(
            config.embedding_dim * config.ffn_proj_factor,
            config.ffn_round_up_to_multiple_of,
        )

        if self.config.weight_mode == "single":
            self.proj_up_gate = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
            self.proj_up = nn.Linear(
                in_features=config.embedding_dim,
                out_features=self.up_proj_dim,
                bias=self.config.use_bias,
            )
        elif self.config.weight_mode == "fused":
            self.proj_up_gate_z = nn.Linear(
                in_features=config.embedding_dim,
                out_features=2 * self.up_proj_dim,
                bias=self.config.use_bias,
            )

        self.proj_down = nn.Linear(
            in_features=self.up_proj_dim,
            out_features=config.embedding_dim,
            bias=self.config.use_bias,
        )

        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.weight_mode == "single":
            x = self.act_fn(self.proj_up_gate(x)) * self.proj_up(x)
        elif self.config.weight_mode == "fused":
            x = self.proj_up_gate_z(x)
            gate, z = torch.tensor_split(x, (self.up_proj_dim,), dim=-1)
            x = self.act_fn(gate) * z

        y = self.proj_down(x)
        return y


class mLSTMLayerVision(nn.Module):
    def __init__(self, config: mLSTMVisionBlockConfig, seqlens=[16,16]):
        super().__init__()
        self.config = config
        self.backend_end_config = mLSTMBackendConfig(
                    chunkwise_kernel=config.chunkwise_kernel,
                    sequence_kernel=config.sequence_kernel,
                    step_kernel=config.step_kernel,
                    mode=config.mode,
                    chunk_size=64,
                    return_last_states=config.return_last_states,
                    autocast_kernel_dtype=config.autocast_kernel_dtype,
                    eps=config.eps,
                    inference_state_dtype=config.inference_state_dtype,
                )

        self._state: Optional[mLSTMLayerStateType] = None

        self.v_dim = int(config.embedding_dim * config.v_dim_factor)
        self.qk_dim = int(config.embedding_dim * config.qk_dim_factor)

        # Add up-projection and convolution layers
        self.up_proj = nn.Linear(self.config.embedding_dim, self.config.embedding_dim, bias=self.config.use_bias)
        self.conv = SequenceConv2d(self.v_dim, self.v_dim, kernel_size=3, padding=1, bias=True, seqlens=seqlens)

        if self.config.weight_mode == "single":
            # q and k now project from v_dim (after convolution) instead of embedding_dim
            self.q = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            self.k = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.qk_dim,
                bias=self.config.use_bias,
            )
            # v projects from up-projected x_mlstm (v_dim)
            self.v = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )
            # o_preact projects from z (v_dim)
            self.ogate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.v_dim,
                bias=self.config.use_bias,
            )
            # i and f gates remain projected from input x
            self.igate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
            self.fgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=self.config.num_heads,
                bias=True,
            )
        elif self.config.weight_mode == "fused":
            # For simplicity, fused mode is not modified here; can be extended similarly if needed
            self.qkv_opreact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.qk_dim + 2 * self.v_dim,
                bias=self.config.use_bias,
            )
            self.ifgate_preact = nn.Linear(
                in_features=self.config.embedding_dim,
                out_features=2 * self.config.num_heads,
                bias=True,
            )

        self.ogate_act_fn = nn.Sigmoid()
        self.mlstm_backend = mLSTMBackend(config=self.backend_end_config
        )

        # 2) CPU: override config to use only native_autograd/sequence/step kernels
        cpu_conf = mLSTMBackendConfig(
                chunkwise_kernel="chunkwise--native_autograd",
                sequence_kernel="native_sequence__native",
                step_kernel="native",
                mode=self.config.mode,           # preserve train/inference
                autocast_kernel_dtype=self.config.autocast_kernel_dtype,
                eps=self.config.eps,
                inference_state_dtype=self.config.inference_state_dtype,
                return_last_states=self.config.return_last_states,
                chunk_size=64,
                )
    
        self.mlstm_backend_cpu = mLSTMBackend(config=cpu_conf)

        self.multihead_norm = MultiHeadLayerNorm(
            num_heads=self.config.num_heads,
            head_dim=self.v_dim // self.config.num_heads,
            eps=self.config.norm_eps,
            use_weight=True,
            use_bias=self.config.use_bias,
            force_float32_reductions=self.config.norm_reduction_force_float32,
        )
        self.out_proj = nn.Linear(
            in_features=self.v_dim,
            out_features=self.config.embedding_dim,
            bias=self.config.use_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Input must have shape [B, S, D], got {x.shape}"
        B, S, _ = x.shape

        if self.config.weight_mode == "single":
            # Convolutional processing for q and k
            conv_output = self.conv(x)  # [B, S, v_dim]
            act_conv = F.silu(conv_output)  # [B, S, v_dim]

            # Projections
            q = self.q(act_conv)  # [B, S, qk_dim]
            k = self.k(act_conv)  # [B, S, qk_dim]
            v = self.v(x)  # [B, S, v_dim]
            o_preact = self.ogate_preact(x)  # [B, S, v_dim]
            i_preact = soft_cap(
                self.igate_preact(x), cap_value=self.config.gate_soft_cap
            )
            f_preact = soft_cap(
                self.fgate_preact(x), cap_value=self.config.gate_soft_cap
            )

        elif self.config.weight_mode == "fused":
            # Original fused mode unchanged
            qkv_opreact = self.qkv_opreact(x)
            q, k, v, o_preact = torch.tensor_split(
                qkv_opreact,
                (
                    self.qk_dim,
                    2 * self.qk_dim,
                    2 * self.qk_dim + self.v_dim,
                ),
                dim=-1,
            )
            if_preact = soft_cap(
                self.ifgate_preact(x), cap_value=self.config.gate_soft_cap
            )
            i_preact, f_preact = torch.tensor_split(
                if_preact, (self.config.num_heads,), dim=-1
            )

        # Reshape and transpose for multi-head processing
        q = q.reshape(B, S, self.config.num_heads, -1).transpose(1, 2).contiguous()
        k = k.reshape(B, S, self.config.num_heads, -1).transpose(1, 2).contiguous()
        v = v.reshape(B, S, self.config.num_heads, -1).transpose(1, 2).contiguous()
        i_preact = i_preact.transpose(1, 2).contiguous()
        f_preact = f_preact.transpose(1, 2).contiguous()

        # Also make any initial states contiguous if they exist
        c0 = self._state[0].contiguous() if self._state is not None else None
        n0 = self._state[1].contiguous() if self._state is not None else None
        m0 = self._state[2].contiguous() if self._state is not None else None

        if self.config.return_last_states:
            # we know the kernel will return (H, (C_last, N_last, M_last))
            h, new_state = (
                self.mlstm_backend_cpu if x.device.type == "cpu" 
                else self.mlstm_backend
            )(
                q=q, k=k, v=v,
                i=i_preact, f=f_preact,
                c_initial=None if self._state is None else self._state[0],
                n_initial=None if self._state is None else self._state[1],
                m_initial=None if self._state is None else self._state[2],
                return_last_states=True,
                mode=(self.mlstm_backend_cpu if x.device.type == "cpu" 
                    else self.mlstm_backend).config.mode,
            )

            # update your saved state
            if self._state is None:
                self._state = new_state
            else:
                for idx in range(3):
                    self._state[idx].copy_(new_state[idx])

        else:
            # we know the kernel will return just H
            h = (
                self.mlstm_backend_cpu if x.device.type == "cpu" 
                else self.mlstm_backend
            )(
                q=q, k=k, v=v,
                i=i_preact, f=f_preact,
                c_initial=None if self._state is None else self._state[0],
                n_initial=None if self._state is None else self._state[1],
                m_initial=None if self._state is None else self._state[2],
                return_last_states=False,
                mode=(self.mlstm_backend_cpu if x.device.type == "cpu" 
                    else self.mlstm_backend).config.mode,
            )


        expected_h_shape = (
            B,
            self.config.num_heads,
            S,
            self.v_dim // self.config.num_heads,
        )


        assert (
            h.shape == expected_h_shape
        ), f"Got {h.shape}, expected {expected_h_shape}"

        # Post-processing
        h = h.transpose(1, 2)
        h_norm = self.multihead_norm(h)
        h_norm = h_norm.reshape(B, S, -1)
        h_out = self.ogate_act_fn(o_preact) * h_norm
        y = self.out_proj(h_out)

        return y


class mLSTMBlock(nn.Module):
    def __init__(self, config: mLSTMVisionBlockConfig):
        super().__init__()
        self.config = config
        self.norm_mlstm = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.mlstm_layer = mLSTMLayerVision(
            mLSTMVisionBlockConfig(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                use_bias=config.use_bias,
                norm_eps=config.norm_eps,
                norm_reduction_force_float32=config.norm_reduction_force_float32,
                qk_dim_factor=config.qk_dim_factor,
                v_dim_factor=config.v_dim_factor,
                gate_soft_cap=config.gate_soft_cap,
                weight_mode=config.weight_mode,
                chunkwise_kernel=config.chunkwise_kernel,
                sequence_kernel=config.sequence_kernel,
                step_kernel=config.step_kernel,
                mode=config.mode,
                chunk_size=config.chunk_size,
                return_last_states=config.return_last_states,
                autocast_kernel_dtype=config.autocast_kernel_dtype,
                eps=config.eps,
                inference_state_dtype=config.inference_state_dtype,
            )
        )
        self.norm_ffn = RMSNorm(
            num_features=config.embedding_dim,
            eps=config.norm_eps,
            use_weight=True,
            use_bias=config.use_bias,
            force_float32_reductions=config.norm_reduction_force_float32,
        )
        self.ffn = FeedForward(config)

    def forward(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        x_mlstm = self.norm_mlstm(x)
        x_mlstm = self.mlstm_layer(x_mlstm)
        x = x + x_mlstm
        x_ffn = self.norm_ffn(x)
        x_ffn = self.ffn(x_ffn)
        x = x + x_ffn

        return x



class VilLayerUpdated(nn.Module):
    """
    Vision mLSTM + FFN block with traversal and optional state return.
    """
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        use_bias: bool = False,
        norm_eps: float = 1e-6,
        norm_reduction_force_float32: bool = False,
        qk_dim_factor: float = 0.5,
        v_dim_factor: float = 1.0,
        gate_soft_cap: float = 15.0,
        weight_mode: WeightModeType = "single",
        return_last_states: bool = False,
        ffn_proj_factor: float = 2.6667,
        ffn_round_up_to_multiple_of: int = 64,
        chunkwise_kernel: ChunkwiseKernelType = "chunkwise--triton_limit_chunk",
        sequence_kernel: SequenceKernelType = "native_sequence__triton",
        step_kernel: StepKernelType = "triton",
        mode: BackendModeType = "train",
        chunk_size: int = 64,
        autocast_kernel_dtype: DtypeType = "bfloat16",
        eps: float = 1e-6,
        inference_state_dtype: DtypeType = "bfloat16",
        seqlens: Optional[list[int]] = None,
        direction: SequenceTraversal = SequenceTraversal.ROWWISE_FROM_TOP_LEFT,
        num_blocks = 1
    ):
        super().__init__()
        self.config = mLSTMVisionBlockConfig(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            use_bias=use_bias,
            norm_eps=norm_eps,
            norm_reduction_force_float32=norm_reduction_force_float32,
            qk_dim_factor=qk_dim_factor,
            v_dim_factor=v_dim_factor,
            gate_soft_cap=gate_soft_cap,
            weight_mode=weight_mode,
            ffn_proj_factor=ffn_proj_factor,
            ffn_round_up_to_multiple_of=ffn_round_up_to_multiple_of,
            chunkwise_kernel=chunkwise_kernel,
            sequence_kernel=sequence_kernel,
            step_kernel=step_kernel,
            mode=mode,
            chunk_size=chunk_size,
            return_last_states=return_last_states,
            autocast_kernel_dtype=autocast_kernel_dtype,
            eps=eps,
            inference_state_dtype=inference_state_dtype,
            seqlens=seqlens,
            direction=direction,
            num_blocks=12
        )
        self.block = mLSTMBlock(self.config)
        self.direction = direction


    def forward(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            x = x.flip(dims=[1])
        y = self.block(x)
        if self.direction == SequenceTraversal.ROWWISE_FROM_BOT_RIGHT:
            y = y.flip(dims=[1])
        return y
        
    def reset_parameters(self):
        # init inproj
        small_init_init_(self.block.mlstm_layer.up_proj.weight, dim=self.config.embedding_dim)
        if self.block.mlstm_layer.up_proj.bias is not None:
            nn.init.zeros_(self.block.mlstm_layer.up_proj.bias)
        # init outproj
        wang_init_(self.block.mlstm_layer.out_proj.weight, dim=self.config.embedding_dim, num_blocks=self.config.num_blocks)
        if self.block.mlstm_layer.out_proj is not None:
            nn.init.zeros_(self.block.mlstm_layer.out_proj.bias)

        # nn.init.ones_(self.learnable_skip)

        def _init_qkv_proj(qkv_proj):
            # use the embedding dim instead of the inner embedding dim
            small_init_init_(qkv_proj.weight, dim=self.config.embedding_dim)
            if qkv_proj.bias is not None:
                nn.init.zeros_(qkv_proj.bias)

        _init_qkv_proj(self.block.mlstm_layer.q)
        _init_qkv_proj(self.block.mlstm_layer.k)
        _init_qkv_proj(self.block.mlstm_layer.v)


        torch.nn.init.zeros_(self.block.mlstm_layer.fgate_preact.weight)
        bias_linspace_init_(self.block.mlstm_layer.fgate_preact.bias, start=3.0, end=6.0)
        # input gate initialization
        torch.nn.init.zeros_(self.block.mlstm_layer.igate_preact.weight)
        torch.nn.init.normal_(self.block.mlstm_layer.igate_preact.bias, mean=0.0, std=0.1)



        small_init_init_(self.block.ffn.proj_up.weight, dim=self.config.embedding_dim)
        if self.block.ffn.proj_up.bias is not None:
            nn.init.zeros_(self.block.ffn.proj_up.bias)
        wang_init_(
            self.block.ffn.proj_down.weight,
            dim=self.config.embedding_dim,
            num_blocks=self.config.num_blocks,
        )
        if self.block.ffn.proj_down.bias is not None:
            nn.init.zeros_(self.block.ffn.proj_down.bias)



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

def bias_linspace_init_(param: torch.Tensor, start: float = 3.4, end: float = 6.0) -> torch.Tensor:
    """Linearly spaced bias init across dimensions."""
    assert param.dim() == 1, f"param must be 1-dimensional (typically a bias), got {param.dim()}"
    n_dims = param.shape[0]
    init_vals = torch.linspace(start, end, n_dims)
    with torch.no_grad():
        param.copy_(init_vals)
    return param

