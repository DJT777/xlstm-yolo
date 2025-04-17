import collections
import itertools
import math

import einops
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# adapted from timm (timm/models/layers/helpers.py)
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            assert len(x) == n
            return x
        return tuple(itertools.repeat(x, n))

    return parse


# adapted from timm (timm/models/layers/helpers.py)
def to_ntuple(x, n):
    return _ntuple(n=n)(x)


# Interpolates positional embeddings with configurable mode (from pos_embed.py)
def interpolate_sincos(embed, seqlens, mode: str = "bicubic", interpolate_offset: float = None):
    old_dtype = embed.dtype
    assert embed.ndim - 2 == len(seqlens)
    embed = einops.rearrange(embed, "1 ... dim -> 1 dim ...").float()
    if interpolate_offset:
        scale_factor = [(seqlens[i] + interpolate_offset) / embed.size(i + 2) for i in range(len(seqlens))]
        embed = F.interpolate(embed, scale_factor=scale_factor, mode=mode)
    else:
        embed = F.interpolate(embed, size=seqlens, mode=mode)
    embed = einops.rearrange(embed, "1 dim ... -> 1 ... dim")
    return embed.to(old_dtype)


# Sine-cosine positional embedding functions (from pos_embed.py)
def get_sincos_1d_from_seqlen(seqlen: int, dim: int, max_wavelength: int = 10000):
    grid = torch.arange(seqlen, dtype=torch.double)
    return get_sincos_1d_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_sincos_1d_from_grid(grid, dim: int, max_wavelength: int = 10000):
    if dim % 2 == 0:
        padding = None
    else:
        padding = torch.zeros(*grid.shape, 1)
        dim -= 1
    omega = 1. / max_wavelength ** (torch.arange(0, dim, 2, dtype=torch.double) / dim)
    out = grid.unsqueeze(-1) @ omega.unsqueeze(0)
    emb_sin = torch.sin(out)
    emb_cos = torch.cos(out)
    emb = torch.concat([emb_sin, emb_cos], dim=-1).float()
    if padding is None:
        return emb
    else:
        return torch.concat([emb, padding], dim=-1)


def get_sincos_pos_embed_from_seqlens(seqlens, dim: int, max_wavelength: int = 10000, indexing="ij"):
    assert isinstance(seqlens, (tuple, list))
    grids = [torch.arange(seqlen, dtype=torch.double) for seqlen in seqlens]
    if indexing == "xy":
        grids = reversed(grids)
    grid = torch.stack(torch.meshgrid(*grids, indexing=indexing))
    return get_sincos_pos_embed_from_grid(grid=grid, dim=dim, max_wavelength=max_wavelength)


def get_sincos_pos_embed_from_grid(grid, dim: int, max_wavelength: int = 10000):
    ndim = grid.size(0)
    if dim % ndim == 0:
        padding = None
    else:
        padding_dim = dim % ndim
        padding = torch.zeros(*grid.shape[1:], padding_dim)
        dim -= padding_dim
    pos_embed = torch.concat(
        [
            get_sincos_1d_from_grid(grid=grid[i], dim=dim // ndim, max_wavelength=max_wavelength)
            for i in range(ndim)
        ],
        dim=-1,
    )
    if padding is None:
        return pos_embed
    else:
        return torch.concat([pos_embed, padding], dim=-1)


# SequenceConv2d (unchanged from original)
class SequenceConv2d(nn.Conv2d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x):
        assert x.ndim == 3
        if self.seqlens is None:
            h = math.sqrt(x.size(1))
            assert h.is_integer()
            h = int(h)
        else:
            assert len(self.seqlens) == 2
            h = self.seqlens[0]
        x = einops.rearrange(x, "b (h w) d -> b d h w", h=h)
        x = super().forward(x)
        x = einops.rearrange(x, "b d h w -> b (h w) d")
        return x


# New SequenceConv3d for 3D sequences
class SequenceConv3d(nn.Conv3d):
    def __init__(self, *args, seqlens=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seqlens = seqlens

    def forward(self, x):
        assert x.ndim == 3
        if self.seqlens is None:
            total_seq_len = x.size(1)
            d = round(total_seq_len ** (1/3))
            assert d ** 3 == total_seq_len, "Sequence length must be a perfect cube if seqlens is not provided"
            d, h, w = d, d, d
        else:
            assert len(self.seqlens) == 3
            d, h, w = self.seqlens
        x = einops.rearrange(x, "b (d h w) c -> b c d h w", d=d, h=h, w=w)
        x = super().forward(x)
        x = einops.rearrange(x, "b c d h w -> b (d h w) c")
        return x


# VitPatchEmbed (unchanged from original, supports 3D)
class VitPatchEmbed(nn.Module):
    def __init__(self, dim, num_channels, resolution, patch_size, stride=None, init_weights="xavier_uniform"):
        super().__init__()
        self.resolution = resolution
        self.init_weights = init_weights
        self.ndim = len(resolution)
        self.patch_size = to_ntuple(patch_size, n=self.ndim)
        if stride is None:
            self.stride = self.patch_size
        else:
            self.stride = to_ntuple(stride, n=self.ndim)
        for i in range(self.ndim):
            assert resolution[i] % self.patch_size[i] == 0, \
                f"resolution[{i}] % patch_size[{i}] != 0 (resolution={resolution} patch_size={patch_size})"
        self.seqlens = [resolution[i] // self.patch_size[i] for i in range(self.ndim)]
        if self.patch_size == self.stride:
            self.num_patches = int(np.prod(self.seqlens))
        else:
            if self.ndim == 1:
                conv_func = F.conv1d
            elif self.ndim == 2:
                conv_func = F.conv2d
            elif self.ndim == 3:
                conv_func = F.conv3d
            else:
                raise NotImplementedError
            self.num_patches = conv_func(
                input=torch.zeros(1, 1, *resolution),
                weight=torch.zeros(1, 1, *self.patch_size),
                stride=self.stride,
            ).numel()

        if self.ndim == 1:
            conv_ctor = nn.Conv1d
        elif self.ndim == 2:
            conv_ctor = nn.Conv2d
        elif self.ndim == 3:
            conv_ctor = nn.Conv3d
        else:
            raise NotImplementedError

        self.proj = conv_ctor(num_channels, dim, kernel_size=self.patch_size, stride=self.stride)
        self.reset_parameters()

    def reset_parameters(self):
        if self.init_weights == "torch":
            pass
        elif self.init_weights == "xavier_uniform":
            w = self.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            nn.init.zeros_(self.proj.bias)
        else:
            raise NotImplementedError

    def forward(self, x):
        assert all(x.size(i + 2) % self.patch_size[i] == 0 for i in range(self.ndim)), \
            f"x.shape={x.shape} incompatible with patch_size={self.patch_size}"
        x = self.proj(x)
        x = einops.rearrange(x, "b c ... -> b ... c")
        return x


# General VitPosEmbed supporting arbitrary dimensions (from vit_pos_embed.py)
class VitPosEmbed(nn.Module):
    def __init__(
            self,
            seqlens,
            dim: int,
            is_learnable: bool = False,
            allow_interpolation: bool = True,
            interpolate_offset: float = None,
    ):
        super().__init__()
        self.seqlens = seqlens
        self.dim = dim
        self.is_learnable = is_learnable
        self.allow_interpolation = allow_interpolation
        self.interpolate_offset = interpolate_offset
        if is_learnable:
            self.embed = nn.Parameter(torch.zeros(1, *seqlens, dim))
        else:
            self.register_buffer("embed", get_sincos_pos_embed_from_seqlens(seqlens=seqlens, dim=dim).unsqueeze(0))
        self.reset_parameters()

    @property
    def _expected_x_ndim(self):
        return len(self.seqlens) + 2

    def reset_parameters(self):
        if self.is_learnable:
            nn.init.trunc_normal_(self.embed, std=.02)

    def forward(self, x):
        assert x.ndim == self._expected_x_ndim
        if x.shape[1:] != self.embed.shape[1:]:
            assert self.allow_interpolation
            # Select interpolation mode based on number of dimensions
            if len(self.seqlens) == 1:
                mode = "linear"
            elif len(self.seqlens) == 2:
                mode = "bicubic"
            elif len(self.seqlens) == 3:
                mode = "trilinear"
            else:
                raise ValueError(f"Unsupported number of dimensions: {len(self.seqlens)}")
            embed = interpolate_sincos(
                embed=self.embed,
                seqlens=x.shape[1:-1],
                mode=mode,
                interpolate_offset=self.interpolate_offset,
            )
        else:
            embed = self.embed
        return x + embed


# Subclasses for specific dimensions (from vit_pos_embed.py)
class VitPosEmbed1d(VitPosEmbed):
    def __init__(self, seqlens, *args, **kwargs):
        assert len(seqlens) == 1
        super().__init__(seqlens=seqlens, *args, **kwargs)


class VitPosEmbed2d(VitPosEmbed):
    def __init__(self, seqlens, *args, **kwargs):
        assert len(seqlens) == 2
        super().__init__(seqlens=seqlens, *args, **kwargs)


class VitPosEmbed3d(VitPosEmbed):
    def __init__(self, seqlens, *args, **kwargs):
        assert len(seqlens) == 3
        super().__init__(seqlens=seqlens, *args, **kwargs)


# DropPath (unchanged from original)
class DropPath(nn.Sequential):
    def __init__(
            self,
            *args,
            drop_prob: float = 0.,
            scale_by_keep: bool = True,
            stochastic_drop_prob: bool = False,
            drop_prob_tolerance: float = 0.01,
    ):
        super().__init__(*args)
        assert 0. <= drop_prob < 1.
        self._drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        self.stochastic_drop_prob = stochastic_drop_prob
        self.drop_prob_tolerance = drop_prob_tolerance

    @property
    def drop_prob(self):
        return self._drop_prob

    @drop_prob.setter
    def drop_prob(self, value):
        assert 0. <= value < 1.
        self._drop_prob = value

    @property
    def keep_prob(self):
        return 1. - self.drop_prob

    def forward(self, x, residual_path=None, residual_path_kwargs=None):
        assert (len(self) == 0) ^ (residual_path is None)
        residual_path_kwargs = residual_path_kwargs or {}
        if self.drop_prob == 0. or not self.training:
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs)
            else:
                return x + residual_path(x, **residual_path_kwargs)
        bs = len(x)
        keep_count = max(int(bs * self.keep_prob), 1)
        actual_keep_prob = keep_count / bs
        drop_path_delta = self.keep_prob - actual_keep_prob
        if self.stochastic_drop_prob or drop_path_delta > self.drop_prob_tolerance:
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = x.new_empty(shape).bernoulli_(self.keep_prob)
            if self.scale_by_keep:
                random_tensor.div_(self.keep_prob)
            if residual_path is None:
                return x + super().forward(x, **residual_path_kwargs) * random_tensor
            else:
                return x + residual_path(x, **residual_path_kwargs) * random_tensor
        scale = bs / keep_count
        perm = torch.randperm(bs, device=x.device)[:keep_count]
        if self.scale_by_keep:
            alpha = scale
        else:
            alpha = 1.
        residual_path_kwargs = {
            key: value[perm] if torch.is_tensor(value) else value
            for key, value in residual_path_kwargs.items()
        }
        if residual_path is None:
            residual = super().forward(x[perm], **residual_path_kwargs)
        else:
            residual = residual_path(x[perm], **residual_path_kwargs)
        return torch.index_add(
            x.flatten(start_dim=1),
            dim=0,
            index=perm,
            source=residual.to(x.dtype).flatten(start_dim=1),
            alpha=alpha,
        ).view_as(x)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob, 3):0.3f}'