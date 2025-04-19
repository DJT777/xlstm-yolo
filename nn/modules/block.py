# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Block modules."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.utils.torch_utils import fuse_conv_and_bn

from .conv import Conv, DWConv, GhostConv, LightConv, RepConv, autopad
from .transformer import TransformerBlock
from .vision_lstm.vision_lstm2 import VitPatchEmbed, VitPosEmbed2d, ViLBlockPair, SequenceConv2d, LayerNorm, MultiHeadLayerNorm
from .vision_lstm.vision_lstm_hierarchical import PatchMerge, MultiScaleFusion
from .vision_lstm.vision_lstm_util import VitPosEmbed, DropPath  # Adjust import path as needed
  # run once, before Ultralytics builds the optimiser


# Import from vision_lstm.py (assuming it's in the same directory or a submodule)

__all__ = (
    "DFL",
    "HGBlock",
    "HGStem",
    "SPP",
    "SPPF",
    "C1",
    "C2",
    "C3",
    "C2f",
    "C2fAttn",
    "ImagePoolingAttn",
    "ContrastiveHead",
    "BNContrastiveHead",
    "C3x",
    "C3TR",
    "C3Ghost",
    "GhostBottleneck",
    "Bottleneck",
    "BottleneckCSP",
    "Proto",
    "RepC3",
    "ResNetLayer",
    "RepNCSPELAN4",
    "ELAN1",
    "ADown",
    "AConv",
    "SPPELAN",
    "CBFuse",
    "CBLinear",
    "C3k2",
    "C2fPSA",
    "C2PSA",
    "RepVGGDW",
    "CIB",
    "C2fCIB",
    "Attention",
    "PSA",
    "SCDown",
    "TorchVision",
    "VisionLSTM",
    "VisionLSTMTorch",
    "FeatureSplitIndex",
    "VitPatchEmbedBlock",
    "VitPosEmbed2dBlock",
    "ViLBlockPairBlock",
    "PatchMergeBlock",
    "XViLBlockPairBlock"
    "ViLFusionBlock"
    "ViLLayerNormBlock"
)


class DFL(nn.Module):
    """
    Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1=16):
        """Initialize a convolutional layer with a given number of input channels."""
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        """Applies a transformer layer on input tensor 'x' and returns a tensor."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Proto(nn.Module):
    """YOLOv8 mask Proto module for segmentation models."""

    def __init__(self, c1, c_=256, c2=32):
        """
        Initializes the YOLOv8 mask Proto module with specified number of protos and masks.

        Input arguments are ch_in, number of protos, number of masks.
        """
        super().__init__()
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.ConvTranspose2d(c_, c_, 2, 2, 0, bias=True)  # nn.Upsample(scale_factor=2, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        """Performs a forward pass through layers using an upsampled input image."""
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))


class HGStem(nn.Module):
    """
    StemBlock of PPHGNetV2 with 5 convolutions and one maxpool2d.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2):
        """Initialize the SPP layer with input/output channels and specified kernel sizes for max pooling."""
        super().__init__()
        self.stem1 = Conv(c1, cm, 3, 2, act=nn.ReLU())
        self.stem2a = Conv(cm, cm // 2, 2, 1, 0, act=nn.ReLU())
        self.stem2b = Conv(cm // 2, cm, 2, 1, 0, act=nn.ReLU())
        self.stem3 = Conv(cm * 2, cm, 3, 2, act=nn.ReLU())
        self.stem4 = Conv(cm, c2, 1, 1, act=nn.ReLU())
        self.pool = nn.MaxPool2d(kernel_size=2, stride=1, padding=0, ceil_mode=True)

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        x = self.stem1(x)
        x = F.pad(x, [0, 1, 0, 1])
        x2 = self.stem2a(x)
        x2 = F.pad(x2, [0, 1, 0, 1])
        x2 = self.stem2b(x2)
        x1 = self.pool(x)
        x = torch.cat([x1, x2], dim=1)
        x = self.stem3(x)
        x = self.stem4(x)
        return x


class HGBlock(nn.Module):
    """
    HG_Block of PPHGNetV2 with 2 convolutions and LightConv.

    https://github.com/PaddlePaddle/PaddleDetection/blob/develop/ppdet/modeling/backbones/hgnet_v2.py
    """

    def __init__(self, c1, cm, c2, k=3, n=6, lightconv=False, shortcut=False, act=nn.ReLU()):
        """Initializes a CSP Bottleneck with 1 convolution using specified input and output channels."""
        super().__init__()
        block = LightConv if lightconv else Conv
        self.m = nn.ModuleList(block(c1 if i == 0 else cm, cm, k=k, act=act) for i in range(n))
        self.sc = Conv(c1 + n * cm, c2 // 2, 1, 1, act=act)  # squeeze conv
        self.ec = Conv(c2 // 2, c2, 1, 1, act=act)  # excitation conv
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Forward pass of a PPHGNetV2 backbone layer."""
        y = [x]
        y.extend(m(y[-1]) for m in self.m)
        y = self.ec(self.sc(torch.cat(y, 1)))
        return y + x if self.add else y


class SPP(nn.Module):
    """Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729."""

    def __init__(self, c1, c2, k=(5, 9, 13)):
        """Initialize the SPP layer with input/output channels and pooling kernel sizes."""
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        """Forward pass of the SPP layer, performing spatial pyramid pooling."""
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


class C1(nn.Module):
    """CSP Bottleneck with 1 convolution."""

    def __init__(self, c1, c2, n=1):
        """Initializes the CSP Bottleneck with configurations for 1 convolution with arguments ch_in, ch_out, number."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.m = nn.Sequential(*(Conv(c2, c2, 3) for _ in range(n)))

    def forward(self, x):
        """Applies cross-convolutions to input in the C3 module."""
        y = self.cv1(x)
        return self.m(y) + y


class C2(nn.Module):
    """CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes a CSP Bottleneck with 2 convolutions and optional shortcut connection."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c2, 1)  # optional act=FReLU(c2)
        # self.attention = ChannelAttention(2 * self.c)  # or SpatialAttention()
        self.m = nn.Sequential(*(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        a, b = self.cv1(x).chunk(2, 1)
        return self.cv2(torch.cat((self.m(a), b), 1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = self.cv1(x).split((self.c, self.c), 1)
        y = [y[0], y[1]]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    """C3 module with cross-convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3TR instance and set default parameters."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.c_ = int(c2 * e)
        self.m = nn.Sequential(*(Bottleneck(self.c_, self.c_, shortcut, g, k=((1, 3), (3, 1)), e=1) for _ in range(n)))


class RepC3(nn.Module):
    """Rep C3."""

    def __init__(self, c1, c2, n=3, e=1.0):
        """Initialize CSP Bottleneck with a single convolution using input channels, output channels, and number."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.m = nn.Sequential(*[RepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1, 1) if c_ != c2 else nn.Identity()

    def forward(self, x):
        """Forward pass of RT-DETR neck layer."""
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))


class C3TR(C3):
    """C3 module with TransformerBlock()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize C3Ghost module with GhostBottleneck()."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3Ghost(C3):
    """C3 module with GhostBottleneck()."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize 'SPP' module with various pooling sizes for spatial pyramid pooling."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class GhostBottleneck(nn.Module):
    """Ghost Bottleneck https://github.com/huawei-noah/ghostnet."""

    def __init__(self, c1, c2, k=3, s=1):
        """Initializes GhostBottleneck module with arguments ch_in, ch_out, kernel, stride."""
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),  # pw-linear
        )
        self.shortcut = (
            nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()
        )

    def forward(self, x):
        """Applies skip connection and concatenation to input tensor."""
        return self.conv(x) + self.shortcut(x)


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    """CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes the CSP Bottleneck given arguments for ch_in, ch_out, number, shortcut, groups, expansion."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        """Applies a CSP bottleneck with 3 convolutions."""
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class ResNetBlock(nn.Module):
    """ResNet block with standard convolution layers."""

    def __init__(self, c1, c2, s=1, e=4):
        """Initialize convolution with given parameters."""
        super().__init__()
        c3 = e * c2
        self.cv1 = Conv(c1, c2, k=1, s=1, act=True)
        self.cv2 = Conv(c2, c2, k=3, s=s, p=1, act=True)
        self.cv3 = Conv(c2, c3, k=1, act=False)
        self.shortcut = nn.Sequential(Conv(c1, c3, k=1, s=s, act=False)) if s != 1 or c1 != c3 else nn.Identity()

    def forward(self, x):
        """Forward pass through the ResNet block."""
        return F.relu(self.cv3(self.cv2(self.cv1(x))) + self.shortcut(x))


class ResNetLayer(nn.Module):
    """ResNet layer with multiple ResNet blocks."""

    def __init__(self, c1, c2, s=1, is_first=False, n=1, e=4):
        """Initializes the ResNetLayer given arguments."""
        super().__init__()
        self.is_first = is_first

        if self.is_first:
            self.layer = nn.Sequential(
                Conv(c1, c2, k=7, s=2, p=3, act=True), nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
        else:
            blocks = [ResNetBlock(c1, c2, s, e=e)]
            blocks.extend([ResNetBlock(e * c2, c2, 1, e=e) for _ in range(n - 1)])
            self.layer = nn.Sequential(*blocks)

    def forward(self, x):
        """Forward pass through the ResNet layer."""
        return self.layer(x)


class MaxSigmoidAttnBlock(nn.Module):
    """Max Sigmoid attention block."""

    def __init__(self, c1, c2, nh=1, ec=128, gc=512, scale=False):
        """Initializes MaxSigmoidAttnBlock with specified arguments."""
        super().__init__()
        self.nh = nh
        self.hc = c2 // nh
        self.ec = Conv(c1, ec, k=1, act=False) if c1 != ec else None
        self.gl = nn.Linear(gc, ec)
        self.bias = nn.Parameter(torch.zeros(nh))
        self.proj_conv = Conv(c1, c2, k=3, s=1, act=False)
        self.scale = nn.Parameter(torch.ones(1, nh, 1, 1)) if scale else 1.0

    def forward(self, x, guide):
        """Forward process."""
        bs, _, h, w = x.shape

        guide = self.gl(guide)
        guide = guide.view(bs, -1, self.nh, self.hc)
        embed = self.ec(x) if self.ec is not None else x
        embed = embed.view(bs, self.nh, self.hc, h, w)

        aw = torch.einsum("bmchw,bnmc->bmhwn", embed, guide)
        aw = aw.max(dim=-1)[0]
        aw = aw / (self.hc**0.5)
        aw = aw + self.bias[None, :, None, None]
        aw = aw.sigmoid() * self.scale

        x = self.proj_conv(x)
        x = x.view(bs, self.nh, -1, h, w)
        x = x * aw.unsqueeze(2)
        return x.view(bs, -1, h, w)


class C2fAttn(nn.Module):
    """C2f module with an additional attn module."""

    def __init__(self, c1, c2, n=1, ec=128, nh=1, gc=512, shortcut=False, g=1, e=0.5):
        """Initializes C2f module with attention mechanism for enhanced feature extraction and processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((3 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.attn = MaxSigmoidAttnBlock(self.c, self.c, gc=gc, ec=ec, nh=nh)

    def forward(self, x, guide):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x, guide):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        y.append(self.attn(y[-1], guide))
        return self.cv2(torch.cat(y, 1))


class ImagePoolingAttn(nn.Module):
    """ImagePoolingAttn: Enhance the text embeddings with image-aware information."""

    def __init__(self, ec=256, ch=(), ct=512, nh=8, k=3, scale=False):
        """Initializes ImagePoolingAttn with specified arguments."""
        super().__init__()

        nf = len(ch)
        self.query = nn.Sequential(nn.LayerNorm(ct), nn.Linear(ct, ec))
        self.key = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.value = nn.Sequential(nn.LayerNorm(ec), nn.Linear(ec, ec))
        self.proj = nn.Linear(ec, ct)
        self.scale = nn.Parameter(torch.tensor([0.0]), requires_grad=True) if scale else 1.0
        self.projections = nn.ModuleList([nn.Conv2d(in_channels, ec, kernel_size=1) for in_channels in ch])
        self.im_pools = nn.ModuleList([nn.AdaptiveMaxPool2d((k, k)) for _ in range(nf)])
        self.ec = ec
        self.nh = nh
        self.nf = nf
        self.hc = ec // nh
        self.k = k

    def forward(self, x, text):
        """Executes attention mechanism on input tensor x and guide tensor."""
        bs = x[0].shape[0]
        assert len(x) == self.nf
        num_patches = self.k**2
        x = [pool(proj(x)).view(bs, -1, num_patches) for (x, proj, pool) in zip(x, self.projections, self.im_pools)]
        x = torch.cat(x, dim=-1).transpose(1, 2)
        q = self.query(text)
        k = self.key(x)
        v = self.value(x)

        # q = q.reshape(1, text.shape[1], self.nh, self.hc).repeat(bs, 1, 1, 1)
        q = q.reshape(bs, -1, self.nh, self.hc)
        k = k.reshape(bs, -1, self.nh, self.hc)
        v = v.reshape(bs, -1, self.nh, self.hc)

        aw = torch.einsum("bnmc,bkmc->bmnk", q, k)
        aw = aw / (self.hc**0.5)
        aw = F.softmax(aw, dim=-1)

        x = torch.einsum("bmnk,bkmc->bnmc", aw, v)
        x = self.proj(x.reshape(bs, -1, self.ec))
        return x * self.scale + text


class ContrastiveHead(nn.Module):
    """Implements contrastive learning head for region-text similarity in vision-language models."""

    def __init__(self):
        """Initializes ContrastiveHead with specified region-text similarity parameters."""
        super().__init__()
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.tensor(1 / 0.07).log())

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = F.normalize(x, dim=1, p=2)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class BNContrastiveHead(nn.Module):
    """
    Batch Norm Contrastive Head for YOLO-World using batch norm instead of l2-normalization.

    Args:
        embed_dims (int): Embed dimensions of text and image features.
    """

    def __init__(self, embed_dims: int):
        """Initialize ContrastiveHead with region-text similarity parameters."""
        super().__init__()
        self.norm = nn.BatchNorm2d(embed_dims)
        # NOTE: use -10.0 to keep the init cls loss consistency with other losses
        self.bias = nn.Parameter(torch.tensor([-10.0]))
        # use -1.0 is more stable
        self.logit_scale = nn.Parameter(-1.0 * torch.ones([]))

    def forward(self, x, w):
        """Forward function of contrastive learning."""
        x = self.norm(x)
        w = F.normalize(w, dim=-1, p=2)
        x = torch.einsum("bchw,bkc->bkhw", x, w)
        return x * self.logit_scale.exp() + self.bias


class RepBottleneck(Bottleneck):
    """Rep bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a RepBottleneck module with customizable in/out channels, shortcuts, groups and expansion."""
        super().__init__(c1, c2, shortcut, g, k, e)
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = RepConv(c1, c_, k[0], 1)


class RepCSP(C3):
    """Repeatable Cross Stage Partial Network (RepCSP) module for efficient feature extraction."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initializes RepCSP layer with given channels, repetitions, shortcut, groups and expansion ratio."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))


class RepNCSPELAN4(nn.Module):
    """CSP-ELAN."""

    def __init__(self, c1, c2, c3, c4, n=1):
        """Initializes CSP-ELAN layer with specified channel sizes, repetitions, and convolutions."""
        super().__init__()
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.Sequential(RepCSP(c3 // 2, c4, n), Conv(c4, c4, 3, 1))
        self.cv3 = nn.Sequential(RepCSP(c4, c4, n), Conv(c4, c4, 3, 1))
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)

    def forward(self, x):
        """Forward pass through RepNCSPELAN4 layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend((m(y[-1])) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3])
        return self.cv4(torch.cat(y, 1))


class ELAN1(RepNCSPELAN4):
    """ELAN1 module with 4 convolutions."""

    def __init__(self, c1, c2, c3, c4):
        """Initializes ELAN1 layer with specified channel sizes."""
        super().__init__(c1, c2, c3, c4)
        self.c = c3 // 2
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = Conv(c3 // 2, c4, 3, 1)
        self.cv3 = Conv(c4, c4, 3, 1)
        self.cv4 = Conv(c3 + (2 * c4), c2, 1, 1)


class AConv(nn.Module):
    """AConv."""

    def __init__(self, c1, c2):
        """Initializes AConv module with convolution layers."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 3, 2, 1)

    def forward(self, x):
        """Forward pass through AConv layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        return self.cv1(x)


class ADown(nn.Module):
    """ADown."""

    def __init__(self, c1, c2):
        """Initializes ADown module with convolution layers to downsample input from channels c1 to c2."""
        super().__init__()
        self.c = c2 // 2
        self.cv1 = Conv(c1 // 2, self.c, 3, 2, 1)
        self.cv2 = Conv(c1 // 2, self.c, 1, 1, 0)

    def forward(self, x):
        """Forward pass through ADown layer."""
        x = torch.nn.functional.avg_pool2d(x, 2, 1, 0, False, True)
        x1, x2 = x.chunk(2, 1)
        x1 = self.cv1(x1)
        x2 = torch.nn.functional.max_pool2d(x2, 3, 2, 1)
        x2 = self.cv2(x2)
        return torch.cat((x1, x2), 1)


class SPPELAN(nn.Module):
    """SPP-ELAN."""

    def __init__(self, c1, c2, c3, k=5):
        """Initializes SPP-ELAN block with convolution and max pooling layers for spatial pyramid pooling."""
        super().__init__()
        self.c = c3
        self.cv1 = Conv(c1, c3, 1, 1)
        self.cv2 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv3 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv4 = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c3, c2, 1, 1)

    def forward(self, x):
        """Forward pass through SPPELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in [self.cv2, self.cv3, self.cv4])
        return self.cv5(torch.cat(y, 1))


class CBLinear(nn.Module):
    """CBLinear."""

    def __init__(self, c1, c2s, k=1, s=1, p=None, g=1):
        """Initializes the CBLinear module, passing inputs unchanged."""
        super().__init__()
        self.c2s = c2s
        self.conv = nn.Conv2d(c1, sum(c2s), k, s, autopad(k, p), groups=g, bias=True)

    def forward(self, x):
        """Forward pass through CBLinear layer."""
        return self.conv(x).split(self.c2s, dim=1)


class CBFuse(nn.Module):
    """CBFuse."""

    def __init__(self, idx):
        """Initializes CBFuse module with layer index for selective feature fusion."""
        super().__init__()
        self.idx = idx

    def forward(self, xs):
        """Forward pass through CBFuse layer."""
        target_size = xs[-1].shape[2:]
        res = [F.interpolate(x[self.idx[i]], size=target_size, mode="nearest") for i, x in enumerate(xs[:-1])]
        return torch.sum(torch.stack(res + xs[-1:]), dim=0)


class C3f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv((2 + n) * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(c_, c_, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = [self.cv2(x), self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        return self.cv3(torch.cat(y, 1))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class RepVGGDW(torch.nn.Module):
    """RepVGGDW is a class that represents a depth wise separable convolutional block in RepVGG architecture."""

    def __init__(self, ed) -> None:
        """Initializes RepVGGDW with depthwise separable convolutional layers for efficient processing."""
        super().__init__()
        self.conv = Conv(ed, ed, 7, 1, 3, g=ed, act=False)
        self.conv1 = Conv(ed, ed, 3, 1, 1, g=ed, act=False)
        self.dim = ed
        self.act = nn.SiLU()

    def forward(self, x):
        """
        Performs a forward pass of the RepVGGDW block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x) + self.conv1(x))

    def forward_fuse(self, x):
        """
        Performs a forward pass of the RepVGGDW block without fusing the convolutions.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor after applying the depth wise separable convolution.
        """
        return self.act(self.conv(x))

    @torch.no_grad()
    def fuse(self):
        """
        Fuses the convolutional layers in the RepVGGDW block.

        This method fuses the convolutional layers and updates the weights and biases accordingly.
        """
        conv = fuse_conv_and_bn(self.conv.conv, self.conv.bn)
        conv1 = fuse_conv_and_bn(self.conv1.conv, self.conv1.bn)

        conv_w = conv.weight
        conv_b = conv.bias
        conv1_w = conv1.weight
        conv1_b = conv1.bias

        conv1_w = torch.nn.functional.pad(conv1_w, [2, 2, 2, 2])

        final_conv_w = conv_w + conv1_w
        final_conv_b = conv_b + conv1_b

        conv.weight.data.copy_(final_conv_w)
        conv.bias.data.copy_(final_conv_b)

        self.conv = conv
        del self.conv1


class CIB(nn.Module):
    """
    Conditional Identity Block (CIB) module.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        shortcut (bool, optional): Whether to add a shortcut connection. Defaults to True.
        e (float, optional): Scaling factor for the hidden channels. Defaults to 0.5.
        lk (bool, optional): Whether to use RepVGGDW for the third convolutional layer. Defaults to False.
    """

    def __init__(self, c1, c2, shortcut=True, e=0.5, lk=False):
        """Initializes the custom model with optional shortcut, scaling factor, and RepVGGDW layer."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = nn.Sequential(
            Conv(c1, c1, 3, g=c1),
            Conv(c1, 2 * c_, 1),
            RepVGGDW(2 * c_) if lk else Conv(2 * c_, 2 * c_, 3, g=2 * c_),
            Conv(2 * c_, c2, 1),
            Conv(c2, c2, 3, g=c2),
        )

        self.add = shortcut and c1 == c2

    def forward(self, x):
        """
        Forward pass of the CIB module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            (torch.Tensor): Output tensor.
        """
        return x + self.cv1(x) if self.add else self.cv1(x)


class C2fCIB(C2f):
    """
    C2fCIB class represents a convolutional block with C2f and CIB modules.

    Args:
        c1 (int): Number of input channels.
        c2 (int): Number of output channels.
        n (int, optional): Number of CIB modules to stack. Defaults to 1.
        shortcut (bool, optional): Whether to use shortcut connection. Defaults to False.
        lk (bool, optional): Whether to use local key connection. Defaults to False.
        g (int, optional): Number of groups for grouped convolution. Defaults to 1.
        e (float, optional): Expansion ratio for CIB modules. Defaults to 0.5.
    """

    def __init__(self, c1, c2, n=1, shortcut=False, lk=False, g=1, e=0.5):
        """Initializes the module with specified parameters for channel, shortcut, local key, groups, and expansion."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(CIB(self.c, self.c, shortcut, e=1.0, lk=lk) for _ in range(n))


class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x


class PSABlock(nn.Module):
    """
    PSABlock class implementing a Position-Sensitive Attention block for neural networks.

    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers
    with optional shortcut connections.

    Attributes:
        attn (Attention): Multi-head attention module.
        ffn (nn.Sequential): Feed-forward neural network module.
        add (bool): Flag indicating whether to add shortcut connections.

    Methods:
        forward: Performs a forward pass through the PSABlock, applying attention and feed-forward layers.

    Examples:
        Create a PSABlock and perform a forward pass
        >>> psablock = PSABlock(c=128, attn_ratio=0.5, num_heads=4, shortcut=True)
        >>> input_tensor = torch.randn(1, 128, 32, 32)
        >>> output_tensor = psablock(input_tensor)
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSA(nn.Module):
    """
    PSA class for implementing Position-Sensitive Attention in neural networks.

    This class encapsulates the functionality for applying position-sensitive attention and feed-forward networks to
    input tensors, enhancing feature extraction and processing capabilities.

    Attributes:
        c (int): Number of hidden channels after applying the initial convolution.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        attn (Attention): Attention module for position-sensitive attention.
        ffn (nn.Sequential): Feed-forward network for further processing.

    Methods:
        forward: Applies position-sensitive attention and feed-forward network to the input tensor.

    Examples:
        Create a PSA module and apply it to an input tensor
        >>> psa = PSA(c1=128, c2=128, e=0.5)
        >>> input_tensor = torch.randn(1, 128, 64, 64)
        >>> output_tensor = psa.forward(input_tensor)
    """

    def __init__(self, c1, c2, e=0.5):
        """Initializes the PSA module with input/output channels and attention mechanism for feature extraction."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=self.c // 64)
        self.ffn = nn.Sequential(Conv(self.c, self.c * 2, 1), Conv(self.c * 2, self.c, 1, act=False))

    def forward(self, x):
        """Executes forward pass in PSA module, applying attention and feed-forward layers to the input tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSA(nn.Module):
    """
    C2PSA module with attention mechanism for enhanced feature extraction and processing.

    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.Sequential): Sequential container of PSABlock modules for attention and feed-forward operations.

    Methods:
        forward: Performs a forward pass through the C2PSA module, applying attention and feed-forward operations.

    Notes:
        This module essentially is the same as PSA module, but refactored to allow stacking more PSABlock modules.

    Examples:
        >>> c2psa = C2PSA(c1=256, c2=256, n=3, e=0.5)
        >>> input_tensor = torch.randn(1, 256, 64, 64)
        >>> output_tensor = c2psa(input_tensor)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSA(C2f):
    """
    C2fPSA module with enhanced feature extraction using PSA blocks.

    This class extends the C2f module by incorporating PSA blocks for improved attention mechanisms and feature extraction.

    Attributes:
        c (int): Number of hidden channels.
        cv1 (Conv): 1x1 convolution layer to reduce the number of input channels to 2*c.
        cv2 (Conv): 1x1 convolution layer to reduce the number of output channels to c.
        m (nn.ModuleList): List of PSA blocks for feature extraction.

    Methods:
        forward: Performs a forward pass through the C2fPSA module.
        forward_split: Performs a forward pass using split() instead of chunk().

    Examples:
        >>> import torch
        >>> from ultralytics.models.common import C2fPSA
        >>> model = C2fPSA(c1=64, c2=64, n=3, e=0.5)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> output = model(x)
        >>> print(output.shape)
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2fPSA module, a variant of C2f with PSA blocks for enhanced feature extraction."""
        assert c1 == c2
        super().__init__(c1, c2, n=n, e=e)
        self.m = nn.ModuleList(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n))


class SCDown(nn.Module):
    """
    SCDown module for downsampling with separable convolutions.

    This module performs downsampling using a combination of pointwise and depthwise convolutions, which helps in
    efficiently reducing the spatial dimensions of the input tensor while maintaining the channel information.

    Attributes:
        cv1 (Conv): Pointwise convolution layer that reduces the number of channels.
        cv2 (Conv): Depthwise convolution layer that performs spatial downsampling.

    Methods:
        forward: Applies the SCDown module to the input tensor.

    Examples:
        >>> import torch
        >>> from ultralytics import SCDown
        >>> model = SCDown(c1=64, c2=128, k=3, s=2)
        >>> x = torch.randn(1, 64, 128, 128)
        >>> y = model(x)
        >>> print(y.shape)
        torch.Size([1, 128, 64, 64])
    """

    def __init__(self, c1, c2, k, s):
        """Initializes the SCDown module with specified input/output channels, kernel size, and stride."""
        super().__init__()
        self.cv1 = Conv(c1, c2, 1, 1)
        self.cv2 = Conv(c2, c2, k=k, s=s, g=c2, act=False)

    def forward(self, x):
        """Applies convolution and downsampling to the input tensor in the SCDown module."""
        return self.cv2(self.cv1(x))


class TorchVision(nn.Module):
    """
    TorchVision module to allow loading any torchvision model.

    This class provides a way to load a model from the torchvision library, optionally load pre-trained weights, and customize the model by truncating or unwrapping layers.

    Attributes:
        m (nn.Module): The loaded torchvision model, possibly truncated and unwrapped.

    Args:
        model (str): Name of the torchvision model to load.
        weights (str, optional): Pre-trained weights to load. Default is "DEFAULT".
        unwrap (bool, optional): If True, unwraps the model to a sequential containing all but the last `truncate` layers. Default is True.
        truncate (int, optional): Number of layers to truncate from the end if `unwrap` is True. Default is 2.
        split (bool, optional): Returns output from intermediate child modules as list. Default is False.
    """

    def __init__(self, model, weights="DEFAULT", unwrap=True, truncate=2, split=False):
        """Load the model and weights from torchvision."""
        import torchvision  # scope for faster 'import ultralytics'

        super().__init__()
        if hasattr(torchvision.models, "get_model"):
            self.m = torchvision.models.get_model(model, weights=weights)
        else:
            self.m = torchvision.models.__dict__[model](pretrained=bool(weights))
        if unwrap:
            layers = list(self.m.children())
            if isinstance(layers[0], nn.Sequential):  # Second-level for some models like EfficientNet, Swin
                layers = [*list(layers[0].children()), *layers[1:]]
            self.m = nn.Sequential(*(layers[:-truncate] if truncate else layers))
            self.split = split
        else:
            self.split = False
            self.m.head = self.m.heads = nn.Identity()

    def forward(self, x):
        """Forward pass through the model."""
        if self.split:
            y = [x]
            y.extend(m(y[-1]) for m in self.m)
        else:
            y = self.m(x)
        return y



################################################################################
# 1) VisionLSTM - multi-output wrapper
################################################################################
import einops

class VisionLSTMTorch(nn.Module):
    """
    A custom YOLO/Ultralytics block that loads 'VisionLSTM2' from the
    torch.hub repo 'nx-ai/vision-lstm' and returns multiple outputs
    at specified block indices as well as the final feature map.

    Usage in a YOLO-style YAML:
      - [-1, 1, VisionLSTM, [192, {
          "depth": 12,
          "output_indices": [3, 7, 11],
          "mode": "features",
          "pooling": "to_image",
          ...
      }]]
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Ultralytics parse_model typically provides (c1, dim, config) as *args
        # e.g. args = [<input_channels>, <dim=192>, <dict_with_keys>]
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        if len(args) < 2:
            raise ValueError("VisionLSTM requires at least (c1, dim) arguments.")

        # Extract c1, dim, config
        self.c1 = args[0]    # input channels from the previous layer
        self.dim = args[1]   # e.g. 192
        self.config = args[2] if len(args) > 2 else {}

        # Pop custom hyperparameters that Torch Hub's VisionLSTM2 doesn't expect
        self.depth = self.config.pop("depth", 12)
        self.output_indices = self.config.pop("output_indices", [])
        self.mode = self.config.pop("mode", "features")
        self.pooling = self.config.pop("pooling", "to_image")

        # Validate output_indices vs. depth
        max_idx = max(self.output_indices) if self.output_indices else 0
        if self.depth <= max_idx:
            raise ValueError(
                f"Config depth={self.depth} is too small for output_indices={self.output_indices}. "
                "Increase depth or reduce output_indices."
            )

        # Load VisionLSTM2 from torch.hub, overriding 'pooling' with None, if we plan to do custom pooling
        self.m = torch.hub.load(
            "nx-ai/vision-lstm",
            "VisionLSTM2",
            dim=self.dim,
            depth=self.depth,
            mode=self.mode,
            pooling=None,  # we'll do manual pooling in forward
            **self.config
        )

    def forward(self, x):
        """
        Forward pass:
          1) Patch + pos embed
          2) Pass through each block in .blocks
          3) Collect partial outputs at self.output_indices
          4) Final norm & optional "to_image" reshape
          5) Return a list of partial + final outputs
        """
        # 1) Patch embed & pos embed
        x = self.m.patch_embed(x)
        x = self.m.pos_embed(x)
        x = einops.einops.rearrange(x, "b h w d -> b (h w) d")  # Flatten to (B, H*W, dim)

        partial_outputs = []

        # 2) Pass through each block in self.m.blocks
        for i, block in enumerate(self.m.blocks):
            x = block(x)
            if i in self.output_indices:
                seqlen_h, seqlen_w = self.m.patch_embed.seqlens
                # IMPORTANT: use (h w) on left side to match h,w on right
                out = einops.einops.rearrange(
                    x, "b (h w) d -> b d h w",
                    h=seqlen_h, w=seqlen_w
                )
                out = self.m.norm(out)
                partial_outputs.append(out)

        # 3) Final normalization
        x = self.m.norm(x)

        # 4) Manual pooling if "to_image"
        if self.pooling == "to_image":
            if hasattr(self.m, "legacy_norm"):
                x = self.m.legacy_norm(x)
            seqlen_h, seqlen_w = self.m.patch_embed.seqlens
            x = einops.einops.rearrange(
                x, "b (h w) d -> b d h w",
                h=seqlen_h, w=seqlen_w
            )

        # 5) Append final output
        partial_outputs.append(x)
        return partial_outputs



################################################################################
# 2) FeatureSplitIndex - splits one feature from a list
################################################################################
class FeatureSplitIndex(nn.Module):
    def __init__(self, index, *args, **kwargs):
        super().__init__()
        self.index = index
    def forward(self, x):
        # print(self.index)
        if not isinstance(x, (list, tuple)):
            raise ValueError("Input must be a list or tuple of tensors")
        if self.index >= len(x):
            raise ValueError(f"Index {self.index} out of range for list of length {len(x)}")
        return x[self.index]



class AAttn(nn.Module):
    """
    Area-attention module for YOLO models, providing efficient attention mechanisms.

    This module implements an area-based attention mechanism that processes input features in a spatially-aware manner,
    making it particularly effective for object detection tasks.

    Attributes:
        area (int): Number of areas the feature map is divided.
        num_heads (int): Number of heads into which the attention mechanism is divided.
        head_dim (int): Dimension of each attention head.
        qkv (Conv): Convolution layer for computing query, key and value tensors.
        proj (Conv): Projection convolution layer.
        pe (Conv): Position encoding convolution layer.

    Methods:
        forward: Applies area-attention to input tensor.

    Examples:
        >>> attn = AAttn(dim=256, num_heads=8, area=4)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = attn(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, area=1):
        """
        Initializes an Area-attention module for YOLO models.

        Args:
            dim (int): Number of hidden channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            area (int): Number of areas the feature map is divided, default is 1.
        """
        super().__init__()
        self.area = area

        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        all_head_dim = head_dim * self.num_heads

        self.qkv = Conv(dim, all_head_dim * 3, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x):
        """Processes the input tensor 'x' through the area-attention."""
        B, C, H, W = x.shape
        N = H * W

        qkv = self.qkv(x).flatten(2).transpose(1, 2)
        if self.area > 1:
            qkv = qkv.reshape(B * self.area, N // self.area, C * 3)
            B, N, _ = qkv.shape
        q, k, v = (
            qkv.view(B, N, self.num_heads, self.head_dim * 3)
            .permute(0, 2, 3, 1)
            .split([self.head_dim, self.head_dim, self.head_dim], dim=2)
        )
        attn = (q.transpose(-2, -1) @ k) * (self.head_dim**-0.5)
        attn = attn.softmax(dim=-1)
        x = v @ attn.transpose(-2, -1)
        x = x.permute(0, 3, 1, 2)
        v = v.permute(0, 3, 1, 2)

        if self.area > 1:
            x = x.reshape(B // self.area, N * self.area, C)
            v = v.reshape(B // self.area, N * self.area, C)
            B, N, _ = x.shape

        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        v = v.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

        x = x + self.pe(v)
        return self.proj(x)


class ABlock(nn.Module):
    """
    Area-attention block module for efficient feature extraction in YOLO models.

    This module implements an area-attention mechanism combined with a feed-forward network for processing feature maps.
    It uses a novel area-based attention approach that is more efficient than traditional self-attention while
    maintaining effectiveness.

    Attributes:
        attn (AAttn): Area-attention module for processing spatial features.
        mlp (nn.Sequential): Multi-layer perceptron for feature transformation.

    Methods:
        _init_weights: Initializes module weights using truncated normal distribution.
        forward: Applies area-attention and feed-forward processing to input tensor.

    Examples:
        >>> block = ABlock(dim=256, num_heads=8, mlp_ratio=1.2, area=1)
        >>> x = torch.randn(1, 256, 32, 32)
        >>> output = block(x)
        >>> print(output.shape)
        torch.Size([1, 256, 32, 32])
    """

    def __init__(self, dim, num_heads, mlp_ratio=1.2, area=1):
        """
        Initializes an Area-attention block module for efficient feature extraction in YOLO models.

        This module implements an area-attention mechanism combined with a feed-forward network for processing feature
        maps. It uses a novel area-based attention approach that is more efficient than traditional self-attention
        while maintaining effectiveness.

        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of heads into which the attention mechanism is divided.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            area (int): Number of areas the feature map is divided.
        """
        super().__init__()

        self.attn = AAttn(dim, num_heads=num_heads, area=area)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(Conv(dim, mlp_hidden_dim, 1), Conv(mlp_hidden_dim, dim, 1, act=False))

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Initialize weights using a truncated normal distribution."""
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Forward pass through ABlock, applying area-attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x)
        return x + self.mlp(x)


class A2C2f(nn.Module):
    """
    Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

    This module extends the C2f architecture by incorporating area-attention and ABlock layers for improved feature
    processing. It supports both area-attention and standard convolution modes.

    Attributes:
        cv1 (Conv): Initial 1x1 convolution layer that reduces input channels to hidden channels.
        cv2 (Conv): Final 1x1 convolution layer that processes concatenated features.
        gamma (nn.Parameter | None): Learnable parameter for residual scaling when using area attention.
        m (nn.ModuleList): List of either ABlock or C3k modules for feature processing.

    Methods:
        forward: Processes input through area-attention or standard convolution pathway.

    Examples:
        >>> m = A2C2f(512, 512, n=1, a2=True, area=1)
        >>> x = torch.randn(1, 512, 32, 32)
        >>> output = m(x)
        >>> print(output.shape)
        torch.Size([1, 512, 32, 32])
    """

    def __init__(self, c1, c2, n=1, a2=True, area=1, residual=False, mlp_ratio=2.0, e=0.5, g=1, shortcut=True):
        """
        Area-Attention C2f module for enhanced feature extraction with area-based attention mechanisms.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            n (int): Number of ABlock or C3k modules to stack.
            a2 (bool): Whether to use area attention blocks. If False, uses C3k blocks instead.
            area (int): Number of areas the feature map is divided.
            residual (bool): Whether to use residual connections with learnable gamma parameter.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            e (float): Channel expansion ratio for hidden channels.
            g (int): Number of groups for grouped convolutions.
            shortcut (bool): Whether to use shortcut connections in C3k blocks.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        assert c_ % 32 == 0, "Dimension of ABlock be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.gamma = nn.Parameter(0.01 * torch.ones(c2), requires_grad=True) if a2 and residual else None
        self.m = nn.ModuleList(
            nn.Sequential(*(ABlock(c_, c_ // 32, mlp_ratio, area) for _ in range(2)))
            if a2
            else C3k(c_, c_, 2, shortcut, g)
            for _ in range(n)
        )

    def forward(self, x):
        """Forward pass through R-ELAN layer."""
        y = [self.cv1(x)]
        y.extend(m(y[-1]) for m in self.m)
        y = self.cv2(torch.cat(y, 1))
        if self.gamma is not None:
            return x + self.gamma.view(-1, len(self.gamma), 1, 1) * y
        return y

#TO-DO, change padding to a YAML property
class SequenceConv2dBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        in_channels, out_channels, kernel_size, stride, config = args
        seqlens = config.get("seqlens")
        if seqlens is None:
            raise ValueError("seqlens must be provided for SequenceConv2dBlock")
        
        # Set padding to maintain proper downsampling when stride > 1
        padding = kernel_size // 2 if stride > 1 else 0
        self.module = SequenceConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,  # Explicitly set padding
            seqlens=seqlens
        )

    def forward(self, x):
        #print(f"Layer {self.i}: SequenceConv2dBlock input shape: {x.shape}")
        x = self.module(x)
        #print(f"Layer {self.i}: SequenceConv2dBlock output shape: {x.shape}")
        return x


class SequenceToImage(nn.Module):
    """
    A module that reshapes a flattened sequence tensor into either a 2D image or a 3D video sequence
    based on the number of dimensions provided in the sequence lengths.

    Args:
        *args: Variable length argument list. If a single argument is provided and it is a list or tuple,
               it is unpacked to set the sequence lengths (seqlens). Otherwise, args should directly
               contain the sequence lengths (e.g., H, W or T, H, W).
        **kwargs: Additional keyword arguments (not used).

    Attributes:
        seqlens (tuple): The sequence lengths defining the target shape.
                         - If len(seqlens) == 2: reshapes to (B, D, H, W) for an image.
                         - If len(seqlens) == 3: reshapes to (B, T, H, W, D) for a video sequence.

    Raises:
        ValueError: If seqlens is not a list or tuple of length 2 or 3, if its elements are not integers,
                    or if the input sequence length S does not match the product of seqlens.
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        # Handle case where seqlens is passed as a single list/tuple (e.g., [H, W] or [T, H, W])
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self.seqlens = args
        # Validate seqlens
        if not isinstance(self.seqlens, (list, tuple)) or len(self.seqlens) not in [2, 3]:
            raise ValueError("seqlens must be a list or tuple of length 2 or 3")
        if not all(isinstance(dim, int) for dim in self.seqlens):
            raise ValueError("All elements in seqlens must be integers")

    def forward(self, x):
        """
        Reshape the input tensor based on the number of dimensions in seqlens.

        Args:
            x (Tensor): Input tensor of shape (B, S, D), where B is batch size, S is sequence length,
                        and D is the feature dimension.

        Returns:
            Tensor: Reshaped tensor.
                    - If len(seqlens) == 2: (B, D, H, W) for an image.
                    - If len(seqlens) == 3: (B, T, H, W, D) for a video sequence.

        Raises:
            ValueError: If the sequence length S does not match the product of seqlens dimensions.
        """
        B, S, D = x.shape
        if len(self.seqlens) == 2:
            # 2D case: reshape to image (B, D, H, W)
            H, W = self.seqlens
            if S != H * W:
                raise ValueError(f"Sequence length {S} does not match seqlens {H}*{W}={H*W}")
            x = x.view(B, H, W, D).permute(0, 3, 1, 2)  # (B, D, H, W)
        elif len(self.seqlens) == 3:
            # 3D case: reshape to video sequence (B, T, H, W, D)
            T, H, W = self.seqlens
            if S != T * H * W:
                raise ValueError(f"Sequence length {S} does not match seqlens {T}*{H}*{W}={T*H*W}")
            x = x.view(B, T, H, W, D)  # (B, T, H, W, D)
        return x
    
class VitPatchEmbedBlock(nn.Module):
    """
    A block that wraps VitPatchEmbed for Ultralytics YAML integration, converting input tensors into patch embeddings.

    Args:
        *args: Variable length argument list, typically [c1, c2, resolution, patch_size].
            - c1 (int): Input channels (total, e.g., num_channels * num_frames for 3D).
            - c2 (int): Embedding dimension (dim).
            - resolution (list/tuple): Input resolution, e.g., [H, W] for 2D or [T, H, W] for 3D.
            - patch_size (int or list/tuple): Patch size, e.g., P for 2D ([P, P]) or [T_patch, H_patch, W_patch] for 3D.
        **kwargs: Additional arguments.
            - init_weights (str): Weight initialization method ('xavier_uniform' or 'torch', default: 'xavier_uniform').

    Input:
        x (torch.Tensor): Input tensor.
            - For 2D: Shape (B, C, H, W).
            - For 3D: Shape (B, C*T, H, W), where C is channels per frame, T is number of frames.

    Output:
        torch.Tensor: Patch embeddings.
            - For 2D: Shape (B, H', W', dim).
            - For 3D: Shape (B, T', H', W', dim), where T' = T // T_patch, etc.

    YAML Example:
        - [-1, 1, VitPatchEmbedBlock, [12, 256, [4, 256, 256], [1, 16, 16]]]
          # c1=12 (3 channels * 4 frames), c2=256 (dim), resolution=[T=4, H=256, W=256], patch_size=[1, 16, 16]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Handle Ultralytics YAML parsing where args is a single list
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
         
        c1, c2, resolution, patch_size = args
        self.ndim = len(resolution)
        
        # For 3D sequences, c1 is total channels (C * T), so compute channels per frame
        num_frames = resolution[0] if self.ndim == 3 else 1
        num_channels = c1 // num_frames
        
        # Initialize the VitPatchEmbed module
        self.module = VitPatchEmbed(
            dim=c2,
            num_channels=num_channels,
            resolution=resolution,
            patch_size=patch_size,
            init_weights=kwargs.get('init_weights', 'xavier_uniform')
        )
        self.seqlens = self.module.seqlens  # e.g., [T', H', W'] for 3D or [H', W'] for 2D

    def forward(self, x):
        return self.module(x)  # Outputs (B, T', H', W', dim) for 3D or (B, H', W', dim) for 2D



class VitPosEmbedBlock(nn.Module):
    """
    A block that wraps VitPosEmbed for Ultralytics YAML integration, adding positional embeddings to patch embeddings.

    Args:
        *args: Variable length argument list, typically [c1, c2, seqlens].
            - c1 (int): Input dimension (must equal c2, included for YAML compatibility).
            - c2 (int): Embedding dimension (dim).
            - seqlens (list/tuple): Sequence lengths, e.g., [H', W'] for 2D or [T', H', W'] for 3D.
        **kwargs: Additional arguments.
            - is_learnable (bool): Use learnable embeddings if True (default: False).
            - allow_interpolation (bool): Allow resizing embeddings if True (default: True).
            - interpolate_offset (float): Offset for interpolation (default: None).

    Input:
        x (torch.Tensor): Patch embeddings.
            - For 2D: Shape (B, H', W', dim).
            - For 3D: Shape (B, T', H', W', dim).

    Output:
        torch.Tensor: Patch embeddings with positional embeddings added, same shape as input.

    YAML Example:
        - [-1, 1, VitPosEmbedBlock, [256, 256, [4, 16, 16]]]
          # c1=256 (input dim), c2=256 (dim), seqlens=[T'=4, H'=16, W'=16]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Handle Ultralytics YAML parsing where args is a single list
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        
        c1, c2, seqlens = args
        assert c1 == c2, "Input and output dimensions must be equal for positional embedding"
        
        # Initialize the VitPosEmbed module
        self.module = VitPosEmbed(
            seqlens=seqlens,
            dim=c2,
            is_learnable=kwargs.get('is_learnable', False),
            allow_interpolation=kwargs.get('allow_interpolation', True),
            interpolate_offset=kwargs.get('interpolate_offset', None)
        )
        self.seqlens = seqlens

    def forward(self, x):
        return self.module(x)  # Adds positional embeddings, preserving input shape
  

import torch
import torch.nn as nn

class ViLBlockPairBlock(nn.Module):
    """
    A block for VisionLSTM that processes sequence features without managing hidden states.
    Designed for object detection, handling features and returning output tensors.

    Args:
        c1 (int): Input channel dimension (typically from previous layer).
        c2 (int): Output channel dimension (e.g., 256).
        config (dict): Configuration dictionary with keys like 'seqlens', 'chunk_size', etc.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Handle args from YAML parsing (if necessary)
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        c1, c2, config = args if len(args) > 2 else (args[0], args[1], {})

        self.c1 = c1  # Input channels
        self.c2 = c2  # Output channels (dim)
        self.config = config
        # print(config)

        # Extract sequence lengths (e.g., [H, W] for 2D or [T, H, W] for 3D)
        seqlens = config.get("seqlens")
        if not seqlens:
            raise ValueError("seqlens is required in config for ViLBlockPairBlock")
        if not isinstance(seqlens, (list, tuple)):
            raise ValueError("seqlens must be a list or tuple")

        # Determine mode and compute the expected sequence length
        if len(seqlens) == 2:
            self.mode = "2d"
            self.expected_seq = seqlens[0] * seqlens[1]  # H * W
        elif len(seqlens) == 3:
            self.mode = "3d"
            self.expected_seq = seqlens[0] * seqlens[1] * seqlens[2]  # T * H * W
            # If conv_kind hasn't been explicitly set to "3d", override it
            if config.get("conv_kind", "2d") != "3d":
                config["conv_kind"] = "3d"
        else:
            raise ValueError("seqlens must be a list/tuple of length 2 (2D) or 3 (3D)")

        # print(seqlens)
        # print(config.get("chunk_size", 256))
        # Initialize the underlying ViLBlockPair without hidden state management
        self.module = ViLBlockPair(
            dim=c2,
            drop_path=config.get("drop_path", 0.0),
            conv_kind=config.get("conv_kind", "2d"),
            conv_kernel_size=config.get("conv_kernel_size", 3),
            proj_bias=config.get("proj_bias", True),
            norm_bias=config.get("norm_bias", True),
            seqlens=seqlens,
            qkv_block_size = config.get("qkv_block_size", 4),
            num_blocks=config.get("num_blocks", None),
            init_weights=config.get("init_weights", "original"),
            chunk_size=config.get("chunk_size", 256),
        )
        self.seqlens = seqlens  # Store for reference

    def forward(self, x):
        """
        Forward pass of the block, processing input and returning only the output tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, S, D), where S = expected sequence length.
                               For 2D, S = H * W; for 3D, S = T * H * W.

        Returns:
            torch.Tensor: Output tensor of shape (B, S, D).
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected x to be a torch.Tensor, got {type(x)}")
        # print("SHAPE OF VIL INPUT" + str(x.shape))

        #flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")
        B, S, D = x.shape

        # Check that the sequence length matches the expected product of seqlens dims
        assert S == self.expected_seq, (
            f"Input sequence length {S} does not match expected {self.expected_seq} "
            f"computed from seqlens {self.seqlens}"
        )
        # Check feature dimension matches
        assert D == self.c2, f"Input dimension {D} does not match expected {self.c2}"


        # Stateless forward pass through the underlying module
        out = self.module(x)
        return out
        

class SequenceToImage(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        self.seqlens = args

    def forward(self, x):
        B, S, D = x.shape
        if len(self.seqlens) == 2:
            h, w = self.seqlens
            if S != h * w:
                raise ValueError(f"Sequence length {S} does not match seqlens {h}*{w}={h*w}")
            return x.view(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, H, W) for 2D
        elif len(self.seqlens) == 3:
            t, h, w = self.seqlens
            if S != t * h * w:
                raise ValueError(f"Sequence length {S} does not match seqlens {t}*{h}*{w}={t*h*w}")
            return x.view(B, t, h, w, D)  # (B, T, H, W, D) for 3D
        else:
            raise ValueError("Unsupported seqlens dimension")
    
class PatchMergeBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        # Handle args as a single iterable (e.g., from YAML) or separate arguments
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        input_seqlens, merge_factor, in_dim, out_dim = args
        self.module = PatchMerge(
            input_seqlens=input_seqlens,
            merge_factor=merge_factor,
            in_dim=in_dim,
            out_dim=out_dim
        )

    def forward(self, x):
        return self.module(x)
    
class MultiScaleFusionBlock(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        high_res_dim, low_res_dim, fused_dim = args
        self.module = MultiScaleFusion(high_res_dim=high_res_dim, low_res_dim=low_res_dim, fused_dim=fused_dim)

    def forward(self, high_res, low_res):
        return self.module(high_res, low_res)

class VisionLSTM(nn.Module):
    """
    VisionLSTM block using local components.
    Usage: [-1, 1, VisionLSTM, [c1, dim, {"depth": 12, "resolution": [h, w], "patch_size": 16, "output_indices": [3, 7, 11]}]]
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]
        c1, dim, config = args if len(args) > 2 else (args[0], args[1], {})
        
        self.depth = config.get("depth", 12)
        self.output_indices = config.get("output_indices", [])
        self.pooling = config.get("pooling", "to_image")
        resolution = config.get("resolution", [224, 224])
        patch_size = config.get("patch_size", 16)
        
        max_idx = max(self.output_indices) if self.output_indices else 0
        if self.depth <= max_idx:
            raise ValueError(f"Depth {self.depth} too small for output_indices {self.output_indices}")

        self.patch_embed = VitPatchEmbedBlock(c1, dim, resolution, patch_size)
        self.pos_embed = VitPosEmbed2dBlock(dim, dim, self.patch_embed.seqlens)
        self.blocks = nn.ModuleList([
            ViLBlockPairBlock(dim, dim, {"seqlens": self.patch_embed.seqlens, "drop_path": config.get("drop_path", 0.0)})
            for _ in range(self.depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.to_image = SequenceToImage(self.patch_embed.seqlens) if self.pooling == "to_image" else nn.Identity()




    def forward(self, x):
        x = self.patch_embed(x)
        x = self.pos_embed(x)

                # flatten to 1d
        x = einops.rearrange(x, "b ... d -> b (...) d")
        
        partial_outputs = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.output_indices:
                out = self.norm(x)
                out = self.to_image(out) if self.pooling == "to_image" else x
                partial_outputs.append(out)
        
        x = self.norm(x)
        x = self.to_image(x)
        partial_outputs.append(x)
        #print(partial_outputs)
        return partial_outputs
    


class VisionClueMerge(nn.Module):
    def __init__(self, dim, out_dim, config=None, *args, **kwargs):
        super().__init__()
        if config is None:
            config = {}
        self.dim = dim  # Input dimension D
        self.out_dim = out_dim  # Output dimension
        self.config = config
        self.seqlens = config.get("seqlens")
        self.hidden = int(dim * 4)  # 4*D channels after concatenation
        self.pw_linear = nn.Sequential(
            nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_dim),
            nn.SiLU()
        )

    def forward(self, x):
        # Input shape: [B, H*W, D]
        B, N, D = x.shape
        H, W = self.seqlens
        assert N == H * W, f"Expected N = H*W, got N={N}, H*W={H*W}"
        assert D == self.dim, f"Expected D={self.dim}, got D={D}"
        
        # Reshape to [B, H, W, D] and permute to [B, D, H, W]
        x = x.view(B, H, W, D).permute(0, 3, 1, 2)
        
        # Slice and concatenate along channel dimension
        y = torch.cat([
            x[:, :, ::2, ::2],    # Even rows, even cols
            x[:, :, 1::2, ::2],   # Odd rows, even cols
            x[:, :, ::2, 1::2],   # Even rows, odd cols
            x[:, :, 1::2, 1::2]   # Odd rows, odd cols
        ], dim=1)  # Shape: [B, 4*D, H//2, W//2]
        
        # Apply pointwise convolution
        y = self.pw_linear(y)  # Shape: [B, out_dim, H//2, W//2]
        
        # Reshape back to sequence
        y = y.permute(0, 2, 3, 1).reshape(B, (H//2)*(W//2), self.out_dim)
        
        return y  # Output shape: [B, (H//2)*(W//2), out_dim]
    
        # ------------------------------------------------------------------
    #  Exclude BN γ/β from weight‑decay (Ultralytics trainer will call this)
    # ------------------------------------------------------------------
    @torch.jit.ignore
    def no_weight_decay(self) -> set[str]:
        return {
            "pw_linear.1.weight",   # BatchNorm2d weight
            "pw_linear.1.bias"      # BatchNorm2d bias
        }


# class VisionClueMerge(nn.Module):
#     def __init__(self, dim, out_dim):
#         super().__init__()
#         self.hidden = int(dim * 4)

#         self.pw_linear = nn.Sequential(
#             nn.Conv2d(self.hidden, out_dim, kernel_size=1, stride=1, padding=0),
#             nn.BatchNorm2d(out_dim),
#             nn.SiLU()
#         )

#     def forward(self, x):
#         y = torch.cat([
#             x[..., ::2, ::2],
#             x[..., 1::2, ::2],
#             x[..., ::2, 1::2],
#             x[..., 1::2, 1::2]
#         ], dim=1)
#         return self.pw_linear(y)
    


class PatchMerger(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_dim, in_dim))

    def forward(self, x):
        # x: (B, N, D)
        scores = torch.einsum('md,bnd->bmn', self.W, x)  # (B, M, N)
        attention = F.softmax(scores, dim=2)  # (B, M, N)
        y = torch.einsum('bmn,bnd->bmd', attention, x)  # (B, M, D)
        return y
    
class XViLBlockPairBlock(nn.Module):
    """
    An adapted block for VisionLSTM that processes concatenated feature maps for object detection.
    Scales down input channels before processing with ViLBlockPair, returning output in 4D format.

    Args:
        c1 (int): Input channel dimension (e.g., 1536 from concatenation).
        c2 (int): Output channel dimension (e.g., 512).
        config (dict): Configuration dictionary with keys like 'seqlens', 'chunk_size', etc.
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        # print("XVIL ARGS")
        # print(args)
        # Handle args from YAML parsing
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            args = args[0]

        
        c1, c2, config = args if len(args) > 2 else (args[0], args[1], {})

        self.c1 = c1  # Input channels (e.g., concatenated channels)
        self.c2 = c2  # Output channels (dim)
        self.config = config

        # print("INNER C1 " + str(c1))
        # print("INNER C2 " + str(c2))

        # Extract sequence lengths (e.g., [H, W] for 2D or [T, H, W] for 3D)
        seqlens = config.get("seqlens")
        if not seqlens:
            raise ValueError("seqlens is required in config for XViLBlockPairBlock")
        if not isinstance(seqlens, (list, tuple)):
            raise ValueError("seqlens must be a list or tuple")

        # Determine mode and compute expected sequence length
        if len(seqlens) == 2:
            self.mode = "2d"
            self.expected_seq = seqlens[0] * seqlens[1]  # H * W
        elif len(seqlens) == 3:
            self.mode = "3d"
            self.expected_seq = seqlens[0] * seqlens[1] * seqlens[2]  # T * H * W
            if config.get("conv_kind", "2d") != "3d":
                config["conv_kind"] = "3d"
        else:
            raise ValueError("seqlens must be a list/tuple of length 2 (2D) or 3 (3D)")

        # Input projection to scale down channels (like XSSBlock's in_proj)
        if c1 != c2:
            self.in_proj = nn.Sequential(
                nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(c2),
                nn.SiLU()
            )
        else:
            self.in_proj = nn.Identity()

        # Underlying ViLBlockPair
        self.module = ViLBlockPair(
            dim=c2,
            drop_path=config.get("drop_path", 0.0),
            conv_kind=config.get("conv_kind", "2d"),
            conv_kernel_size=config.get("conv_kernel_size", 3),
            proj_bias=config.get("proj_bias", True),
            norm_bias=config.get("norm_bias", True),
            seqlens=seqlens,
            num_blocks=config.get("num_blocks", None),
            init_weights=config.get("init_weights", "original"),
            chunk_size=config.get("chunk_size", 256),
        )
        self.seqlens = seqlens  # Store for reference

    def forward(self, x):
        """
        Forward pass of the block, processing 4D input and returning 4D output tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, c2, H, W).
        """
        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected x to be a torch.Tensor, got {type(x)}")

        # Input shape: (B, C_in, H, W)
        B, C_in, H, W = x.shape
        assert C_in == self.c1, f"Input channels {C_in} must match c1 {self.c1}"

        # Scale down channels
        x = self.in_proj(x)  # (B, c2, H, W)

        # Flatten to sequence
        S = H * W if self.mode == "2d" else H * W * self.seqlens[0]
        x_seq = x.view(B, self.c2, -1).permute(0, 2, 1)  # (B, S, c2)
        assert x_seq.shape[1] == self.expected_seq, (
            f"Sequence length {x_seq.shape[1]} does not match expected {self.expected_seq}"
        )

        # Process with ViLBlockPair
        out_seq = self.module(x_seq)  # (B, S, c2)

        # Reshape back to 4D
        out = out_seq.permute(0, 2, 1).view(B, self.c2, H, W)  # (B, c2, H, W)
        return out
    


class RGBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 channels_first=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, kernel_size=1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True,
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.act(self.dwconv(x) + x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LSBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_features, hidden_features, kernel_size=3, padding=3 // 2, groups=hidden_features)
        self.norm = nn.BatchNorm2d(hidden_features)
        self.fc2 = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)
        self.act = act_layer()
        self.fc3 = nn.Conv2d(hidden_features, in_features, kernel_size=1, padding=0)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        input = x
        x = self.fc1(x)
        x = self.norm(x)
        x = self.fc2(x)
        x = self.act(x)
        x = self.fc3(x)
        x = input + self.drop(x)
        return x
    


class ViLLayerNormBlock(nn.Module):
    """
    Thin wrapper so Ultralytics YAML can instantiate the Vision‑LSTM LayerNorm.

    Args:
        dim (int)       – embedding dimension (normalized_shape)
        eps (float)     – epsilon (default 1e‑5)
        weight (bool)   – create γ (default True)
        bias (bool)     – create β (default False)
    """
    def __init__(self, dim, eps=1e-5, weight=True, bias=False):
        super().__init__()
        self.ln = LayerNorm(ndim=dim, eps=eps, weight=weight, bias=bias)

    def forward(self, x):
        return self.ln(x)



class ViLFusionBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        config: dict,
        n: int = 1,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        mlp_act_layer=nn.GELU,
        mlp_drop_rate: float = 0.0,
    ):
        """
        Initialize the FusionViLBlock, mirroring XSSBlock logic with ViLBlockPairBlock.

        Args:
            in_channels (int): Number of input channels.
            hidden_dim (int): Hidden dimension after projection.
            config (dict): Configuration dict with 'seqlens', etc.
            n (int): Number of ViLBlockPairBlock repetitions.
            mlp_ratio (float): Expansion ratio for MLP hidden dimension.
            drop_path (float): Drop path probability.
            norm_layer (callable): Normalization layer constructor.
            mlp_act_layer (type): Activation layer for MLP.
            mlp_drop_rate (float): Dropout rate for MLP.
        """
        super().__init__()

        # print(config)
        # Extract seqlens from config
        seqlens = config.get("seqlens")
        if not seqlens or not isinstance(seqlens, (list, tuple)) or len(seqlens) not in [2, 3]:
            raise ValueError("config['seqlens'] must be a list/tuple of length 2 or 3")

        self.seqlens = seqlens
        self.hidden_dim = hidden_dim

        # Input projection (same as XSSBlock)
        self.in_proj = (
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU()
            ) if in_channels != hidden_dim else nn.Identity()
        )

        # Local spatial processing (LSBlock from XSSBlock)
        self.lsblock = LSBlock(hidden_dim, hidden_dim)

        # Normalization for sequence input (B, S, D)
        self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)

        # ViLBlockPairBlock replacing SS2D
        self.vil = nn.Sequential(*[
            ViLBlockPairBlock(hidden_dim, hidden_dim, config)
            for _ in range(n)
        ])

        # DropPath from vision_lstm_util.py
        self.drop_path = DropPath(drop_prob=drop_path) if drop_path > 0 else nn.Identity()

        # Optional MLP branch (same as XSSBlock)
        self.mlp_branch = mlp_ratio > 0
        if self.mlp_branch:
            self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)  # Sequence norm
            mlp_hidden_dim = int(hidden_dim * mlp_ratio)
            self.mlp = RGBlock(
                in_features=hidden_dim,
                hidden_features=mlp_hidden_dim,
                act_layer=mlp_act_layer,
                drop=mlp_drop_rate
            )

    def forward(self, x):
        """
        Forward pass mirroring XSSBlock logic.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C_in, H, W).

        Returns:
            torch.Tensor: Output tensor of shape (B, hidden_dim, H, W).
        """
        # Input projection
        x = self.in_proj(x)  # (B, hidden_dim, H, W)

        # Local spatial processing
        x_local = self.lsblock(x)  # (B, hidden_dim, H, W)

        # Flatten to sequence for ViLBlockPairBlock
        B, C, H, W = x_local.shape
        seq = einops.rearrange(x_local, "b c h w -> b (h w) c")  # (B, S, hidden_dim)

        # Normalize and process with ViLBlockPairBlock
        seq_norm = self.norm(seq)  # (B, S, hidden_dim)
        seq_out = self.vil(seq_norm)  # (B, S, hidden_dim)

        # Apply drop path and residual connection in sequence space
        seq = seq + self.drop_path(seq_out)  # (B, S, hidden_dim)

        # Reshape back to 4D
        x_global = einops.rearrange(seq, "b (h w) c -> b c h w", h=H, w=W)  # (B, hidden_dim, H, W)

        # Residual connection with original input
        x = x + x_global  # (B, hidden_dim, H, W)

        # Optional MLP branch
        if self.mlp_branch:
            # Flatten again for MLP (RGBlock expects 4D input)
            seq = einops.rearrange(x, "b c h w -> b (h w) c")  # (B, S, hidden_dim)
            # seq_norm = self.norm2(seq)  # (B, S, hidden_dim)
            # Reshape to 4D for RGBlock
            x_mlp = einops.rearrange(seq, "b (h w) c -> b c h w", h=H, w=W)  # (B, hidden_dim, H, W)
            x_mlp = self.mlp(x_mlp)  # (B, hidden_dim, H, W)
            # Add back with drop path
            x = x + self.drop_path(x_mlp)  # (B, hidden_dim, H, W)

        return x


# class ViLFusionBlock(nn.Module):
#     class _LocalSpatial(nn.Module):
#         def __init__(self, dim, act_layer=nn.GELU, drop=0.):
#             super().__init__()
#             self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
#             self.bn = nn.BatchNorm2d(dim)
#             self.pw1 = nn.Conv2d(dim, dim, 1)
#             self.act = act_layer()
#             self.pw2 = nn.Conv2d(dim, dim, 1)
#             self.drop = nn.Dropout(drop)

#         def forward(self, x):
#             res = x
#             x = self.pw2(self.act(self.pw1(self.bn(self.dw3(x)))))
#             return res + self.drop(x)

#     def __init__(self, in_channels, hidden_dim, config):
#         super().__init__()
#         seqlens = config["seqlens"]
#         n_vil = config.get("n_vil", 1)
#         drop_path = config.get("drop_path", 0.0)
#         mlp_ratio = config.get("mlp_ratio", 4.0)

#         self.in_proj = (
#             nn.Sequential(
#                 nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.SiLU(),
#             ) if in_channels != hidden_dim else nn.Identity()
#         )

#         self.local = ViLFusionBlock._LocalSpatial(hidden_dim)

#         self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
#         self.vil = nn.Sequential(*[
#             ViLBlockPair(
#                 dim=hidden_dim,
#                 drop_path=drop_path,
#                 seqlens=seqlens,
#                 conv_kind="3d" if len(seqlens) == 3 else "2d",
#             ) for _ in range(n_vil)
#         ])
#         self.dp = DropPath(drop_prob=drop_path)

#         self.use_mlp = mlp_ratio > 0
#         if self.use_mlp:
#             mlp_hidden = int(hidden_dim * mlp_ratio)
#             self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
#             self.mlp = nn.Sequential(
#                 nn.Conv2d(hidden_dim, mlp_hidden * 2, 1),
#                 nn.GELU(),
#                 nn.Conv2d(mlp_hidden, hidden_dim, 1),
#             )
#             self.dp2 = DropPath(drop_prob=drop_path)

#     @staticmethod
#     def _hw_flat(x):
#         B, D, *spatial = x.shape
#         return x.flatten(2).transpose(1, 2), spatial

#     @staticmethod
#     def _hw_unflat(seq, spatial):
#         B, S, D = seq.shape
#         return seq.transpose(1, 2).view(B, D, *spatial)

#     def forward(self, x):
#         x = self.in_proj(x)  # Shape: [B, hidden_dim, H, W]
#         x_local = self.local(x)  # Shape: [B, hidden_dim, H, W]
        
#         seq, spatial = self._hw_flat(x_local)  # seq: [B, S, hidden_dim], spatial: (H, W)
#         seq_id = seq
#         seq_res = self.vil(self.norm(seq_id))  # Shape: [B, S, hidden_dim]
#         seq = self.dp(seq_id, lambda _: seq_res)  # Shape: [B, S, hidden_dim]
#         x_global = self._hw_unflat(seq, spatial)  # Shape: [B, hidden_dim, H, W]
        
#         x = x + x_global  # Shape: [B, hidden_dim, H, W]
        
#         if self.use_mlp:
#             x_id = x  # Shape: [B, hidden_dim, H, W]
#             # Step 1: Permute to move channel dimension to the end
#             x_permuted = x_id.permute(0, 2, 3, 1)  # Shape: [B, H, W, hidden_dim]
#             # Step 2: Apply LayerNorm over the last dimension (hidden_dim = 128)
#             x_norm = self.norm2(x_permuted)  # Shape: [B, H, W, hidden_dim]
#             # Step 3: Permute back to original order
#             x_norm = x_norm.permute(0, 3, 1, 2)  # Shape: [B, hidden_dim, H, W]
            
#             x_res = self.mlp(x_norm)  # Shape: [B, hidden_dim, H, W]
#             x = self.dp2(x_id, lambda _: x_res)  # Shape: [B, hidden_dim, H, W]
        
#         return x

# class ViLFusionBlock(nn.Module):
#     """
#     XSS‑style fusion layer that uses ViL for the long‑range part and
#     **re‑implements LSBlock locally** (no external dependency).

#     Parameters
#     ----------
#     in_channels : int   # C_in from backbone
#     hidden_dim  : int   # embedding dimension D
#     seqlens     : tuple # (H, W) or (T, H, W) – passed to ViLBlockPair
#     n_vil       : int   # stacked ViLBlockPair layers
#     drop_path   : float
#     mlp_ratio   : float # 0 disables the conv‑MLP branch
#     """

#     # ---------- 1.  LOCAL‑SPATIAL BLOCK (rewritten LSBlock) -----------------
#     class _LocalSpatial(nn.Module):
#         """Depth‑wise 3×3 → BN → 1×1 → GELU → 1×1 with residual & dropout
#         (identical math to LSBlock)"""
#         def __init__(self, dim, act_layer=nn.GELU, drop=0.):
#             super().__init__()
#             self.dw3 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)   # depth‑wise
#             self.bn  = nn.BatchNorm2d(dim)
#             self.pw1 = nn.Conv2d(dim, dim, 1)
#             self.act = act_layer()
#             self.pw2 = nn.Conv2d(dim, dim, 1)
#             self.drop = nn.Dropout(drop)

#         def forward(self, x):
#             res = x
#             x = self.dw3(x)
#             x = self.bn(x)
#             x = self.pw1(x)
#             x = self.act(x)
#             x = self.pw2(x)
#             return res + self.drop(x)
#     # ------------------------------------------------------------------------

#     def __init__(self,
#                  in_channels: int,
#                  hidden_dim: int,
#                  seqlens,
#                  n_vil: int = 1,
#                  drop_path: float = 0.,
#                  mlp_ratio: float = 4.0):
#         super().__init__()

#         # 2. in‑projection (identical to XSSBlock)
#         self.in_proj = (
#             nn.Sequential(
#                 nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
#                 nn.BatchNorm2d(hidden_dim),
#                 nn.SiLU()
#             ) if in_channels != hidden_dim else nn.Identity()
#         )

#         # 3. local‑spatial stage (rewritten)
#         self.local = ViLFusionBlock._LocalSpatial(hidden_dim)

#         # 4. ViL sequence module
#         self.norm = nn.LayerNorm(hidden_dim, eps=1e-6)
#         self.vil = nn.Sequential(*[
#             ViLBlockPair(
#                 dim=hidden_dim,
#                 drop_path=drop_path,
#                 seqlens=seqlens,
#                 conv_kind="3d" if len(seqlens) == 3 else "2d",
#             ) for _ in range(n_vil)
#         ])
#         self.dp = DropPath(drop_path)

#         # 5. optional conv‑MLP branch (same RGBlock maths)
#         self.use_mlp = mlp_ratio > 0
#         if self.use_mlp:
#             mlp_hidden = int(hidden_dim * mlp_ratio)
#             self.norm2 = nn.LayerNorm(hidden_dim, eps=1e-6)
#             self.mlp = nn.Sequential(
#                 nn.Conv2d(hidden_dim, mlp_hidden * 2, 1),
#                 nn.GELU(),
#                 nn.Conv2d(mlp_hidden, hidden_dim, 1)  # channel‑mixing conv‑MLP
#             )
#             self.dp2 = DropPath(drop_path)

#     # ---------- helper: flatten ↔︎ unflatten -----------------
#     @staticmethod
#     def _hw_flat(x):
#         B, D, *spatial = x.shape
#         return x.flatten(2).transpose(1, 2), spatial   # (B,S,D), (H,W[,T])

#     @staticmethod
#     def _hw_unflat(seq, spatial):
#         B, S, D = seq.shape
#         return seq.transpose(1, 2).view(B, D, *spatial)
#     # ---------------------------------------------------------

#     def forward(self, x):
#         """
#         x : (B, C_in, H, W)  or  (B, C_in, T, H, W)
#         """
#         x = self.in_proj(x)

#         # local spatial conditioning
#         x_local = self.local(x)

#         # ViL global fusion
#         seq, spatial = self._hw_flat(x_local)
#         seq = seq + self.dp(self.vil(self.norm(seq)))
#         x_global = self._hw_unflat(seq, spatial)

#         x = x + x_global      # first residual

#         # optional MLP
#         if self.use_mlp:
#             x = x + self.dp2(self.mlp(self.norm2(x)))

#         return x

nn.ViLLayerNormBlock = ViLLayerNormBlock
nn.ViLInternalNorm = LayerNorm 
nn.ViLInternalMultiHeadNorm = MultiHeadLayerNorm 