import torch
import torch.nn as nn
import einops

from .vision_lstm_util import VitPatchEmbed, VitPosEmbed2d
# Assume that HierarchicalBlockGroup has been defined as above
# (and PatchMerge and MultiScaleFusion are imported as well)
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    A simple Layer Normalization module that supports a 'bias' parameter.
    
    Args:
        normalized_shape (int or tuple): The shape of the dimensions to normalize.
        eps (float): A small constant for numerical stability.
        bias (bool): If True, includes a learnable bias parameter.
    """
    def __init__(self, normalized_shape, eps=1e-6, bias=True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        # Always create a learnable weight parameter.
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        # Create a learnable bias parameter if requested; otherwise, register None.
        if bias:
            self.bias = nn.Parameter(torch.zeros(normalized_shape))
        else:
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)



import torch
import torch.nn as nn
import einops
import torch.nn as nn
import einops

class PatchMerge(nn.Module):
    def __init__(self, input_seqlens, merge_factor=2, in_dim=192, out_dim=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.merge_factor = merge_factor
        self.H, self.W = input_seqlens  # Input height and width
        self.out_dim = out_dim if out_dim is not None else in_dim * (merge_factor ** 2)
        self.proj = nn.Linear(in_dim * (merge_factor ** 2), self.out_dim)  # Linear projection
        self.norm = norm_layer(self.out_dim) if norm_layer else nn.Identity()  # Normalization layer

    def forward(self, x):
        B, N, C = x.shape  # Batch size, number of patches, channels
        # Reshape into a grid
        x = einops.rearrange(x, "b (h w) c -> b h w c", h=self.H, w=self.W)
        # Reduce resolution by merging patches
        new_H, new_W = self.H // self.merge_factor, self.W // self.merge_factor
        x = x.unfold(1, self.merge_factor, self.merge_factor).unfold(2, self.merge_factor, self.merge_factor)
        x = einops.rearrange(x, "b h w m1 m2 c -> b h w (m1 m2 c)")
        x = einops.rearrange(x, "b h w c -> b (h w) c")
        # Apply linear projection
        x = self.proj(x)
        # Apply normalization
        x = self.norm(x)
        return x



class MultiScaleFusion(nn.Module):
    """
    Fuses high-resolution and low-resolution token features.

    This module takes two inputs:
      - high_res: features from the local branch (e.g. shape (B, N_high, D1))
      - low_res: features from the global branch (e.g. shape (B, N_low, D2))
      
    The high-res features are pooled (here via average pooling) to generate a representative
    feature for each sample, then expanded to match the low-res token count. The features are concatenated
    along the channel dimension and projected to the desired fused dimension.
    """
    def __init__(self, high_res_dim, low_res_dim, fused_dim):
        """
        Args:
            high_res_dim (int): Channel dimension of high-resolution features.
            low_res_dim (int): Channel dimension of low-resolution features.
            fused_dim (int): Desired output dimension after fusion.
        """
        super().__init__()
        self.fusion_proj = nn.Linear(high_res_dim + low_res_dim, fused_dim)

    def forward(self, high_res, low_res):
        """
        Args:
            high_res: Tensor of shape (B, N_high, D1) -- features from local branch.
            low_res: Tensor of shape (B, N_low, D2) -- features from global branch.
            
        Returns:
            fused: Tensor of shape (B, N_low, fused_dim)
        """
        B, N_low, _ = low_res.shape
        # Pool the high-res features: use average pooling over the token dimension
        high_res_pool = high_res.mean(dim=1, keepdim=True)  # shape: (B, 1, D1)
        # Expand the pooled high-res features to match low_res token count
        high_res_expanded = high_res_pool.expand(B, N_low, -1)  # shape: (B, N_low, D1)
        # Concatenate along the channel dimension
        fusion_input = torch.cat([high_res_expanded, low_res], dim=-1)  # (B, N_low, D1 + D2)
        # Project to the fused dimension
        fused = self.fusion_proj(fusion_input)
        return fused



from .vision_lstm2 import ViLBlockPair  # Ensure this is imported from the VisionLSTM library

class HierarchicalBlockGroup(nn.Module):
    """
    HierarchicalBlockGroup encapsulates one level of hierarchical processing.
    
    It consists of:
      - A local branch: processes the input tokens with a set of Vision-LSTM blocks.
      - A downsampling step: uses PatchMerge to create a coarser token grid.
      - A global branch: processes the downsampled tokens with additional Vision-LSTM blocks.
      - A fusion step: fuses the local (high-res) features and the global (merged) features.
    
    The output is a fused token sequence that can be fed into further hierarchical groups.
    """
    def __init__(self, in_dim, local_depth, global_depth, 
                 merge_factor=2, conv_kind="2d", conv_kernel_size=3, 
                 proj_bias=True, norm_bias=True, num_blocks=None,
                 fusion_fused_dim=None, seqlens=None, init_weights="original"):
        """
        Args:
            in_dim (int): Input token dimension.
            local_depth (int): Number of Vision-LSTM block pairs for the local branch.
            global_depth (int): Number of Vision-LSTM block pairs for the global branch.
            merge_factor (int): Factor to downsample the token grid (e.g., 2 for 2×2 merge).
            conv_kind (str): Convolution kind to be passed to the block pairs.
            conv_kernel_size (int): Kernel size for convolution in blocks.
            proj_bias (bool): Whether to use bias in projection layers.
            norm_bias (bool): Whether to use bias in normalization layers.
            num_blocks (int or None): Parameter for the blocks (if needed).
            fusion_fused_dim (int or None): Output dimension for fusion. If None, defaults to in_dim.
            seqlens (tuple or None): Spatial dimensions (H, W) of the token grid.
            init_weights (str): Weight initialization strategy.
        """
        super().__init__()
        self.in_dim = in_dim
        self.merge_factor = merge_factor

        # Local branch: process input tokens with a set of Vision-LSTM blocks.
        self.local_blocks = nn.ModuleList([
            ViLBlockPair(
                dim=in_dim,
                drop_path=0.0,
                conv_kind=conv_kind,
                conv_kernel_size=conv_kernel_size,
                proj_bias=proj_bias,
                norm_bias=norm_bias,
                seqlens=seqlens,
                num_blocks=num_blocks,
                init_weights=init_weights,
            ) for _ in range(local_depth)
        ])
        
        # PatchMerge: downsample tokens from the local branch.
        if seqlens is None:
            raise ValueError("seqlens (spatial dimensions of token grid) must be provided")
        self.patch_merge = PatchMerge(input_seqlens=seqlens, 
                                      merge_factor=merge_factor, 
                                      in_dim=in_dim)
        
        # Compute the global dimension from PatchMerge output
        global_dim = self.patch_merge.out_dim  # e.g., in_dim * (merge_factor ** 2) if out_dim is None
        
        # Update seqlens for the global branch
        new_H = seqlens[0] // merge_factor
        new_W = seqlens[1] // merge_factor
        global_seqlens = (new_H, new_W)
        
        # Global branch: process the merged tokens with the correct dimension.
        self.global_blocks = nn.ModuleList([
            ViLBlockPair(
                dim=global_dim,  # Use the computed global_dim
                drop_path=0.0,
                conv_kind=conv_kind,
                conv_kernel_size=conv_kernel_size,
                proj_bias=proj_bias,
                norm_bias=norm_bias,
                seqlens=global_seqlens,  # Updated spatial dimensions
                num_blocks=num_blocks,
                init_weights=init_weights,
            ) for _ in range(global_depth)
        ])
        
        # Fusion module: fuse local (high-res) and global (merged) features.
        fusion_out_dim = fusion_fused_dim if fusion_fused_dim is not None else in_dim
        self.fusion = MultiScaleFusion(high_res_dim=in_dim, low_res_dim=global_dim, fused_dim=fusion_out_dim)

    def forward(self, x):
        """
        Args:
            x: Token sequence of shape (B, N, in_dim)
        Returns:
            Fused token sequence of shape (B, N_global, fusion_out_dim)
        """
        # Local branch: process with local Vision-LSTM blocks.
        local_features = x
        for block in self.local_blocks:
            local_features = block(local_features)
        
        # Downsample tokens to create global branch input.
        global_tokens = self.patch_merge(local_features)  # (B, N_global, global_dim)
        
        # Global branch: process the downsampled tokens.
        for block in self.global_blocks:
            global_tokens = block(global_tokens)
        
        # Fuse local and global branches.
        fused_tokens = self.fusion(high_res=local_features, low_res=global_tokens)
        return fused_tokens
    
    
class HierarchicalVisionLSTM(nn.Module):
    def __init__(self,
                 input_shape=(3, 224, 224),
                 patch_size=16,
                 base_dim=192,
                 num_groups=3,
                 local_depth=2,
                 global_depth=2,
                 merge_factor=2,
                 output_shape=(1000,),
                 mode="classifier",
                 pooling="bilateral_flatten",
                 conv_kind="2d",
                 conv_kernel_size=3,
                 proj_bias=True,
                 norm_bias=True,
                 init_weights="original"):
        super().__init__()
        
        # Patch embedding
        self.patch_embed = VitPatchEmbed(dim=base_dim,
                                         num_channels=input_shape[0],
                                         resolution=input_shape[1:],
                                         patch_size=patch_size)
        self.pos_embed = VitPosEmbed2d(seqlens=self.patch_embed.seqlens, dim=base_dim)
        
        # Initialize seqlens for the first group
        current_seqlens = self.patch_embed.seqlens
        
        # Create hierarchical block groups with updated seqlens
        self.hierarchical_groups = nn.ModuleList()
        for _ in range(num_groups):
            group = HierarchicalBlockGroup(
                in_dim=base_dim,
                local_depth=local_depth,
                global_depth=global_depth,
                merge_factor=merge_factor,
                conv_kind=conv_kind,
                conv_kernel_size=conv_kernel_size,
                proj_bias=proj_bias,
                norm_bias=norm_bias,
                num_blocks=local_depth + global_depth,
                fusion_fused_dim=base_dim,
                seqlens=current_seqlens,
                init_weights=init_weights
            )
            self.hierarchical_groups.append(group)
            # Update seqlens for the next group
            current_seqlens = (current_seqlens[0] // merge_factor, current_seqlens[1] // merge_factor)
        
        # Final normalization
        self.norm = LayerNorm(base_dim, bias=norm_bias, eps=1e-6)
        
        # Determine the input features for the classification head based on pooling strategy
        if mode == "classifier":
            if pooling == "bilateral_flatten":
                in_features = 2 * base_dim  # Two tokens are concatenated, doubling the dimension
            else:
                in_features = base_dim  # Default case for other pooling strategies
            self.head = nn.Linear(in_features, output_shape[0])
            nn.init.trunc_normal_(self.head.weight, std=2e-5)
            nn.init.zeros_(self.head.bias)
        else:
            self.head = None
        
        # Store mode and pooling as instance variables
        self.mode = mode
        self.pooling = pooling

    def forward(self, x):
        # x: Input image tensor of shape (B, C, H, W)
        # 1. Patch Embedding & Positional Encoding
        x = self.patch_embed(x)  # shape: (B, H', W', base_dim)
        x = self.pos_embed(x)    # same shape as above
        # Flatten to token sequence: (B, N, base_dim)
        x = einops.rearrange(x, "b ... d -> b (...) d")
        
        # 2. Process iteratively through hierarchical block groups
        for group in self.hierarchical_groups:
            x = group(x)
        
        # 3. Final normalization and pooling
        x = self.norm(x)
        if self.pooling == "to_image":
            # Reshape tokens back to image feature map if desired
            seqlen_h, seqlen_w = self.patch_embed.seqlens
            x = einops.rearrange(x, "b (h w) d -> b d h w", h=seqlen_h, w=seqlen_w)
        elif self.pooling == "bilateral_flatten":
            # Example pooling: concatenate first and last token
            x = torch.cat([x[:, 0:1], x[:, -1:]], dim=1)
            x = einops.rearrange(x, "b n d -> b (n d)")
        # Otherwise, keep x as is (i.e., token sequence)
        
        if self.head is not None:
            x = self.head(x)
        return x


