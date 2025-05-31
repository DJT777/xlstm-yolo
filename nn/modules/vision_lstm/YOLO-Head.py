import torch
import torch.nn as nn
from .vision_lstm_util import small_init_, wang_init_

class YOLOMLPHead(nn.Module):
    """A YOLO-style MLP detection head for VisionLSTM2.

    This head transforms the transformer output into detection predictions,
    producing bounding box coordinates, confidence scores, and class probabilities
    for each patch in the feature map.

    Args:
        dim (int): Input dimension (embedding size from VisionLSTM2).
        num_classes (int): Number of object classes to detect.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension in the MLP. Defaults to 512.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.hidden_dim = hidden_dim
        self.init_weights = init_weights
        self.num_blocks = num_blocks

        # Output size per patch: num_boxes * (4 coords + 1 confidence + num_classes)
        self.output_size = num_boxes * (5 + num_classes)

        # MLP architecture
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.output_size),
        )

        self.reset_parameters()

    def forward(self, x):
        """Forward pass to generate detection predictions.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, dim].

        Returns:
            torch.Tensor: Detection predictions of shape [batch_size, num_patches, num_boxes, 5 + num_classes].
        """
        batch_size, num_patches, _ = x.size()
        predictions = self.mlp(x)  # [batch_size, num_patches, output_size]
        predictions = predictions.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)
        return predictions

    def reset_parameters(self):
        """Initialize parameters following the library's conventions."""
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                if self.init_weights == "original":
                    small_init_(layer.weight, dim=self.dim)
                elif self.init_weights == "original-fixed":
                    if self.num_blocks is None:
                        raise ValueError("num_blocks must be specified for 'original-fixed' initialization")
                    wang_init_(layer.weight, dim=self.dim, num_blocks=self.num_blocks)
                else:
                    raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def extra_repr(self):
        """String representation of the module."""
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, num_boxes={self.num_boxes}, "
            f"hidden_dim={self.hidden_dim}, init_weights={self.init_weights}, num_blocks={self.num_blocks}"
        )
        
        
# Residual Block for ResidualMLPHead
class ResidualBlock(nn.Module):
    """A residual block with a linear layer, GELU activation, dropout, and layer normalization."""
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.gelu(out)
        out = self.dropout(out)
        out = self.norm(out + residual)
        return out

# Deep MLP Head
class DeepMLPHead(nn.Module):
    """A deep MLP detection head with two hidden layers, GELU activation, and dropout for regularization.
    
    This head is designed for complex feature transformations and aligns with modern transformer-based architectures.
    
    Args:
        dim (int): Input dimension (embedding size from VisionLSTM2).
        num_classes (int): Number of object classes to detect.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension in the MLP. Defaults to 512.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.hidden_dim = hidden_dim
        self.init_weights = init_weights
        self.num_blocks = num_blocks
        self.output_size = num_boxes * (5 + num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.output_size),
        )
        self.reset_parameters()

    def forward(self, x):
        """Forward pass to generate detection predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, dim].
        
        Returns:
            torch.Tensor: Detection predictions of shape [batch_size, num_patches, num_boxes, 5 + num_classes].
        """
        batch_size, num_patches, _ = x.size()
        predictions = self.mlp(x)
        predictions = predictions.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)
        return predictions

    def reset_parameters(self):
        """Initialize parameters following the library's conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.init_weights == "original":
                    small_init_(module.weight, dim=self.dim)
                elif self.init_weights == "original-fixed":
                    if self.num_blocks is None:
                        raise ValueError("num_blocks must be specified for 'original-fixed' initialization")
                    wang_init_(module.weight, dim=self.dim, num_blocks=self.num_blocks)
                else:
                    raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extra_repr(self):
        """String representation of the module."""
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, num_boxes={self.num_boxes}, "
            f"hidden_dim={self.hidden_dim}, init_weights={self.init_weights}, num_blocks={self.num_blocks}"
        )

# Separated MLP Head
class SeparatedMLPHead(nn.Module):
    """A detection head with separate MLPs for box regression and classification, optimized for task-specific performance.
    
    The box regression head uses a 3-layer MLP, while the classification head uses a single linear layer, inspired by DETR.
    
    Args:
        dim (int): Input dimension (embedding size from VisionLSTM2).
        num_classes (int): Number of object classes to detect.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension for the box regression MLP. Defaults to 512.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.hidden_dim = hidden_dim
        self.init_weights = init_weights
        self.num_blocks = num_blocks
        self.box_output_size = num_boxes * 5
        self.cls_output_size = num_boxes * num_classes
        self.box_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.box_output_size),
        )
        self.cls_mlp = nn.Linear(dim, self.cls_output_size)  # Single linear layer for classification
        self.reset_parameters()

    def forward(self, x):
        """Forward pass to generate detection predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, dim].
        
        Returns:
            torch.Tensor: Detection predictions of shape [batch_size, num_patches, num_boxes, 5 + num_classes].
        """
        batch_size, num_patches, _ = x.size()
        box_predictions = self.box_mlp(x)
        cls_predictions = self.cls_mlp(x)
        box_predictions = box_predictions.view(batch_size, num_patches, self.num_boxes, 5)
        cls_predictions = cls_predictions.view(batch_size, num_patches, self.num_boxes, self.num_classes)
        predictions = torch.cat([box_predictions, cls_predictions], dim=-1)
        return predictions

    def reset_parameters(self):
        """Initialize parameters following the library's conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.init_weights == "original":
                    small_init_(module.weight, dim=self.dim)
                elif self.init_weights == "original-fixed":
                    if self.num_blocks is None:
                        raise ValueError("num_blocks must be specified for 'original-fixed' initialization")
                    wang_init_(module.weight, dim=self.dim, num_blocks=self.num_blocks)
                else:
                    raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extra_repr(self):
        """String representation of the module."""
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, num_boxes={self.num_boxes}, "
            f"hidden_dim={self.hidden_dim}, init_weights={self.init_weights}, num_blocks={self.num_blocks}"
        )

# Residual MLP Head
class ResidualMLPHead(nn.Module):
    """A detection head with residual connections, GELU activation, dropout, and layer normalization for improved training stability.
    
    This head is particularly useful for deeper networks, ensuring stable gradients and better performance.
    
    Args:
        dim (int): Input dimension (embedding size from VisionLSTM2).
        num_classes (int): Number of object classes to detect.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension in the MLP. Defaults to 512.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.num_boxes = num_boxes
        self.hidden_dim = hidden_dim
        self.init_weights = init_weights
        self.num_blocks = num_blocks
        self.output_size = num_boxes * (5 + num_classes)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, self.output_size),
        )
        self.reset_parameters()

    def forward(self, x):
        """Forward pass to generate detection predictions.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, num_patches, dim].
        
        Returns:
            torch.Tensor: Detection predictions of shape [batch_size, num_patches, num_boxes, 5 + num_classes].
        """
        batch_size, num_patches, _ = x.size()
        predictions = self.mlp(x)
        predictions = predictions.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)
        return predictions

    def reset_parameters(self):
        """Initialize parameters following the library's conventions."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if self.init_weights == "original":
                    small_init_(module.weight, dim=self.dim)
                elif self.init_weights == "original-fixed":
                    if self.num_blocks is None:
                        raise ValueError("num_blocks must be specified for 'original-fixed' initialization")
                    wang_init_(module.weight, dim=self.dim, num_blocks=self.num_blocks)
                else:
                    raise NotImplementedError(f"Unknown init_weights: {self.init_weights}")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def extra_repr(self):
        """String representation of the module."""
        return (
            f"dim={self.dim}, num_classes={self.num_classes}, num_boxes={self.num_boxes}, "
            f"hidden_dim={self.hidden_dim}, init_weights={self.init_weights}, num_blocks={self.num_blocks}"
        )
    

import torch
import torch.nn as nn

class DETRMLPHead(nn.Module):
    """A DETR-inspired MLP detection head with separate MLPs for box regression and classification.

    Inspired by DETR ("End-to-End Object Detection with Transformers"), this head uses a 3-layer MLP
    for box regression, mirroring DETR’s approach, and a single linear layer for classification.
    It processes patch embeddings to predict multiple boxes per patch in a YOLO-like format.

    Args:
        dim (int): Input dimension (embedding size from VisionLSTM2).
        num_classes (int): Number of object classes.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension in the box MLP. Defaults to 512.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        # Box regression: 3-layer MLP as in DETR
        self.box_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),  # Modern activation from latest research
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_boxes * 5),  # 5: [x, y, w, h, confidence]
        )

        # Classification: Single linear layer as in DETR
        self.cls_mlp = nn.Linear(dim, num_boxes * num_classes)

        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        box_preds = self.box_mlp(x)  # [batch_size, num_patches, num_boxes * 5]
        cls_preds = self.cls_mlp(x)  # [batch_size, num_patches, num_boxes * num_classes]
        
        # Reshape to YOLO-like format
        box_preds = box_preds.view(batch_size, num_patches, self.num_boxes, 5)
        cls_preds = cls_preds.view(batch_size, num_patches, self.num_boxes, self.num_classes)
        return torch.cat([box_preds, cls_preds], dim=-1)  # [batch_size, num_patches, num_boxes, 5 + num_classes]

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class DINOMLPHead(DETRMLPHead):
    """A DINO-inspired MLP detection head, currently mirroring DETRMLPHead.

    Based on DINO ("DETR with Improved DeNoising"), which extends DETR, this head uses the same
    structure as DETRMLPHead. Future extensions could incorporate denoising mechanisms or anchor
    box adjustments if specified by the latest research context.

    Args:
        Same as DETRMLPHead.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Note: Could be extended with DINO-specific features (e.g., anchor box denoising) if needed


# class YOLOMLPHead(nn.Module):
#     """A YOLO-inspired MLP detection head with a 2-layer MLP for efficiency.

#     Inspired by YOLO’s efficient grid-based detection, this head uses a streamlined 2-layer MLP to
#     predict box coordinates, confidence, and class probabilities per patch, adapted for MLP use.

#     Args:
#         dim (int): Input dimension (embedding size from VisionLSTM2).
#         num_classes (int): Number of object classes.
#         num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
#         hidden_dim (int, optional): Hidden layer dimension in the MLP. Defaults to 512.
#         dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
#         init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
#         num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
#     """
#     def __init__(
#         self,
#         dim,
#         num_classes,
#         num_boxes=2,
#         hidden_dim=512,
#         dropout=0.1,
#         init_weights="original",
#         num_blocks=None,
#     ):
#         super().__init__()
#         self.num_boxes = num_boxes
#         self.num_classes = num_classes

#         # Simple 2-layer MLP for YOLO-like efficiency
#         self.mlp = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, num_boxes * (5 + num_classes)),
#         )

#         self.reset_parameters(init_weights, num_blocks)

#     def forward(self, x):
#         batch_size, num_patches, _ = x.size()
#         preds = self.mlp(x)  # [batch_size, num_patches, num_boxes * (5 + num_classes)]
#         return preds.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)

#     def reset_parameters(self, init_weights, num_blocks):
#         for module in self.modules():
#             if isinstance(module, nn.Linear):
#                 if init_weights == "original":
#                     small_init_(module.weight, dim=module.weight.size(1))
#                 elif init_weights == "original-fixed":
#                     if num_blocks is None:
#                         raise ValueError("num_blocks required for 'original-fixed'")
#                     wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
#                 if module.bias is not None:
#                     nn.init.zeros_(module.bias)


class ResidualMLPHead(nn.Module):
    """An advanced MLP detection head with residual connections for improved performance.

    Incorporates residual blocks with layer normalization, reflecting modern research trends for
    deeper, more stable architectures.

    Args:
        dim (int): Input dimension (embedding size from VisionLSTM2).
        num_classes (int): Number of object classes.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension in the MLP. Defaults to 512.
        dropout (float, optional): Dropout probability for regularization. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.num_classes = num_classes

        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
            nn.Linear(hidden_dim, num_boxes * (5 + num_classes)),
        )

        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)


class ResidualBlock(nn.Module):
    """A residual block with layer normalization for the ResidualMLPHead."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual
    
import torch
import torch.nn as nn

# Utility functions for weight initialization (assumed to exist)
def small_init_(weight, dim):
    nn.init.normal_(weight, mean=0.0, std=(1.0 / dim) ** 0.5)

def wang_init_(weight, dim, num_blocks):
    nn.init.normal_(weight, mean=0.0, std=(2.0 / (dim * num_blocks)) ** 0.5)

class ResidualBlock(nn.Module):
    """A residual block with layer normalization for ResidualMLPHead."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual

class DeepMLPHead(nn.Module):
    """A deep MLP detection head with configurable hidden layers, GELU, and dropout.

    Args:
        dim (int): Input dimension.
        num_classes (int): Number of object classes.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 512.
        num_hidden_layers (int, optional): Number of hidden layers (excluding input/output). Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        num_hidden_layers=2,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.output_size = num_boxes * (5 + num_classes)
        layers = [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class SeparatedMLPHead(nn.Module):
    """A detection head with separate MLPs for box regression and classification, inspired by DETR.

    Args:
        Same as DeepMLPHead, with fixed 3-layer box MLP and single-layer classification.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.box_output_size = num_boxes * 5
        self.cls_output_size = num_boxes * num_classes
        self.box_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.box_output_size),
        )
        self.cls_mlp = nn.Linear(dim, self.cls_output_size)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        box_preds = self.box_mlp(x)
        cls_preds = self.cls_mlp(x)
        box_preds = box_preds.view(batch_size, num_patches, self.num_boxes, 5)
        cls_preds = cls_preds.view(batch_size, num_patches, self.num_boxes, self.num_classes)
        return torch.cat([box_preds, cls_preds], dim=-1)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class ResidualMLPHead(nn.Module):
    """A detection head with configurable residual blocks, GELU, and dropout for stability.

    Args:
        Same as DeepMLPHead, with configurable number of residual blocks.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        num_residual_blocks=2,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.output_size = num_boxes * (5 + num_classes)
        layers = [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class GatedMLPHead(nn.Module):
    """A detection head with gated mechanisms, inspired by Mamba YOLO, for enhanced local dependency modeling.

    Args:
        Same as DeepMLPHead, with gated aggregation for improved feature capture.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        num_hidden_layers=2,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.output_size = num_boxes * (5 + num_classes)
        layers = [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim * 2),  # For gating
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),  # Gated output
                nn.Sigmoid(),  # Gating function
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)



import torch
import torch.nn as nn

# Utility functions for weight initialization (assumed to exist)
def small_init_(weight, dim):
    nn.init.normal_(weight, mean=0.0, std=(1.0 / dim) ** 0.5)

def wang_init_(weight, dim, num_blocks):
    nn.init.normal_(weight, mean=0.0, std=(2.0 / (dim * num_blocks)) ** 0.5)

class ResidualBlock(nn.Module):
    """A residual block with layer normalization for ResidualMLPHead."""
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.linear = nn.Linear(dim, dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x + residual

class DeepMLPHead(nn.Module):
    """A deep MLP detection head with configurable hidden layers, GELU, and dropout.

    Args:
        dim (int): Input dimension.
        num_classes (int): Number of object classes.
        num_boxes (int, optional): Number of bounding boxes per patch. Defaults to 2.
        hidden_dim (int, optional): Hidden layer dimension. Defaults to 512.
        num_hidden_layers (int, optional): Number of hidden layers (excluding input/output). Defaults to 2.
        dropout (float, optional): Dropout probability. Defaults to 0.1.
        init_weights (str, optional): Weight initialization method ('original' or 'original-fixed').
        num_blocks (int, optional): Number of blocks for 'original-fixed' initialization.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        num_hidden_layers=2,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.output_size = num_boxes * (5 + num_classes)
        layers = [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_hidden_layers):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.Linear(hidden_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + self.num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class SeparatedMLPHead(nn.Module):
    """A detection head with separate MLPs for box regression and classification, inspired by DETR.

    Args:
        Same as DeepMLPHead, with fixed 3-layer box MLP and single-layer classification.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.box_output_size = num_boxes * 5
        self.cls_output_size = num_boxes * num_classes
        self.box_mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.box_output_size),
        )
        self.cls_mlp = nn.Linear(dim, self.cls_output_size)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        box_preds = self.box_mlp(x)
        cls_preds = self.cls_mlp(x)
        box_preds = box_preds.view(batch_size, num_patches, self.num_boxes, 5)
        cls_preds = cls_preds.view(batch_size, num_patches, self.num_boxes, self.num_classes)
        return torch.cat([box_preds, cls_preds], dim=-1)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class ResidualMLPHead(nn.Module):
    """A detection head with configurable residual blocks, GELU, and dropout for stability.

    Args:
        Same as DeepMLPHead, with configurable number of residual blocks.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        num_residual_blocks=2,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.output_size = num_boxes * (5 + num_classes)
        layers = [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_residual_blocks):
            layers.append(ResidualBlock(hidden_dim, dropout))
        layers.append(nn.Linear(hidden_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

class GatedMLPHead(nn.Module):
    """A detection head with gated mechanisms, inspired by Mamba YOLO, for enhanced local dependency modeling.

    Args:
        Same as DeepMLPHead, with gated aggregation for improved feature capture.
    """
    def __init__(
        self,
        dim,
        num_classes,
        num_boxes=2,
        hidden_dim=512,
        num_hidden_layers=2,
        dropout=0.1,
        init_weights="original",
        num_blocks=None,
    ):
        super().__init__()
        self.num_boxes = num_boxes
        self.output_size = num_boxes * (5 + num_classes)
        layers = [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_hidden_layers):
            layers.extend([
                nn.Linear(hidden_dim, hidden_dim * 2),  # For gating
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),  # Gated output
                nn.Sigmoid(),  # Gating function
                nn.Dropout(dropout)
            ])
        layers.append(nn.Linear(hidden_dim, self.output_size))
        self.mlp = nn.Sequential(*layers)
        self.reset_parameters(init_weights, num_blocks)

    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        preds = self.mlp(x)
        return preds.view(batch_size, num_patches, self.num_boxes, 5 + num_classes)

    def reset_parameters(self, init_weights, num_blocks):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if init_weights == "original":
                    small_init_(module.weight, dim=module.weight.size(1))
                elif init_weights == "original-fixed":
                    if num_blocks is None:
                        raise ValueError("num_blocks required for 'original-fixed'")
                    wang_init_(module.weight, dim=module.weight.size(1), num_blocks=num_blocks)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)