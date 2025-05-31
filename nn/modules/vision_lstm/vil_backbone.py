# vision_lstm_wrapper.py
import torch
import torch.nn as nn
from typing import List

class multi_layer_feature_extractor(nn.Module):
    """Extracts features from multiple layers of a backbone model."""
    
    def __init__(self, backbone: nn.Module, layers_to_extract: List[int]):
        """
        Initialize the feature extractor.

        Args:
            backbone (nn.Module): The backbone model (e.g., VisionLSTM2).
            layers_to_extract (List[int]): Indices of layers to extract features from.
        """
        super().__init__()
        self.backbone = backbone
        self.layers_to_extract = sorted(layers_to_extract)
        self.hooks = []
        self.features = {}
        
        # Register forward hooks
        for idx in self.layers_to_extract:
            hook = self.backbone.keys[idx].register_forward_hook(
                lambda m, i, o, idx=idx: self.features.update({idx: o})
            )
            self.hooks.append(hook)
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Extract features from specified layers.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            List[torch.Tensor]: Feature tensors from specified layers, each shaped (B, C, H, W).
        
        Raises:
            ValueError: If input tensor shape does not match backbone expectations.
        """
        if x.shape[1:] != self.backbone.input_shape:
            raise ValueError(f"Input shape {x.shape[1:]} does not match expected {self.backbone.input_shape}")
        
        self.features.clear()
        _ = self.backbone(x)
        features = [self.features[idx] for idx in self.layers_to_extract]
        
        # Reshape from (B, N, C) to (B, C, H, W)
        B = x.shape[0]
        grid_size = int((self.backbone.input_shape[1] // self.backbone.patch_size) ** 0.5)
        reshaped_features = [
            feat.view(B, grid_size, grid_size, -1).permute(0, 3, 1, 2) for feat in features
        ]
        return reshaped_features
    
    def __del__(self):
        """Remove hooks on object deletion."""
        for hook in self.hooks:
            hook.remove()

if __name__ == "__main__":
    from vision_lstm import VisionLSTM2
    backbone = VisionLSTM2(dim=192, input_shape=(3, 512, 512), patch_size=32, depth=12, mode="features", pooling=None)
    extractor = multi_layer_feature_extractor(backbone, [9, 10, 11])
    x = torch.randn(2, 3, 512, 512)
    features = extractor(x)
    for i, feat in enumerate(features):
        print(f"Feature {i} shape: {feat.shape}")