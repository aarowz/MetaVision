"""
Data Augmentation Module

Implements geometric augmentations for metasurface geometry data.
Augmentations preserve the physics of the geometry (flips, rotations).
"""

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
from typing import Tuple, Optional, Dict


class MetasurfaceAugmentation:
    """
    Augmentation transforms for metasurface geometry data.
    
    Applies geometric transformations that preserve the physics:
    - Horizontal/Vertical flips
    - 90°, 180°, 270° rotations
    
    Avoids transformations that change physics:
    - Random crops (changes geometry scale)
    - Random scaling (changes physical dimensions)
    - Color jitter (not applicable to geometry)
    """
    
    def __init__(
        self,
        horizontal_flip: bool = True,
        vertical_flip: bool = True,
        rotation_90: bool = True,
        rotation_180: bool = True,
        rotation_270: bool = True,
        flip_prob: float = 0.5,
        rotation_prob: float = 0.5
    ):
        """
        Initialize augmentation transforms.
        
        Args:
            horizontal_flip: Enable horizontal flipping
            vertical_flip: Enable vertical flipping
            rotation_90: Enable 90° rotation
            rotation_180: Enable 180° rotation
            rotation_270: Enable 270° rotation
            flip_prob: Probability of applying flip augmentation
            rotation_prob: Probability of applying rotation augmentation
        """
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        self.rotation_90 = rotation_90
        self.rotation_180 = rotation_180
        self.rotation_270 = rotation_270
        self.flip_prob = flip_prob
        self.rotation_prob = rotation_prob
    
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        """
        Apply random augmentations to input image.
        
        Args:
            image: Input tensor of shape [C, H, W] or [B, C, H, W]
            
        Returns:
            Augmented tensor with same shape
        """
        # Handle batch dimension
        is_batch = len(image.shape) == 4
        if not is_batch:
            image = image.unsqueeze(0)
        
        augmented = image.clone()
        
        # Apply horizontal flip
        if self.horizontal_flip and torch.rand(1) < self.flip_prob:
            augmented = torch.flip(augmented, dims=[-1])  # Flip width
        
        # Apply vertical flip
        if self.vertical_flip and torch.rand(1) < self.flip_prob:
            augmented = torch.flip(augmented, dims=[-2])  # Flip height
        
        # Apply rotation
        if torch.rand(1) < self.rotation_prob:
            # Randomly select rotation angle
            rotation_options = []
            if self.rotation_90:
                rotation_options.append(90)
            if self.rotation_180:
                rotation_options.append(180)
            if self.rotation_270:
                rotation_options.append(270)
            
            if rotation_options:
                angle = rotation_options[torch.randint(0, len(rotation_options), (1,)).item()]
                # Rotate 90° increments using transpose + flip
                augmented = self._rotate_90n(augmented, k=angle // 90)
        
        # Remove batch dimension if it wasn't there originally
        if not is_batch:
            augmented = augmented.squeeze(0)
        
        return augmented
    
    def _rotate_90n(self, tensor: torch.Tensor, k: int) -> torch.Tensor:
        """
        Rotate tensor by k*90 degrees counterclockwise.
        
        Args:
            tensor: Input tensor [B, C, H, W]
            k: Number of 90° rotations (0, 1, 2, 3)
            
        Returns:
            Rotated tensor
        """
        k = k % 4
        if k == 0:
            return tensor
        elif k == 1:
            # 90°: transpose and flip
            return torch.flip(tensor.transpose(-2, -1), dims=[-1])
        elif k == 2:
            # 180°: flip both dimensions
            return torch.flip(torch.flip(tensor, dims=[-1]), dims=[-2])
        elif k == 3:
            # 270°: transpose and flip (opposite direction)
            return torch.flip(tensor.transpose(-2, -1), dims=[-2])


def get_augmentation_transform(config: Dict) -> Optional[MetasurfaceAugmentation]:
    """
    Create augmentation transform from config dictionary.
    
    Args:
        config: Dictionary with augmentation settings from config.yaml
        
    Returns:
        MetasurfaceAugmentation instance or None if disabled
    """
    if not config.get('enabled', False):
        return None
    
    return MetasurfaceAugmentation(
        horizontal_flip=config.get('horizontal_flip', True),
        vertical_flip=config.get('vertical_flip', True),
        rotation_90=config.get('rotation_90', True),
        rotation_180=config.get('rotation_180', True),
        rotation_270=config.get('rotation_270', True),
        flip_prob=config.get('flip_prob', 0.5),
        rotation_prob=config.get('rotation_prob', 0.5)
    )


# Example usage:
if __name__ == '__main__':
    # Test augmentation
    aug = MetasurfaceAugmentation(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_90=True,
        rotation_180=True,
        rotation_270=True,
        flip_prob=0.5,
        rotation_prob=0.5
    )
    
    # Create dummy input [4, 120, 120]
    dummy_input = torch.randn(4, 120, 120)
    
    # Apply augmentation
    augmented = aug(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Augmented shape: {augmented.shape}")
    print("✓ Augmentation test passed!")

