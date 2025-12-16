"""
Test script for MetaVisionViT model forward pass.

Verifies that the model correctly processes input tensors and produces
the expected output shapes.
"""

import sys
from pathlib import Path
import torch
import yaml

# Add src to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model import create_model


def test_forward_pass():
    """Test model forward pass with dummy input."""
    
    # Load config
    config_path = PROJECT_ROOT / 'config.yaml'
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 1
    dummy_input = torch.randn(batch_size, 4, 120, 120)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Expected output shape: [{batch_size}, 6, 120, 120]")
    
    # Forward pass
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")
    
    # Verify output shape
    expected_shape = (batch_size, 6, 120, 120)
    assert output.shape == expected_shape, \
        f"Output shape mismatch! Got {output.shape}, expected {expected_shape}"
    
    print("\n✓ Forward pass test passed!")
    
    # Test with different batch sizes
    print("\nTesting with different batch sizes...")
    for bs in [1, 2, 4]:
        test_input = torch.randn(bs, 4, 120, 120)
        with torch.no_grad():
            test_output = model(test_input)
        assert test_output.shape == (bs, 6, 120, 120), \
            f"Batch size {bs} failed: got {test_output.shape}"
        print(f"  Batch size {bs}: ✓")
    
    print("\n✓ All tests passed!")


if __name__ == '__main__':
    test_forward_pass()

