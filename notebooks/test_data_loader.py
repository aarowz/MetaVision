"""
Test script for MetasurfaceDataset data loader.

Tests basic functionality including:
- File loading
- Shape verification
- Normalization
- Train/val/test splits
- DataLoader integration
"""

import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

# Add src to path
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from data_loader import MetasurfaceDataset, create_train_val_test_splits


def test_basic_loading():
    """Test 1: Basic file loading and shape verification."""
    print("="*60)
    print("TEST 1: Basic Loading and Shapes")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=[0],
        normalize_input=False,  # Test without normalization first
        normalize_output=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Load a sample
    input_tensor, output_tensor = dataset[0]
    
    print(f"\nInput shape: {input_tensor.shape}")
    print(f"Expected input shape: [4, 120, 120]")
    print(f"Input dtype: {input_tensor.dtype}")
    print(f"Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    
    print(f"\nOutput shape: {output_tensor.shape}")
    print(f"Expected output shape: [6, 120, 120] (6 channels: Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag)")
    print(f"Output dtype: {output_tensor.dtype}")
    print(f"Output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
    
    # Verify shapes
    assert input_tensor.shape == (4, 120, 120), f"Input shape mismatch: {input_tensor.shape}"
    assert output_tensor.shape == (6, 120, 120), f"Output shape mismatch: {output_tensor.shape}"
    assert input_tensor.dtype == torch.float32, f"Input dtype mismatch: {input_tensor.dtype}"
    assert output_tensor.dtype == torch.float32, f"Output dtype mismatch: {output_tensor.dtype}"
    
    print("\n✓ Basic loading test passed!")


def test_normalization():
    """Test 2: Normalization functionality."""
    print("\n" + "="*60)
    print("TEST 2: Normalization")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    
    # Test without normalization
    dataset_no_norm = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=[0],
        normalize_input=False,
        normalize_output=False
    )
    input_no_norm, output_no_norm = dataset_no_norm[0]
    
    # Test with normalization
    dataset_norm = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=[0],
        normalize_input=True,
        normalize_output=True
    )
    input_norm, output_norm = dataset_norm[0]
    
    print(f"\nInput without normalization:")
    print(f"  Range: [{input_no_norm.min():.3f}, {input_no_norm.max():.3f}]")
    print(f"  Mean: {input_no_norm.mean():.3f}")
    
    print(f"\nInput with normalization:")
    print(f"  Range: [{input_norm.min():.3f}, {input_norm.max():.3f}]")
    print(f"  Mean: {input_norm.mean():.3f}")
    
    # Check that normalized values are in reasonable range
    # Input should be roughly [0, 1] after normalization
    assert input_norm.min() >= -0.1, f"Normalized input min too low: {input_norm.min()}"
    assert input_norm.max() <= 1.1, f"Normalized input max too high: {input_norm.max()}"
    
    print(f"\nOutput without normalization:")
    print(f"  Range: [{output_no_norm.min():.3f}, {output_no_norm.max():.3f}]")
    
    print(f"\nOutput with normalization:")
    print(f"  Range: [{output_norm.min():.3f}, {output_norm.max():.3f}]")
    
    print("\n✓ Normalization test passed!")


def test_different_resolutions():
    """Test 3: Different output resolutions."""
    print("\n" + "="*60)
    print("TEST 3: Different Output Resolutions")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    
    resolutions = [(120, 120), (240, 240), (60, 60)]
    
    for res in resolutions:
        dataset = MetasurfaceDataset(
            data_dir=data_dir,
            file_indices=[0],
            output_resolution=res,
            normalize_input=False,
            normalize_output=False
        )
        _, output = dataset[0]
        print(f"Resolution {res}: Output shape = {output.shape}")
        assert output.shape == (6, res[0], res[1]), f"Shape mismatch for resolution {res}"
    
    print("\n✓ Resolution test passed!")


def test_multiple_files():
    """Test 4: Loading multiple files."""
    print("\n" + "="*60)
    print("TEST 4: Multiple Files")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=[0, 1, 2],
        normalize_input=False,
        normalize_output=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 3, f"Expected 3 files, got {len(dataset)}"
    
    # Load all samples
    for i in range(len(dataset)):
        input_tensor, output_tensor = dataset[i]
        print(f"  Sample {i}: Input {input_tensor.shape}, Output {output_tensor.shape}")
        assert input_tensor.shape == (4, 120, 120), f"Input shape mismatch for sample {i}"
        assert output_tensor.shape == (6, 120, 120), f"Output shape mismatch for sample {i}"
    
    print("\n✓ Multiple files test passed!")


def test_splits():
    """Test 5: Train/val/test splits."""
    print("\n" + "="*60)
    print("TEST 5: Train/Val/Test Splits")
    print("="*60)
    
    train_idx, val_idx, test_idx = create_train_val_test_splits(
        total_files=11,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        seed=42
    )
    
    print(f"Train indices ({len(train_idx)}): {train_idx}")
    print(f"Val indices ({len(val_idx)}): {val_idx}")
    print(f"Test indices ({len(test_idx)}): {test_idx}")
    
    # Verify no overlap
    all_indices = set(train_idx) | set(val_idx) | set(test_idx)
    assert len(all_indices) == len(train_idx) + len(val_idx) + len(test_idx), \
        "Overlap detected in splits!"
    
    # Verify all indices are present
    assert all_indices == set(range(11)), "Not all files included in splits!"
    
    print("\n✓ Splits test passed!")


def test_dataloader():
    """Test 6: PyTorch DataLoader integration."""
    print("\n" + "="*60)
    print("TEST 6: DataLoader Integration")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=[0, 1, 2, 3],
        normalize_input=False,
        normalize_output=False
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0  # Set to 0 to avoid multiprocessing issues in testing
    )
    
    print(f"DataLoader created with batch_size=2, shuffle=True")
    print(f"Dataset size: {len(dataset)}")
    
    # Test one batch
    for batch_idx, (inputs, outputs) in enumerate(dataloader):
        print(f"\nBatch {batch_idx}:")
        print(f"  Input batch shape: {inputs.shape}")
        print(f"  Output batch shape: {outputs.shape}")
        print(f"  Expected input shape: [batch_size, 4, 120, 120]")
        print(f"  Expected output shape: [batch_size, 6, 120, 120]")
        
        assert inputs.shape[0] == 2 or inputs.shape[0] == len(dataset), \
            f"Unexpected batch size: {inputs.shape[0]}"
        assert inputs.shape[1:] == (4, 120, 120), f"Input shape mismatch: {inputs.shape}"
        assert outputs.shape[1:] == (6, 120, 120), f"Output shape mismatch: {outputs.shape}"
        
        if batch_idx >= 1:  # Just test first 2 batches
            break
    
    print("\n✓ DataLoader test passed!")


def test_custom_norm_params():
    """Test 7: Custom normalization parameters."""
    print("\n" + "="*60)
    print("TEST 7: Custom Normalization Parameters")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    
    # Custom normalization parameters
    custom_input_norm = {
        'R': (0.0, 5.0),
        'H': (0.0, 50.0),
        'D0': (-5.0, 5.0),
        'D1': (-5.0, 5.0)
    }
    
    custom_output_norm = {
        'Ex': 500.0,
        'Ey': 500.0,
        'Ez': 500.0
    }
    
    dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=[0],
        normalize_input=True,
        normalize_output=True,
        input_norm_params=custom_input_norm,
        output_norm_params=custom_output_norm
    )
    
    input_tensor, output_tensor = dataset[0]
    
    print(f"Using custom normalization parameters:")
    print(f"  Input norm params: {custom_input_norm}")
    print(f"  Output norm params: {custom_output_norm}")
    print(f"  Input range: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
    print(f"  Output range: [{output_tensor.min():.3f}, {output_tensor.max():.3f}]")
    
    print("\n✓ Custom normalization test passed!")


def test_all_files_loadable():
    """Test 8: Verify all 11 files can be loaded successfully."""
    print("\n" + "="*60)
    print("TEST 8: All Files Loadable")
    print("="*60)
    
    data_dir = PROJECT_ROOT / 'data' / 'raw'
    
    # Try to load all 11 files
    all_indices = list(range(11))
    dataset = MetasurfaceDataset(
        data_dir=data_dir,
        file_indices=all_indices,
        normalize_input=False,
        normalize_output=False
    )
    
    print(f"Dataset length: {len(dataset)}")
    assert len(dataset) == 11, f"Expected 11 files, got {len(dataset)}"
    
    # Try loading each file
    for i in range(11):
        try:
            input_tensor, output_tensor = dataset[i]
            print(f"  File {i}: ✓ Loaded successfully")
            print(f"    Input shape: {input_tensor.shape}, Output shape: {output_tensor.shape}")
            # Verify shapes are correct
            assert input_tensor.shape == (4, 120, 120), \
                f"File {i}: Input shape mismatch: {input_tensor.shape}"
            assert output_tensor.shape == (6, 120, 120), \
                f"File {i}: Output shape mismatch: {output_tensor.shape}"
        except Exception as e:
            print(f"  File {i}: ❌ Failed to load - {e}")
            raise
    
    print("\n✓ All files loadable test passed!")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("METASURFACE DATASET TEST SUITE")
    print("="*60)
    
    try:
        test_basic_loading()
        test_normalization()
        test_different_resolutions()
        test_multiple_files()
        test_splits()
        test_dataloader()
        test_custom_norm_params()
        test_all_files_loadable()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
