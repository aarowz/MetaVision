import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io as sio
from scipy import ndimage
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import random

"""
Data Loader Module

PyTorch Dataset and DataLoader for loading and preprocessing metasurface data.
Handles .mat file loading, input/output preprocessing, normalization, and train/val/test splits.
"""


class MetasurfaceDataset(Dataset):
    """PyTorch Dataset for metasurface geometry and EM field data."""
    
    def __init__(
        self,
        data_dir: Path,
        file_indices: List[int],
        split: str = 'train',
        output_resolution: Optional[Tuple[int, int]] = None,
        normalize_input: bool = True,
        normalize_output: bool = True,
        input_norm_params: Optional[Dict[str, Tuple[float, float]]] = None,
        output_norm_params: Optional[Dict[str, float]] = None,
        seed: int = 42
    ):
        """Initialize dataset with file paths, split configuration, and normalization parameters."""
        self.data_dir = Path(data_dir)
        self.file_indices = file_indices
        self.split = split
        self.output_resolution = output_resolution or (120, 120)  # Default to input resolution
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output
        
        # Set normalization parameters
        if input_norm_params is None:
            self.input_norm_params = self._get_default_input_norm_params()
        else:
            self.input_norm_params = input_norm_params
            
        if output_norm_params is None:
            self.output_norm_params = self._get_default_output_norm_params()
        else:
            self.output_norm_params = output_norm_params
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return a single sample (input geometry, output EM fields) as tensors."""
        file_idx = self.file_indices[idx]
        file_path = self.data_dir / f'data3d_{file_idx}.mat'
        
        # Load data
        data = self._load_mat_file(file_path)
        
        # Preprocess input
        input_array = self._preprocess_input(data['R'], data['H'], data['D'])
        
        # Preprocess output
        output_array = self._preprocess_output(data['Ex'], data['Ey'], data['Ez'])
        
        # Convert to tensors
        input_tensor = torch.from_numpy(input_array).float()
        output_tensor = torch.from_numpy(output_array).float()
        
        return input_tensor, output_tensor
    
    def _load_mat_file(self, file_path: Path) -> Dict[str, np.ndarray]:
        """Load a .mat file and return dictionary with R, H, D, Ex, Ey, Ez arrays."""
        data = sio.loadmat(str(file_path))
        return {
            'R': data['R'],
            'H': data['H'],
            'D': data['D'],
            'Ex': data['Ex'],
            'Ey': data['Ey'],
            'Ez': data['Ez']
        }
    
    def _preprocess_input(
        self,
        R: np.ndarray,
        H: np.ndarray,
        D: np.ndarray
    ) -> np.ndarray:
        """Stack R, H, D[0], D[1] into 4-channel array and normalize if enabled."""
        # Stack channels: [R, H, D[0], D[1]]
        input_array = np.stack([R, H, D[0], D[1]], axis=0)  # Shape: [4, H, W]
        
        # Normalize each channel if enabled
        if self.normalize_input:
            for channel_idx in range(4):
                input_array[channel_idx] = self._normalize_input(
                    input_array[channel_idx], channel_idx
                )
        
        return input_array
    
    def _preprocess_output(
        self,
        Ex: np.ndarray,
        Ey: np.ndarray,
        Ez: np.ndarray
    ) -> np.ndarray:
        """Convert complex fields to real+imag channels, downsample, and normalize if enabled."""
        # Convert complex to 6 channels (real+imag for each field)
        output_array = self._complex_to_channels(Ex, Ey, Ez)
        
        # Downsample each channel
        downsampled = np.zeros(
            (output_array.shape[0], self.output_resolution[0], self.output_resolution[1]),
            dtype=output_array.dtype
        )
        for channel_idx in range(output_array.shape[0]):
            downsampled[channel_idx] = self._downsample_output(
                output_array[channel_idx], self.output_resolution
            )
        
        # Normalize if enabled
        if self.normalize_output:
            # Normalize each field (2 channels per field: real+imag)
            field_names = ['Ex', 'Ey', 'Ez']
            for field_idx, field_name in enumerate(field_names):
                channel_start = field_idx * 2
                channel_end = channel_start + 2
                downsampled[channel_start:channel_end] = self._normalize_output(
                    downsampled[channel_start:channel_end], field_name
                )
        
        return downsampled
    
    def _downsample_output(
        self,
        field: np.ndarray,
        target_shape: Tuple[int, int]
    ) -> np.ndarray:
        """Downsample EM field from 2883x2883 to target resolution using interpolation."""
        if field.shape == target_shape:
            return field
        
        # Calculate zoom factors
        zoom_factors = (
            target_shape[0] / field.shape[0],
            target_shape[1] / field.shape[1]
        )
        
        # Use order=1 (bilinear) for interpolation
        downsampled = ndimage.zoom(field, zoom_factors, order=1)
        
        return downsampled
    
    def _normalize_input(
        self,
        input_array: np.ndarray,
        channel_idx: int
    ) -> np.ndarray:
        """Normalize a single input channel using min-max scaling to [0, 1]."""
        channel_names = ['R', 'H', 'D0', 'D1']
        channel_name = channel_names[channel_idx]
        
        min_val, max_val = self.input_norm_params[channel_name]
        
        # Min-max normalization to [0, 1]
        normalized = (input_array - min_val) / (max_val - min_val)
        
        return normalized
    
    def _normalize_output(
        self,
        output_array: np.ndarray,
        field_name: str
    ) -> np.ndarray:
        """Normalize output field channels using max absolute value scaling."""
        norm_max = self.output_norm_params[field_name]
        
        # Normalize by max absolute value (scales to roughly [-1, 1] range)
        normalized = output_array / norm_max
        
        return normalized
    
    def _complex_to_channels(
        self,
        Ex: np.ndarray,
        Ey: np.ndarray,
        Ez: np.ndarray
    ) -> np.ndarray:
        """Convert complex Ex, Ey, Ez fields to 6-channel array (real+imag for each)."""
        # Stack as [Ex_real, Ex_imag, Ey_real, Ey_imag, Ez_real, Ez_imag]
        channels = np.stack([
            np.real(Ex),
            np.imag(Ex),
            np.real(Ey),
            np.imag(Ey),
            np.real(Ez),
            np.imag(Ez)
        ], axis=0)
        
        return channels
    
    def _get_default_input_norm_params(self) -> Dict[str, Tuple[float, float]]:
        """Return default input normalization parameters (min, max) for each channel."""
        # These are placeholder values - should be computed from analyze_ranges.py
        # or passed in during initialization
        return {
            'R': (0.0, 10.0),
            'H': (0.0, 100.0),
            'D0': (-10.0, 10.0),
            'D1': (-10.0, 10.0)
        }
    
    def _get_default_output_norm_params(self) -> Dict[str, float]:
        """Return default output normalization parameters (max abs value) for each field."""
        # These are placeholder values - should be computed from analyze_ranges.py
        # or passed in during initialization
        return {
            'Ex': 1000.0,
            'Ey': 1000.0,
            'Ez': 1000.0
        }


def create_train_val_test_splits(
    total_files: int = 11,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits from file indices.
    
    Args:
        total_files: Total number of data files (default: 11)
        train_ratio: Fraction of files for training (default: 0.7)
        val_ratio: Fraction of files for validation (default: 0.15)
        test_ratio: Fraction of files for testing (default: 0.15)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Create list of all file indices
    all_indices = list(range(total_files))
    
    # Set random seed
    random.seed(seed)
    np.random.seed(seed)
    
    # Shuffle indices
    shuffled_indices = all_indices.copy()
    random.shuffle(shuffled_indices)
    
    # Calculate split sizes
    n_train = int(total_files * train_ratio)
    n_val = int(total_files * val_ratio)
    n_test = total_files - n_train - n_val  # Remaining goes to test
    
    # Split indices
    train_indices = shuffled_indices[:n_train]
    val_indices = shuffled_indices[n_train:n_train + n_val]
    test_indices = shuffled_indices[n_train + n_val:]
    
    return train_indices, val_indices, test_indices
