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
        num_blocks_per_file: int = 2000,
        block_size: int = 15,
        compute_norm: bool = False,
        norm_cache_path: Optional[Path] = None,
        normalize_input: bool = True,
        input_norm_params: Optional[Dict[str, Tuple[float, float]]] = None,
        seed: int = 42
    ):
        """Initialize dataset with file paths, split configuration, and normalization parameters."""
        self.data_dir = Path(data_dir)
        self.file_indices = file_indices
        self.split = split
        self.num_blocks_per_file = num_blocks_per_file
        self.block_size = block_size
        self.compute_norm = compute_norm
        self.norm_cache_path = norm_cache_path or (self.data_dir.parent / 'processed' / 'norm_cache.pt')
        self.normalize_input = normalize_input
        
        # Set normalization parameters
        if input_norm_params is None:
            self.input_norm_params = self._get_default_input_norm_params()
        else:
            self.input_norm_params = input_norm_params
        
        # Normalization statistics (computed from training set)
        self.out_mean = None  # torch.Tensor, shape [6]
        self.out_std = None   # torch.Tensor, shape [6]
        
        # Handle normalization statistics
        if self.compute_norm:
            # Training dataset: load from cache if exists, otherwise will compute later
            if self.norm_cache_path.exists():
                self._load_normalization_stats()
        else:
            # Val/test dataset: load from cache (must exist)
            self._load_normalization_stats()
        
        # Store extracted windows (will be populated later)
        self.all_samples = []
        
        # Store seed for reproducibility in block extraction
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.file_indices) * self.num_blocks_per_file
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and return a single 15×15 block sample (input geometry, output EM fields) as tensors."""
        # Map linear index to (file_idx, block_idx)
        file_idx = self.file_indices[idx // self.num_blocks_per_file]
        block_idx = idx % self.num_blocks_per_file
        
        # Load file
        file_path = self.data_dir / f'data3d_{file_idx}.mat'
        data = self._load_mat_file(file_path)
        
        # Get geometry dimensions (should be 120×120)
        H_total, W_total = data['R'].shape
        
        # Generate random top-left indices for this block (reproducible)
        # Use RandomState to avoid global RNG pollution
        rng = np.random.RandomState(self.seed + file_idx * 10000 + block_idx)
        i = rng.randint(0, H_total - self.block_size + 1)
        j = rng.randint(0, W_total - self.block_size + 1)
        
        # Extract geometry block [4, 15, 15]
        R_block = data['R'][i:i+self.block_size, j:j+self.block_size]
        H_block = data['H'][i:i+self.block_size, j:j+self.block_size]
        D0_block = data['D'][0][i:i+self.block_size, j:j+self.block_size]
        D1_block = data['D'][1][i:i+self.block_size, j:j+self.block_size]
        
        # Stack into [4, 15, 15]
        geometry_block = np.stack([R_block, H_block, D0_block, D1_block], axis=0)
        
        # Calculate field scale (high-res / low-res)
        field_scale = data['Ex'].shape[0] // H_total  # e.g., 2883 // 120 = 24
        
        # Map geometry coordinates to field coordinates
        field_i = i * field_scale
        field_j = j * field_scale
        field_size = self.block_size * field_scale  # e.g., 15 * 24 = 360
        
        # Extract high-res field patch
        Ex_patch = data['Ex'][field_i:field_i+field_size, field_j:field_j+field_size]
        Ey_patch = data['Ey'][field_i:field_i+field_size, field_j:field_j+field_size]
        Ez_patch = data['Ez'][field_i:field_i+field_size, field_j:field_j+field_size]
        
        # Downsample using stride sampling (matches GAT)
        Ex_15x15 = Ex_patch[::field_scale, ::field_scale]  # Shape: [15, 15]
        Ey_15x15 = Ey_patch[::field_scale, ::field_scale]
        Ez_15x15 = Ez_patch[::field_scale, ::field_scale]
        
        # Convert complex to 6 channels [Ex_r, Ex_i, Ey_r, Ey_i, Ez_r, Ez_i] - RAW, unnormalized
        output_array = np.stack([
            np.real(Ex_15x15),
            np.imag(Ex_15x15),
            np.real(Ey_15x15),
            np.imag(Ey_15x15),
            np.real(Ez_15x15),
            np.imag(Ez_15x15)
        ], axis=0)  # Shape: [6, 15, 15]
        
        # Normalize geometry input (if enabled)
        if self.normalize_input:
            geometry_block = self._normalize_geometry_block(geometry_block)
        
        # Convert to tensors
        input_tensor = torch.from_numpy(geometry_block).float()  # [4, 15, 15]
        output_tensor = torch.from_numpy(output_array).float()   # [6, 15, 15] - RAW
        
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
    
    def _normalize_geometry_block(self, geometry_block: np.ndarray) -> np.ndarray:
        """Normalize geometry block [4, 15, 15] using min-max scaling per channel."""
        normalized_block = geometry_block.copy()
        
        # Normalize each channel
        for channel_idx in range(4):
            normalized_block[channel_idx] = self._normalize_input(
                geometry_block[channel_idx], channel_idx
            )
        
        return normalized_block
    
    def _load_normalization_stats(self):
        """Load normalization statistics from cache file."""
        if self.norm_cache_path.exists():
            cache = torch.load(self.norm_cache_path, map_location='cpu')
            self.out_mean = cache['out_mean']  # Shape [6]
            self.out_std = cache['out_std']    # Shape [6]
        else:
            if not self.compute_norm:
                raise FileNotFoundError(
                    f"Normalization cache not found: {self.norm_cache_path}. "
                    "Run compute_normalization_stats() on training dataset first."
                )
            # For training dataset, cache doesn't exist yet - will compute later
            self.out_mean = None
            self.out_std = None
    
    def compute_normalization_stats(self):
        """
        Compute per-channel mean and std from training dataset.
        Should only be called on training dataset.
        """
        if not self.compute_norm:
            raise ValueError("compute_normalization_stats() should only be called on training dataset")
        
        print("Computing normalization statistics from training set...")
        
        # Accumulate sums and squared sums
        sums = torch.zeros(6)  # 6 channels
        sums_sq = torch.zeros(6)
        count = 0
        
        # Iterate through all samples
        for idx in range(len(self)):
            _, target = self[idx]  # Get raw target [6, 15, 15]
            
            # Accumulate statistics per channel
            for channel_idx in range(6):
                channel_data = target[channel_idx].flatten()  # [225]
                sums[channel_idx] += channel_data.sum().item()
                sums_sq[channel_idx] += (channel_data ** 2).sum().item()
            
            count += target.numel() // 6  # Number of pixels per channel
            
            if (idx + 1) % 1000 == 0:
                print(f"  Processed {idx + 1}/{len(self)} samples...")
        
        # Compute mean and std
        self.out_mean = sums / count  # Shape [6]
        
        # Variance = E[X^2] - E[X]^2
        variance = (sums_sq / count) - (self.out_mean ** 2)
        self.out_std = torch.sqrt(variance)  # Shape [6]
        
        # Avoid division by zero (set min std to 1e-8)
        self.out_std = torch.clamp(self.out_std, min=1e-8)
        
        # Save to cache
        self.norm_cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'out_mean': self.out_mean,
            'out_std': self.out_std
        }, self.norm_cache_path)
        
        print(f"Normalization statistics computed and saved to {self.norm_cache_path}")
        print(f"  Mean: {self.out_mean}")
        print(f"  Std:  {self.out_std}")
    
    def get_normalization_stats(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get normalization statistics (mean, std) for use in training loop.
        
        Returns:
            (out_mean, out_std): Both tensors of shape [6]
        """
        if self.out_mean is None or self.out_std is None:
            raise ValueError(
                "Normalization statistics not available. "
                "For training dataset, call compute_normalization_stats() first. "
                "For val/test datasets, ensure norm_cache.pt exists."
            )
        return self.out_mean, self.out_std
    
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
            'H': (0.0, 18.0),
            'D0': (-2.0, 2.0),
            'D1': (-2.0, 2.0)
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
