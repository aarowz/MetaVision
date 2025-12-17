# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2025-12-16] - Data Strategy and Architecture Refinement

### Added

- **Data loader rewrite** (`src/data_loader.py`):
  - Block-based data extraction: 15×15 blocks from 120×120 geometry images
  - 2000 random blocks per file (total ~22k training samples, matching GAT)
  - Z-score normalization: Computed from training set, applied in loss function
  - Normalization stats caching: `data/processed/norm_cache.pt` for fast subsequent runs
  - Reproducible block extraction using `np.random.RandomState` per file
- **Training script updates** (`train.py`):
  - Normalization stats computation integrated into training pipeline
  - Z-score normalization applied in loss function (matches GAT's approach)
  - Stats computed once, cached for future runs
  - Proper ordering: compute stats before creating validation dataset

### Changed

- **Model architecture simplified** (`src/model.py`):
  - Removed decoder upsampling stages (no longer needed)
  - Architecture: PatchEmbed → ViT → Reshape → Output Head
  - Input/output: `[B, 4, 15, 15]` → `[B, 6, 15, 15]` (15×15 blocks)
  - Patch size: 8 → 1 (pixel-wise projection for 15×15 input)
  - Total parameters: ~10.7M (vs GAT's ~13M)
- **Config updates** (`config.yaml`):
  - `img_size`: 120 → 15
  - `patch_size`: 8 → 1
  - Decoder section marked as unused (kept for compatibility)
- **Data normalization strategy**:
  - Input: Min-max scaling per channel (unchanged)
  - Output: Max-abs → Z-score normalization (computed from training set)
  - Normalization applied in loss function, not in data loader

### Fixed

- Dataset creation order: Normalization stats computed before validation dataset creation
- Path consistency: Single `norm_cache_path` used throughout training script

### Design Decisions

- **15×15 blocks**: Matches GAT's data strategy for fair comparison
- **Z-score normalization**: Applied in loss function (raw targets from dataset)
- **No decoder**: Simplified architecture, direct projection sufficient for 15×15 output
- **Reproducible sampling**: Uses `RandomState` for deterministic block extraction

## [2025-12-16] - Model Implementation and Training Script Complete

### Added

- Model architecture implementation (`src/model.py`):
  - Custom ViT encoder: 4-channel patch embedding, 6 transformer layers (384 dim, 6 heads)
  - CNN decoder: 3-stage upsampling (15→30→60→120) with channel progression [256,128,64,32]
  - Output head: 6-channel projection (Ex/Ey/Ez real+imag) with no activation
  - Architecture optimized for image-to-image regression (no CLS token, spatial preservation)
  - Total parameters: ~11.5M
- Training script (`train.py`):
  - Complete training pipeline with config-driven setup
  - Data loading with augmentation (training only)
  - Optimizer support (AdamW/Adam/SGD)
  - Learning rate schedulers (Cosine/Step/Plateau)
  - Loss functions (MSE/MAE/Huber/Combined)
  - Checkpointing (best model, latest, periodic)
  - Early stopping with configurable patience
  - TensorBoard logging
  - Mixed precision training support (AMP)
  - Resume training from checkpoint
- Model test script (`notebooks/test_model.py`):
  - Forward pass verification
  - Shape validation for different batch sizes
  - Parameter counting
- Model architecture documentation in README

### Changed

- Config: Updated output activation from "tanh" to "none" (unbounded regression)
- Config: Fixed scheduler T_max from 200 steps to 50 epochs (matches epoch-based stepping)

### Design Decisions

- **No CLS token**: Excluded for spatial image-to-image task (225 patch tokens only)
- **Learnable positional encoding**: Standard for custom ViT without pre-training
- **Patch size 8**: Divides evenly into 120×120 input (15×15 = 225 patches)
- **ViT-Small**: 384 dim, 6 layers optimized for small dataset (~8 training samples)
- **No skip connections**: Baseline architecture (can add later if needed)
- **Unbounded output**: No final activation - normalization handles EM field scaling

### TODO

- Run training and evaluate results
- Add evaluation metrics module (`src/metrics.py`)
- Create inference script (`inference.py`)
- Get real normalization parameters from `analyze_ranges.py` and update dataset defaults

## [2025-12-15] - Configuration and Augmentation Complete

### Added

- Comprehensive `config.yaml` with all hyperparameters:
  - Data configuration (paths, normalization, splits)
  - Model architecture (ViT-small: 384 dim, 6 layers, patch_size=8)
  - Training hyperparameters (optimized for small dataset)
  - System configuration (device, paths, reproducibility)
  - Inference and logging settings
- Data augmentation module (`src/augmentation.py`):
  - Physics-preserving geometric augmentations (flips, rotations)
  - Can 8x training data: 8 samples → 64 augmented samples
  - Critical for small dataset generalization
- Data augmentation configuration in `config.yaml`

### Changed

- Model architecture optimized for small dataset:
  - ViT-base → ViT-small (384 dim, 6 layers vs 768 dim, 12 layers)
  - Patch size: 16 → 8 (divides evenly into 120×120 input)
  - Pre-trained weights: disabled (4-channel input incompatible with ImageNet)
  - Increased dropout (0.0 → 0.1) for regularization
- Training parameters optimized:
  - Batch size: 4 → 2 (better for ~8 training samples)
  - Epochs: 100 → 50 (reduced for small dataset)
  - Weight decay: 0.01 → 0.05 (increased regularization)
  - Early stopping patience: 15 → 10
  - Scheduler T_max: calculated based on training steps (200)

### Fixed

- Patch size now divides evenly: 120 ÷ 8 = 15 patches per side
- Scheduler T_max now uses training steps instead of epochs
- All hyperparameters validated against actual dataset size

### Notes

- Normalization parameters still use placeholders - need to run `analyze_ranges.py`
- Augmentation is critical for this small dataset - enables better generalization

## [2025-12-14] - Project Setup Complete

### Added

- `requirements.txt` with all project dependencies:
  - PyTorch and torchvision for deep learning
  - timm for Vision Transformer models
  - NumPy, SciPy, Matplotlib (already in use)
  - PyYAML for configuration management
  - TensorBoard for training logs
  - tqdm for progress bars
- Virtual environment setup instructions in README
- Installation and setup documentation

### Changed

- Updated README with comprehensive setup instructions
- Added virtual environment recommendations for reproducibility

## [2025-12-14] - Data Loader Complete

### Added

- Complete PyTorch Dataset implementation (`src/data_loader.py`) with:
  - `MetasurfaceDataset` class for loading and preprocessing .mat files
  - Input preprocessing: Stacks R, H, D[0], D[1] into 4-channel tensor `[4, 120, 120]`
  - Output preprocessing: Converts complex Ex, Ey, Ez to 6-channel tensor `[6, H, W]` (real+imag)
  - Downsampling: Bilinear interpolation from 2883×2883 to configurable resolution (default: 120×120)
  - Normalization: Min-max for inputs `[0, 1]`, max-abs for outputs
  - Train/val/test split helper function (`create_train_val_test_splits`)
- Comprehensive test suite (`notebooks/test_data_loader.py`) with 8 tests:
  - Basic loading and shape verification
  - Normalization functionality
  - Different output resolutions
  - Multiple file loading
  - Train/val/test splits
  - PyTorch DataLoader integration
  - Custom normalization parameters
  - All 11 files loadable verification
- Data range analysis script (`notebooks/analyze_ranges.py`) for computing normalization parameters

### Changed

- Updated CHANGELOG TODO: Removed "Implement data pipeline" (completed)

### Fixed

- All 11 data files verified loadable and have correct shapes
- Path resolution in test script works from any directory

### Notes

- Normalization parameters currently use placeholder defaults - should run `analyze_ranges.py` to get actual values
- Default output resolution matches input (120×120) but can be configured

## [2025-12-14] - Data Exploration Complete

### Added

- Enhanced exploration script with CLI support for processing specific files or all files
- Colorbars and value ranges in visualizations for better data interpretation
- Path resolution fixes to work from any directory
- Generated visualizations for all 11 datasets (input geometry and output EM fields)

### Changed

- Updated `.gitignore` to exclude generated PNG files (users can regenerate)
- Improved visualization clarity with colorbars showing value ranges
- Script now processes all 11 files by default with option to specify individual files

### Fixed

- Path resolution issues when running script from different directories

## [2025-12-14] - Initial Setup

### Added

- Project structure with organized directories (`data/`, `src/`, `notebooks/`, `results/`)
- 11 .mat data files containing metasurface geometry and EM field data
- Data exploration script (`notebooks/explore_data.py`) for visualizing input/output data
- Results directory structure with subfolders for exploration, training, and prediction figures
- CHANGELOG.md for tracking project development

### Changed

- Organized figure outputs into `exploration/input/` and `exploration/output/` folders for better documentation

### Discovered

- **Input format**: Geometry parameters (R, H, D) as 4-channel 120×120 images
- **Output format**: EM field components (Ex, Ey, Ez) as complex-valued 2883×2883 arrays
- **Resolution mismatch**: Output resolution is ~24× larger than input (2883 vs 120)
- All 11 .mat files have consistent structure
