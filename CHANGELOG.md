# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### TODO

- Design ViT model architecture
- Implement training loop
- Add evaluation metrics
- Create inference script
- Get real normalization parameters from `analyze_ranges.py` and update dataset defaults

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
