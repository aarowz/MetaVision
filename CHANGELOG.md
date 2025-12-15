# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### TODO

- Implement data pipeline (Dataset class, preprocessing)
- Design ViT model architecture
- Implement training loop
- Add evaluation metrics
- Create inference script

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
