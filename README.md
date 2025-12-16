# MetaVision: Computer Vision Transformer for Electromagnetic Field Prediction

Predicting electromagnetic field distributions from metasurface geometry using Vision Transformers.

## 🎯 Project Goals
- Develop ViT-based model for EM field prediction
- Achieve competitive accuracy with FDTD simulation
- Reduce computation time from hours to seconds

## 🚀 Status
🚧 In Progress (Dec 2025)

## 📊 Results
Coming soon...

## 📦 Data Access

The dataset (11 .mat files) is not included in this repository due to licensing/usage restrictions.

To request access to the dataset, please contact the repository maintainer with:
- Your name and affiliation
- Intended use case
- Brief description of your research/project

The dataset contains:
- 11 metasurface geometry configurations
- Corresponding FDTD-simulated EM field distributions
- Format: MATLAB .mat files (~200MB each)
- Input: Geometry parameters (R, H, D) - 4 channels, 120×120 resolution
- Output: EM field components (Ex, Ey, Ez) - Complex-valued, 2883×2883 resolution

## 🛠️ Tech Stack
- PyTorch
- Vision Transformer (ViT)
- CUDA
- NumPy/SciPy

## 📁 Project Structure
```
MetaVision/
├── data/
│   ├── raw/              # .mat files (not in repo - request access)
│   └── processed/        # Processed numpy arrays
├── src/                  # Source code (data loader, model, training)
├── notebooks/            # Exploration and analysis notebooks
├── results/
│   ├── figures/          # Visualizations (exploration, training, predictions)
│   ├── models/           # Saved model checkpoints
│   └── logs/             # Training logs
└── CHANGELOG.md          # Development history
```

## 📝 License
MIT

## 🚀 Getting Started

### Data Exploration

After obtaining the dataset, you can generate visualizations:

```bash
# Generate visualizations for all 11 files
python3 notebooks/explore_data.py

# Generate visualizations for specific files
python3 notebooks/explore_data.py 0 2 5
```

Visualizations are saved to `results/figures/exploration/` with:
- Input geometry: `input/geometry_{0-10}.png`
- Output EM fields: `output/fields_{0-10}.png`

### Data Loading

The project includes a complete PyTorch Dataset implementation for loading and preprocessing the metasurface data:

```python
from pathlib import Path
from src.data_loader import MetasurfaceDataset, create_train_val_test_splits
from torch.utils.data import DataLoader

# Create train/val/test splits
train_idx, val_idx, test_idx = create_train_val_test_splits(total_files=11)

# Create dataset
train_dataset = MetasurfaceDataset(
    data_dir=Path('data/raw'),
    file_indices=train_idx,
    split='train',
    normalize_input=True,
    normalize_output=True
)

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
```

### Testing

Run the comprehensive test suite to verify data loading:

```bash
python3 notebooks/test_data_loader.py
```

This will verify:
- All 11 files can be loaded successfully
- Input/output shapes are correct
- Normalization works as expected
- DataLoader integration functions properly
