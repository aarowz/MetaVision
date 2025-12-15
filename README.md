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
