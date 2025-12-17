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
- Input: Geometry parameters (R, H, D) - 4 channels, 120×120 resolution (extracted as 15×15 blocks)
- Output: EM field components (Ex, Ey, Ez) - Complex-valued, 2883×2883 resolution (downsampled to 15×15)

## 🛠️ Tech Stack

- PyTorch
- Vision Transformer (ViT)
- CUDA
- NumPy/SciPy

## 🏗️ Model Architecture

**Vision Transformer Encoder** for image-to-image regression (simplified architecture).

### Encoder (ViT)

- **Input**: `[B, 4, 15, 15]` (R, H, D[0], D[1] geometry channels - 15×15 blocks)
- **Patch Embedding**: Conv2d(4→384, kernel=1, stride=1) → 15×15 = 225 patches (pixel-wise projection)
- **Positional Encoding**: Learnable embeddings (no CLS token - spatial task)
- **Transformer Blocks**: 6 layers, 6 heads, 384 dim, GELU activation, pre-norm, stochastic depth (0.1)
- **Output**: `[B, 225, 384]` patch tokens

### Output Head

- **Reshape**: `[B, 225, 384]` → `[B, 384, 15, 15]` (spatial grid reconstruction)
- **Direct Projection**: Conv2d(384→6, kernel=1)
- **Activation**: None (unbounded regression output)
- **Output**: `[B, 6, 15, 15]` (Ex/Ey/Ez real+imaginary components)

### Design Decisions

- **No CLS token**: Image-to-image task requires spatial preservation
- **Patch size 1**: Pixel-wise projection for 15×15 input (15×15 = 225 patches)
- **No decoder**: Direct projection from ViT features to output (simplified architecture)
- **ViT-Small**: 384 dim, 6 layers (optimized for dataset)
- **No pre-training**: 4-channel input incompatible with ImageNet (3-channel)
- **Z-score normalization**: Applied in loss function (matches GAT's approach)

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

## 🔧 Setup

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository** (if you haven't already):

   ```bash
   git clone https://github.com/aarowz/MetaVision.git
   cd MetaVision
   ```

2. **Create a virtual environment** (recommended):

   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:

   ```bash
   # On macOS/Linux:
   source venv/bin/activate

   # On Windows:
   venv\Scripts\activate
   ```

4. **Install dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

5. **Verify installation**:
   ```bash
   python3 notebooks/test_data_loader.py
   ```

### Deactivating the Virtual Environment

When you're done working, deactivate the virtual environment:

```bash
deactivate
```

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

**Data Strategy (matching GAT approach):**
- Extracts 15×15 blocks from larger 120×120 geometry images
- 2000 random blocks per file (total ~22k training samples)
- Input normalization: Min-max scaling per channel
- Output normalization: Z-score normalization (computed from training set, applied in loss)

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
    input_norm_params=config['data']['input_norm_params'],
    num_blocks_per_file=2000,  # Extract 2000 blocks per file
    block_size=15,  # 15×15 blocks
    compute_norm=True,  # Compute normalization stats
    norm_cache_path=Path('data/processed/norm_cache.pt'),
    seed=42
)

# Compute normalization statistics (one-time, cached)
train_dataset.compute_normalization_stats()
out_mean, out_std = train_dataset.get_normalization_stats()

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
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

### Configuration

The project uses `config.yaml` for all hyperparameters. Key settings:

- **Model**: ViT-small (384 dim, 6 layers, patch_size=1, img_size=15)
- **Training**: Batch size 2, 50 epochs, aggressive regularization
- **Data Strategy**: 15×15 blocks, 2000 blocks per file (~22k training samples)
- **Normalization**: Z-score normalization computed from training set, applied in loss function
- **Data Augmentation**: Enabled by default (flips, rotations) for better generalization

Edit `config.yaml` to experiment with different hyperparameters.

### Data Augmentation

The project includes physics-preserving data augmentation (`src/augmentation.py`):

- Horizontal/Vertical flips
- 90°/180°/270° rotations
- Preserves metasurface geometry physics
- Can expand 8 training samples → 64 augmented variants

Configure in `config.yaml` under `data_augmentation` section.

### Training

Train the model using the training script:

```bash
# Activate virtual environment first
source venv/bin/activate

# Run training
python3 train.py
```

The training script will:
- Load configuration from `config.yaml`
- Create train/val/test splits
- Extract 15×15 blocks from geometry images (2000 blocks per file)
- Compute normalization statistics from training set (one-time, cached)
- Apply z-score normalization in loss function (matches GAT approach)
- Apply data augmentation during training
- Save checkpoints to `checkpoints/`
- Log metrics to TensorBoard (`runs/`)
- Implement early stopping if validation loss doesn't improve

**Note**: First run will take ~5-10 minutes to compute normalization statistics. Subsequent runs are instant (uses cache).

Monitor training progress:
```bash
# In a separate terminal
tensorboard --logdir results/logs/tensorboard
```

Then open `http://localhost:6006` in your browser.

**Configuration**: All training hyperparameters are in `config.yaml`:
- Model architecture (ViT encoder, CNN decoder)
- Training parameters (batch size, epochs, learning rate)
- Optimizer and scheduler settings
- Early stopping configuration
- Data augmentation settings
