import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

"""
Data Exploration Script

Loads and visualizes metasurface geometry (R, H, dx, dy) and EM field data (Ex, Ey, Ez)
from .mat files to understand data structure before model training.

Usage:
    python explore_data.py              # Process all 11 files
    python explore_data.py 0 2 5         # Process files 0, 2, and 5
    python explore_data.py 5             # Process only file 5
"""

# Get script directory and project root (works regardless of where script is run from)
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

def visualize_file(file_idx):
    """Load and visualize a single .mat file"""
    file_path = PROJECT_ROOT / 'data' / 'raw' / f'data3d_{file_idx}.mat'
    
    try:
        data = sio.loadmat(str(file_path))
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, skipping...")
        return
    
    # Create output directories
    input_dir = PROJECT_ROOT / 'results' / 'figures' / 'exploration' / 'input'
    output_dir = PROJECT_ROOT / 'results' / 'figures' / 'exploration' / 'output'
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Visualize input geometry
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # R (Radius)
    r_data = data['R']
    im1 = axes[0, 0].imshow(r_data, cmap='viridis')
    axes[0, 0].set_title(f'R (Radius)\nRange: [{r_data.min():.2f}, {r_data.max():.2f}]')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)
    
    # H (Height)
    h_data = data['H']
    im2 = axes[0, 1].imshow(h_data, cmap='viridis')
    axes[0, 1].set_title(f'H (Height)\nRange: [{h_data.min():.2f}, {h_data.max():.2f}]')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # dx (X-displacement)
    dx_data = data['D'][0]
    im3 = axes[1, 0].imshow(dx_data, cmap='viridis')
    axes[1, 0].set_title(f'dx (X-displacement)\nRange: [{dx_data.min():.2f}, {dx_data.max():.2f}]')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)
    
    # dy (Y-displacement)
    dy_data = data['D'][1]
    im4 = axes[1, 1].imshow(dy_data, cmap='viridis')
    axes[1, 1].set_title(f'dy (Y-displacement)\nRange: [{dy_data.min():.2f}, {dy_data.max():.2f}]')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    input_path = input_dir / f'geometry_{file_idx}.png'
    plt.savefig(input_path, dpi=150)
    plt.close()
    print(f"Saved input geometry: {input_path}")
    
    # Visualize output EM fields (downsample for visualization)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Take magnitude of complex fields and downsample
    for i, field_name in enumerate(['Ex', 'Ey', 'Ez']):
        field = data[field_name]
        magnitude = np.abs(field)
        # Downsample for visualization
        downsampled = magnitude[::24, ::24]  # Roughly match input size
        im = axes[i].imshow(downsampled, cmap='hot')
        axes[i].set_title(f'{field_name} magnitude (downsampled)\nRange: [{downsampled.min():.3f}, {downsampled.max():.3f}]')
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_path = output_dir / f'fields_{file_idx}.png'
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved output fields: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize metasurface geometry and EM field data'
    )
    parser.add_argument(
        'files',
        nargs='*',
        type=int,
        help='File indices to process (0-10). If none specified, processes all 11 files.'
    )
    
    args = parser.parse_args()
    
    # Determine which files to process
    if args.files:
        file_indices = args.files
        print(f"Processing specified files: {file_indices}")
    else:
        file_indices = list(range(11))  # All files: 0-10
        print("Processing all 11 files...")
    
    # Process each file
    for file_idx in file_indices:
        if file_idx < 0 or file_idx > 10:
            print(f"Warning: File index {file_idx} out of range (0-10), skipping...")
            continue
        
        print(f"\nProcessing file {file_idx}...")
        visualize_file(file_idx)
    
    print("\n" + "="*60)
    print("Data summary:")
    print(f"Input resolution: 120x120")
    print(f"Output resolution: 2883x2883 ({2883/120:.1f}x larger)")
    print(f"Output is complex-valued")
    print("="*60)


if __name__ == '__main__':
    main()
