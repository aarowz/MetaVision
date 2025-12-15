import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load first file
data = sio.loadmat('../data/raw/data3d_0.mat')

# Visualize input geometry
fig, axes = plt.subplots(2, 2, figsize=(10, 10))

axes[0, 0].imshow(data['R'], cmap='viridis')
axes[0, 0].set_title('R (Radius)')
axes[0, 0].axis('off')

axes[0, 1].imshow(data['H'], cmap='viridis')
axes[0, 1].set_title('H (Height)')
axes[0, 1].axis('off')

axes[1, 0].imshow(data['D'][0], cmap='viridis')
axes[1, 0].set_title('D[0]')
axes[1, 0].axis('off')

axes[1, 1].imshow(data['D'][1], cmap='viridis')
axes[1, 1].set_title('D[1]')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('../results/figures/input_geometry.png', dpi=150)
print("Saved input geometry visualization")

# Visualize output EM fields (downsample for visualization)
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Take magnitude of complex fields and downsample
for i, field_name in enumerate(['Ex', 'Ey', 'Ez']):
    field = data[field_name]
    magnitude = np.abs(field)
    # Downsample for visualization
    downsampled = magnitude[::24, ::24]  # Roughly match input size
    axes[i].imshow(downsampled, cmap='hot')
    axes[i].set_title(f'{field_name} magnitude (downsampled)')
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('../results/figures/output_fields.png', dpi=150)
print("Saved output EM fields visualization")

print("\nData summary:")
print(f"Input resolution: 120x120")
print(f"Output resolution: 2883x2883 ({2883/120:.1f}x larger)")
print(f"Output is complex-valued")

