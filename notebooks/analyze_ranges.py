import scipy.io as sio
import numpy as np
from pathlib import Path

"""
Data Range Analysis Script

Analyzes min/max values across all .mat files to inform normalization strategies
for the Dataset class. Computes statistics for:
- Input geometry: R, H, D[0], D[1]
- Output EM fields: Ex, Ey, Ez (magnitude, real, imaginary components)
"""

# Get script directory and project root
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

# Collect stats across all files
input_stats = {'R': [], 'H': [], 'D0': [], 'D1': []}
output_stats = {'Ex': [], 'Ey': [], 'Ez': []}

print("Analyzing data ranges across all 11 files...\n")

for i in range(11):
    file_path = PROJECT_ROOT / 'data' / 'raw' / f'data3d_{i}.mat'
    
    try:
        data = sio.loadmat(str(file_path))
    except FileNotFoundError:
        print(f"Warning: {file_path} not found, skipping...")
        continue
    
    # Input stats
    input_stats['R'].append([data['R'].min(), data['R'].max()])
    input_stats['H'].append([data['H'].min(), data['H'].max()])
    input_stats['D0'].append([data['D'][0].min(), data['D'][0].max()])
    input_stats['D1'].append([data['D'][1].min(), data['D'][1].max()])
    
    # Output stats (complex fields)
    for field in ['Ex', 'Ey', 'Ez']:
        field_data = data[field]
        magnitude = np.abs(field_data)
        real = np.real(field_data)
        imag = np.imag(field_data)
        output_stats[field].append({
            'mag': [magnitude.min(), magnitude.max()],
            'real': [real.min(), real.max()],
            'imag': [imag.min(), imag.max()]
        })

# Print summary
print("="*60)
print("INPUT STATISTICS (Geometry Parameters)")
print("="*60)
for key, values in input_stats.items():
    if values:  # Check if list is not empty
        mins = [v[0] for v in values]
        maxs = [v[1] for v in values]
        print(f"{key:3s}: min={min(mins):8.3f}, max={max(maxs):8.3f}, range={max(maxs)-min(mins):8.3f}")

print("\n" + "="*60)
print("OUTPUT STATISTICS (EM Fields)")
print("="*60)
for field, stats_list in output_stats.items():
    if stats_list:  # Check if list is not empty
        mag_mins = [s['mag'][0] for s in stats_list]
        mag_maxs = [s['mag'][1] for s in stats_list]
        real_mins = [s['real'][0] for s in stats_list]
        real_maxs = [s['real'][1] for s in stats_list]
        imag_mins = [s['imag'][0] for s in stats_list]
        imag_maxs = [s['imag'][1] for s in stats_list]
        
        print(f"\n{field}:")
        print(f"  Magnitude:  [{min(mag_mins):8.3f}, {max(mag_maxs):8.3f}]")
        print(f"  Real:       [{min(real_mins):8.3f}, {max(real_maxs):8.3f}]")
        print(f"  Imaginary:  [{min(imag_mins):8.3f}, {max(imag_maxs):8.3f}]")

print("\n" + "="*60)
print("NORMALIZATION RECOMMENDATIONS")
print("="*60)
if input_stats['R']:  # Check if we have data
    print("\nInput normalization (per channel):")
    for key, values in input_stats.items():
        if values:
            mins = [v[0] for v in values]
            maxs = [v[1] for v in values]
            global_min = min(mins)
            global_max = max(maxs)
            print(f"  {key:3s}: Normalize to [0,1] using min={global_min:.3f}, max={global_max:.3f}")

if output_stats['Ex']:  # Check if we have data
    print("\nOutput normalization (for Real+Imaginary representation):")
    for field in ['Ex', 'Ey', 'Ez']:
        stats_list = output_stats[field]
        if stats_list:
            real_mins = [s['real'][0] for s in stats_list]
            real_maxs = [s['real'][1] for s in stats_list]
            imag_mins = [s['imag'][0] for s in stats_list]
            imag_maxs = [s['imag'][1] for s in stats_list]
            
            real_min = min(real_mins)
            real_max = max(real_maxs)
            imag_min = min(imag_mins)
            imag_max = max(imag_maxs)
            
            # Use the larger range for normalization
            norm_max = max(abs(real_min), abs(real_max), abs(imag_min), abs(imag_max))
            
            print(f"  {field}: Normalize by max absolute value = {norm_max:.3f}")

print("\n" + "="*60)
