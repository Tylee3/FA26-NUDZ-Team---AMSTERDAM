import nrrd
import numpy as np
import matplotlib.pyplot as plt

# 1. Load your two segmentation NRRD files
# Replace these strings with your actual file paths
gt_data, gt_header  = nrrd.read('man_ual.nrrd')
auto_data, auto_header = nrrd.read('auto_matic.nrrd')

# Convert to strict binary arrays (0 and 1)
gt_binary = (gt_data > 0).astype(np.uint8)
auto_binary = (auto_data > 0).astype(np.uint8)

# --- 3D METRICS CALCULATION ---
intersection = np.sum(gt_binary * auto_binary)
union = np.sum((gt_binary + auto_binary) > 0)
total_pixels = np.sum(gt_binary) + np.sum(auto_binary)

# Compute 3D IoU (Jaccard Index)
iou_score = intersection / union if union > 0 else 1.0

# Compute 3D Dice Coefficient (F1-Score)
dice_score = (2.0 * intersection) / total_pixels if total_pixels > 0 else 1.0

print("--- GLOBAL 3D METRICS ---")
print(f"Total 3D IoU Score:  {iou_score:.4f} ({iou_score * 100:.2f}%)")
print(f"Total 3D Dice Score: {dice_score:.4f} ({dice_score * 100:.2f}%)")

# 2. Automatically detect the correct slice orientation
sums_axis0 = np.sum(gt_binary, axis=(1, 2))
sums_axis1 = np.sum(gt_binary, axis=(0, 2))
sums_axis2 = np.sum(gt_binary, axis=(0, 1))

if np.max(sums_axis0) > np.max(sums_axis1) and np.max(sums_axis0) > np.max(sums_axis2):
    slice_axis = 0
    slice_index = np.argmax(np.sum(gt_binary * auto_binary, axis=(1, 2)) if np.any(gt_binary * auto_binary) else sums_axis0)
    gt_slice = gt_binary[slice_index, :, :]
    auto_slice = auto_binary[slice_index, :, :]
elif np.max(sums_axis1) > np.max(sums_axis0) and np.max(sums_axis1) > np.max(sums_axis2):
    slice_axis = 1
    slice_index = np.argmax(np.sum(gt_binary * auto_binary, axis=(0, 2)) if np.any(gt_binary * auto_binary) else sums_axis1)
    gt_slice = gt_binary[:, slice_index, :]
    auto_slice = auto_binary[:, slice_index, :]
else:
    slice_axis = 2
    slice_index = np.argmax(np.sum(gt_binary * auto_binary, axis=(0, 1)) if np.any(gt_binary * auto_binary) else sums_axis2)
    gt_slice = gt_binary[:, :, slice_index]
    auto_slice = auto_binary[:, :, slice_index]

# 3. Create RGB Map for Visualization
h, w = gt_slice.shape
display_rgb = np.zeros((h, w, 3), dtype=np.float32)
display_rgb[:, :, 0] = auto_slice  # Red = Automated
display_rgb[:, :, 1] = gt_slice    # Green = Manual

# Auto-zoom bounding box selection
rows, cols = np.where((gt_slice > 0) | (auto_slice > 0))
if len(rows) > 0:
    r_min, r_max = max(0, np.min(rows) - 15), min(h, np.max(rows) + 15)
    c_min, c_max = max(0, np.min(cols) - 15), min(w, np.max(cols) + 15)
else:
    r_min, r_max, c_min, c_max = 0, h, 0, w

# 4. Plotting
plt.figure(figsize=(9, 9))
plt.imshow(display_rgb[r_min:r_max, c_min:c_max])

# Add Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='green', label='Manual Only (FN)'),
    Patch(facecolor='red', label='Automated Only (FP)'),
    Patch(facecolor='yellow', label='Overlap / Agreement (TP)')
]
plt.legend(handles=legend_elements, loc='upper right', facecolor='white', framealpha=0.9)

# Include metrics directly in the image title
title_text = (
    f"Segmentation Overlap (Axis {slice_axis}, Slice {slice_index})\n"
    f"Global 3D IoU: {iou_score * 100:.2f}%  |  3D Dice: {dice_score * 100:.2f}%"
)
plt.title(title_text, fontsize=12, pad=15)
plt.axis('off')
plt.tight_layout()
plt.show()
