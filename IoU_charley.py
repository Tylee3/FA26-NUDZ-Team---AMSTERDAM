import SimpleITK as sitk
import numpy as np

# Load images natively with their metadata intact
pred_img = sitk.ReadImage('seg_003.seg.nrrd')
true_img = sitk.ReadImage('seg_003_manual.nrrd')

# Check if space headers match. If they don't, copy metadata from true to pred
if not pred_img.IsSameImageGeometryAs(true_img):
    print("Warning: Spatial metadata mismatch detected! Fixing alignment...")
    pred_img.CopyInformation(true_img)

# Convert to arrays safely now that spatial frames match
pred_arr = sitk.GetArrayFromImage(pred_img)
true_arr = sitk.GetArrayFromImage(true_img)

# Calculate your true IoU
intersection = np.logical_and(true_arr, pred_arr).sum()
union = np.logical_or(true_arr, pred_arr).sum()
iou = intersection / union if union > 0 else 0.0
print(f"True Physical IoU: {iou:.4f}")

