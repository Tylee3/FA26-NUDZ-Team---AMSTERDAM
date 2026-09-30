import SimpleITK as sitk
import numpy as np

# 1. Define Windows file paths
gt_nrrd = "seg_003_manual.nrrd"
seg_nrrd = "seg_003.seg.nrrd"

gt_minc_out = "seg_003_manual.mnc"
seg_minc_out = "seg_003_.seg.mnc"

def pad_to_cube_and_save(input_path, output_path):
    image = sitk.ReadImage(input_path)
    old_size = image.GetSize()
    max_dim = max(old_size)
    
    # Calculate padding bounds
    pad_lower = []
    pad_upper = []
    for size in old_size:
        total_pad = max_dim - size
        pad_lower.append(total_pad // 2)
        pad_upper.append(total_pad - (total_pad // 2))
        
    # Apply zero padding to ensure an isotropic cubic matrix
    if max_dim != min(old_size):
        image = sitk.ConstantPad(image, pad_lower, pad_upper, 0.0)
        
    sitk.WriteImage(image, output_path)
    return image

print("Converting and padding volumes...")
padded_gt = pad_to_cube_and_save(gt_nrrd, gt_minc_out)
padded_seg = pad_to_cube_and_save(seg_nrrd, seg_minc_out)

# 2. Calculate the exact IoU
# Convert the SimpleITK images to NumPy arrays for easy boolean math
gt_array = sitk.GetArrayFromImage(padded_gt) > 0  # Convert to boolean mask
seg_array = sitk.GetArrayFromImage(padded_seg) > 0  # Convert to boolean mask

intersection = np.logical_and(gt_array, seg_array).sum()
union = np.logical_or(gt_array, seg_array).sum()

if union == 0:
    iou = 1.0 if intersection == 0 else 0.0
else:
    iou = intersection / union

print("\n--- Evaluation Results ---")
print(f"Validated MINC Matrix Dimensions: {padded_gt.GetSize()}")
print(f"Exact Intersection over Union (IoU): {iou:.4f}")
