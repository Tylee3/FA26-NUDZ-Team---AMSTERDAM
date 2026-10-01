"""Builds a Roboflow-ready upload bundle: all 132 slice images + one combined
_annotations.coco.json, in the folder layout Roboflow's COCO import expects.

Ground truth source: Tyler's manual traces ("Me") for the 9 usable pilot subjects
(rat 2 excluded -- its "manual" file is itself atlas-derived, not a real trace). The
other 122 subjects get zero annotations for now -- we don't have trustworthy labels
for them yet. This is deliberately ONE source of truth per image; the atlas-pipeline
output and the second labeler's ("Other") traces are kept out of this file so the
project doesn't end up with conflicting ground truth for the same structure.

Run this, then follow the upload steps it prints at the end.
"""

import csv
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(__file__).parent
SLICES_DIR = ROOT / "roboflow_slices"
MANIFEST = SLICES_DIR / "manifest.csv"
BUNDLE_DIR = ROOT / "roboflow_upload_bundle"
CATEGORY_ID = 1
CATEGORY_NAME = "hypothalamus"

PILOT_SUBJECTS = list(range(1, 11))
EXCLUDE = {2}  # atlas-derived, not a real manual trace


def mask_to_annotation(mask_2d, image_id, ann_id):
    mask_u8 = (mask_2d > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return None, ann_id
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons, total_area = [], 0.0
    x_min = y_min = x_max = y_max = None
    for c in contours:
        if len(c) < 3:
            continue
        pts = c.reshape(-1, 2).astype(float)
        polygons.append(pts.flatten().tolist())
        total_area += cv2.contourArea(c)
        cx0, cy0 = pts[:, 0].min(), pts[:, 1].min()
        cx1, cy1 = pts[:, 0].max(), pts[:, 1].max()
        x_min = cx0 if x_min is None else min(x_min, cx0)
        y_min = cy0 if y_min is None else min(y_min, cy0)
        x_max = cx1 if x_max is None else max(x_max, cx1)
        y_max = cy1 if y_max is None else max(y_max, cy1)
    if not polygons:
        return None, ann_id
    return {
        "id": ann_id, "image_id": image_id, "category_id": CATEGORY_ID,
        "segmentation": polygons, "area": float(total_area),
        "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
        "iscrowd": 0,
    }, ann_id + 1


def main():
    if BUNDLE_DIR.exists():
        shutil.rmtree(BUNDLE_DIR)
    BUNDLE_DIR.mkdir()

    manifest_rows = {}
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            manifest_rows.setdefault(row["subject"], []).append(row)

    mask_volumes = {}
    for n in PILOT_SUBJECTS:
        if n in EXCLUDE:
            continue
        subject_id = f"sub-{n:03d}"
        me_glob = list((ROOT / "Manual Segments" / "Segmented nrrd" / "Me").glob(f"{n}_Segmentation*.nrrd"))
        if not me_glob:
            continue
        t2_path = ROOT / "T2 for segmentation" / f"{subject_id}_ses-1_acq-RARE_T2w.nii"
        reference = sitk.ReadImage(str(t2_path))
        mask = sitk.ReadImage(str(me_glob[0]))
        mask_on_grid = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
        mask_volumes[subject_id] = sitk.GetArrayFromImage(mask_on_grid) > 0

    images, annotations = [], []
    image_id, ann_id = 1, 1
    total_annotated = 0

    for subject_id, rows in manifest_rows.items():
        for row in sorted(rows, key=lambda r: int(r["slice_z"])):
            png_path = SLICES_DIR / row["output_file"]
            if not png_path.exists():
                continue
            with Image.open(png_path) as im:
                w, h = im.size
            images.append({"id": image_id, "file_name": row["output_file"], "width": w, "height": h})
            shutil.copy(png_path, BUNDLE_DIR / row["output_file"])

            if subject_id in mask_volumes:
                z = int(row["slice_z"])
                vol = mask_volumes[subject_id]
                if z < vol.shape[0]:
                    ann, ann_id = mask_to_annotation(vol[z], image_id, ann_id)
                    if ann is not None:
                        annotations.append(ann)
                        total_annotated += 1
            image_id += 1

    coco = {
        "info": {"description": "Hypothalamus ground truth (manual traces, 9 pilot subjects) + unannotated remainder"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": "anatomy"}],
    }
    with open(BUNDLE_DIR / "_annotations.coco.json", "w") as f:
        json.dump(coco, f)

    zip_path = ROOT / "roboflow_upload_bundle.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in BUNDLE_DIR.iterdir():
            zf.write(p, p.name)

    print(f"{len(images)} images, {total_annotated} with a real annotation (manual traces, subjects 1,3-10)")
    print(f"Bundle folder: {BUNDLE_DIR}")
    print(f"Zip ready to upload: {zip_path}")


if __name__ == "__main__":
    main()
