"""Auto-generates COCO instance-segmentation annotations from existing masks, instead of
hand-drawing polygons in Roboflow.

Per the company meeting (2026-10-01): "don't need to manually annotate, just need to work
on how to automate the process." We already have real masks for the 10 pilot subjects --
manual traces (two independent labelers) and the atlas-registration pipeline's automated
output -- so this script traces each mask's exact pixel boundary (no simplification, no
lossy downsampling) into COCO polygon format, lined up with the PNG slices already in
roboflow_slices/ (from extract_slices_for_roboflow.py).

Produces three separate COCO JSON files (kept separate because they're different labelings,
not to be pooled blindly):
  - coco_manual_me.json     -- Tyler's manual traces, 9 subjects (excl. rat 2: atlas-derived, not a real trace)
  - coco_manual_other.json  -- the second labeler's manual traces, same 9 subjects
  - coco_atlas_auto.json    -- the rigid-registration pipeline's automated masks, all 10 subjects

A bilateral structure (like the hypothalamus, which often shows as two lobes in a coronal
slice) is encoded as ONE annotation per slice with a multi-part "segmentation" list (COCO's
standard way to represent one instance split across disconnected regions), not as two
separate instances.

Scope: only the 10 pilot subjects, because that's the only data we have real masks for.
Extending to the other 122 (via the atlas pipeline) is a separate, bigger step -- the rigid
recipe here was only verified against ground truth on the thick-slice protocol, and most of
the remaining subjects are the other (isotropic) protocol.
"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
from PIL import Image

ROOT = Path(__file__).parent
SLICES_DIR = ROOT / "roboflow_slices"
MANIFEST = SLICES_DIR / "manifest.csv"
CATEGORY_ID = 1
CATEGORY_NAME = "hypothalamus"

PILOT_SUBJECTS = list(range(1, 11))
EXCLUDE_FROM_MANUAL = {2}  # rat 2's "manual" file is itself atlas-derived -- not real ground truth


def load_manifest():
    import csv
    rows = {}
    with open(MANIFEST) as f:
        for row in csv.DictReader(f):
            rows.setdefault(row["subject"], []).append(row)
    return rows


def mask_to_annotations(mask_2d: np.ndarray, image_id: int, ann_id_start: int):
    """Traces every contour in a binary mask losslessly into one COCO annotation
    (multi-part segmentation), returns (annotation_dict_or_None, next_ann_id)."""
    mask_u8 = (mask_2d > 0).astype(np.uint8)
    if mask_u8.sum() == 0:
        return None, ann_id_start

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polygons = []
    total_area = 0.0
    x_min, y_min, x_max, y_max = None, None, None, None
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
        return None, ann_id_start

    ann = {
        "id": ann_id_start,
        "image_id": image_id,
        "category_id": CATEGORY_ID,
        "segmentation": polygons,
        "area": float(total_area),
        "bbox": [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)],
        "iscrowd": 0,
    }
    return ann, ann_id_start + 1


def mask_volume_for_subject(n: int, mask_path: Path, reference_t2_path: Path):
    """Loads a mask NRRD and resamples it onto the subject's own T2 grid (identity
    transform, nearest-neighbor) so it's pixel-for-pixel aligned with the exported PNGs,
    even though in practice the two already share the same grid for this dataset."""
    reference = sitk.ReadImage(str(reference_t2_path))
    mask = sitk.ReadImage(str(mask_path))
    mask_on_grid = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    return sitk.GetArrayFromImage(mask_on_grid) > 0  # (z, y, x)


def build_coco(mask_lookup, manifest_rows, subjects):
    images, annotations = [], []
    image_id = 1
    ann_id = 1

    for n in subjects:
        subject_id = f"sub-{n:03d}"
        if subject_id not in mask_lookup:
            print(f"  {subject_id}: no mask available, skipping")
            continue
        mask_path = mask_lookup[subject_id]
        t2_path = ROOT / "T2 for segmentation" / f"{subject_id}_ses-1_acq-RARE_T2w.nii"
        mask_vol = mask_volume_for_subject(n, mask_path, t2_path)

        rows = sorted(manifest_rows.get(subject_id, []), key=lambda r: int(r["slice_z"]))
        n_annotated = 0
        for row in rows:
            z = int(row["slice_z"])
            png_path = SLICES_DIR / row["output_file"]
            if not png_path.exists():
                continue
            with Image.open(png_path) as im:
                w, h = im.size

            images.append({
                "id": image_id,
                "file_name": row["output_file"],
                "width": w,
                "height": h,
            })

            if z < mask_vol.shape[0]:
                ann, ann_id = mask_to_annotations(mask_vol[z], image_id, ann_id)
                if ann is not None:
                    annotations.append(ann)
                    n_annotated += 1
            image_id += 1
        print(f"  {subject_id}: {len(rows)} images, {n_annotated} with a hypothalamus mask")

    return {
        "info": {"description": "Auto-generated from existing masks, not hand-annotated"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": [{"id": CATEGORY_ID, "name": CATEGORY_NAME, "supercategory": "anatomy"}],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="coco_annotations")
    args = parser.parse_args()

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(exist_ok=True)
    manifest_rows = load_manifest()

    me_lookup, other_lookup, atlas_lookup = {}, {}, {}
    for n in PILOT_SUBJECTS:
        subject_id = f"sub-{n:03d}"
        if n not in EXCLUDE_FROM_MANUAL:
            me = list((ROOT / "Manual Segments" / "Segmented nrrd" / "Me").glob(f"{n}_Segmentation*.nrrd"))
            other = list((ROOT / "Manual Segments" / "Segmented nrrd" / "Other").glob(f"{n}_Segmentation*.nrrd"))
            if me:
                me_lookup[subject_id] = me[0]
            if other:
                other_lookup[subject_id] = other[0]
        auto = ROOT / "auto_seg_out" / f"auto_seg_{n:03d}.nrrd"
        if auto.exists():
            atlas_lookup[subject_id] = auto

    for name, lookup in [("coco_manual_me", me_lookup), ("coco_manual_other", other_lookup), ("coco_atlas_auto", atlas_lookup)]:
        print(f"\nBuilding {name}.json ...")
        coco = build_coco(lookup, manifest_rows, PILOT_SUBJECTS)
        out_path = out_dir / f"{name}.json"
        with open(out_path, "w") as f:
            json.dump(coco, f)
        print(f"  wrote {out_path} ({len(coco['images'])} images, {len(coco['annotations'])} annotations)")


if __name__ == "__main__":
    main()
