"""Builds the R-CNN train/test sets from the slice PNGs and existing masks.

train: the 53 thick-slice rats outside the pilot 10, labeled automatically by the rigid
       atlas pipeline (atlas_labels/). This is "the annotations are already there": the
       Waxholm atlas's label 48, carried onto each rat by registration.
test:  the 9 hand-traced pilot rats (1, 3-10), labeled with Tyler's manual traces ("Me"),
       with the second labeler's traces ("Other") kept alongside for a second score.
       None of these rats appear in training.

The 69 isotropic-protocol rats are left out: the atlas pipeline was never checked against
ground truth on that protocol, and their slices are stored rotated.

Writes rcnn_data/{train,test}.npz (image + mask arrays, read directly by the training
code with no format conversion) and rcnn_data/{train,test}_coco.json (the same labels as
COCO polygons, for anyone using another framework).
"""

import csv
import json

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .common import (ATLAS_TRAIN_DIR, DATA_DIR, MANUAL_DIR, PILOT_RATS, SLICES_DIR,
                     TEST_RATS, t2_path)
from generate_coco_annotations import mask_to_annotations


def mask_volume(mask_path, n):
    reference = sitk.ReadImage(str(t2_path(n)))
    mask = sitk.ReadImage(str(mask_path))
    on_grid = sitk.Resample(mask, reference, sitk.Transform(), sitk.sitkNearestNeighbor, 0, mask.GetPixelID())
    return sitk.GetArrayFromImage(on_grid) > 0


def manifest_by_subject():
    rows = {}
    with open(SLICES_DIR / "manifest.csv") as f:
        for r in csv.DictReader(f):
            rows.setdefault(int(r["subject"][4:]), []).append(r)
    return rows


def build(split, rats, mask_sources, manifest):
    images, subjects, zs, files = [], [], [], []
    masks = {name: [] for name in mask_sources}
    for n in rats:
        vols = {name: mask_volume(path_fn(n), n) for name, path_fn in mask_sources.items()}
        for r in sorted(manifest[n], key=lambda r: int(r["slice_z"])):
            z = int(r["slice_z"])
            images.append(np.array(Image.open(SLICES_DIR / r["output_file"])))
            for name in mask_sources:
                masks[name].append(vols[name][z])
            subjects.append(n)
            zs.append(z)
            files.append(r["output_file"])

    arrays = {
        "images": np.stack(images).astype(np.uint8),
        "subjects": np.array(subjects),
        "zs": np.array(zs),
        "files": np.array(files),
    }
    first = next(iter(mask_sources))
    arrays["masks"] = np.stack(masks[first])
    for name in list(mask_sources)[1:]:
        arrays[f"masks_{name}"] = np.stack(masks[name])
    np.savez_compressed(DATA_DIR / f"{split}.npz", **arrays)

    coco_images, coco_anns, ann_id = [], [], 1
    for i, (f, m) in enumerate(zip(files, arrays["masks"]), start=1):
        h, w = m.shape
        coco_images.append({"id": i, "file_name": f, "width": int(w), "height": int(h)})
        ann, ann_id = mask_to_annotations(m, i, ann_id)
        if ann is not None:
            coco_anns.append(ann)
    with open(DATA_DIR / f"{split}_coco.json", "w") as fh:
        json.dump({"images": coco_images, "annotations": coco_anns,
                   "categories": [{"id": 1, "name": "hypothalamus"}]}, fh)

    positives = int(arrays["masks"].reshape(len(files), -1).any(axis=1).sum())
    print(f"{split}: {len(rats)} rats, {len(files)} slices, {positives} with the structure")


def main():
    DATA_DIR.mkdir(exist_ok=True)
    manifest = manifest_by_subject()
    thick = sorted(n for n, rows in manifest.items() if rows[0]["protocol"] == "thick")
    train_rats = [n for n in thick if n not in PILOT_RATS]

    build("train", train_rats, {"atlas": lambda n: ATLAS_TRAIN_DIR / f"auto_seg_{n:03d}.nrrd"}, manifest)
    build("test", TEST_RATS, {
        "me": lambda n: next((MANUAL_DIR / "Me").glob(f"{n}_Segmentation*.nrrd")),
        "other": lambda n: next((MANUAL_DIR / "Other").glob(f"{n}_Segmentation*.nrrd")),
    }, manifest)


if __name__ == "__main__":
    main()
