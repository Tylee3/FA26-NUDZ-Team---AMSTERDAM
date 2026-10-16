"""Tests the trained Mask R-CNN models on public, independently labeled rat brain MRI: the
Waxholm template (ex vivo) and the SIGMA in vivo T2 template, each with its own label 48
("Hypothalamic region") as ground truth. Answers NUDZ's ask (2026-10-15) to test on a
publicly available dataset.

The models were trained on CAMRI-format slices, so each public brain is first resampled
into the geometry of a real CAMRI scan (256x256x12, 0.1x0.1x1.0 mm, CAMRI's orientation)
by aligning it to that rat with the usual rigid registration. Image and label move together,
so where it lands doesn't matter. Three reference rats give three views of each brain.

Limits worth stating with any result:
  - Resampling samples the template at slice centers; it doesn't reproduce a real 1 mm
    slice's blur, so these images are slightly sharper than CAMRI's.
  - The Waxholm template is ex vivo (different contrast); SIGMA's region 48 is drawn
    smaller than Waxholm's, so a correct model can still score lower against it.
  - One brain each: a sanity check, not a full test.

Usage:  python3 public_atlas_test.py
Writes public_atlas_test/results.csv, RESULTS.md and overlays.png.
"""

import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from PIL import Image, ImageDraw

from atlas_registration_pipeline import ATLAS_LABELS, ATLAS_MRI, register_rigid
from rcnn import models
from rcnn.common import image_tensor, pick_device

ROOT = Path(__file__).parent
OUT = ROOT / "public_atlas_test"
BRAINS = {
    "Waxholm (ex vivo)": (ATLAS_MRI, ATLAS_LABELS),
    "SIGMA (in vivo T2)": (ROOT / "sigma" / "SIGMA_InVivo_Anatomical_Brain_template.nii.gz",
                           ROOT / "sigma" / "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz"),
}
REFERENCE_RATS = [3, 6, 9]
RUNS = {"Mask R-CNN": "mask_rcnn", "Mask R-CNN + augmentation": "mask_rcnn_augment"}
SCORE_THRESHOLD = 0.5


def normalize_volume(arr):
    nonzero = arr[arr > 0]
    lo, hi = np.percentile(nonzero, [1, 99]) if nonzero.size else (0, 1)
    hi = max(hi, lo + 1)
    return ((np.clip(arr, lo, hi) - lo) / (hi - lo) * 255).astype(np.uint8)


def virtual_scan(template_path, label_path, rat):
    scan = sitk.ReadImage(str(ROOT / "T2 for segmentation" / f"sub-{rat:03d}_ses-1_acq-RARE_T2w.nii"))
    template = sitk.Cast(sitk.ReadImage(str(template_path)), sitk.sitkFloat32)
    labels = sitk.ReadImage(str(label_path))
    t = register_rigid(sitk.Cast(scan, sitk.sitkFloat32), template)[0]
    image = sitk.GetArrayFromImage(sitk.Resample(template, scan, t, sitk.sitkLinear, 0.0))
    label = np.rint(sitk.GetArrayFromImage(sitk.Resample(labels, scan, t, sitk.sitkNearestNeighbor, 0))) == 48
    return normalize_volume(image), label


@torch.no_grad()
def predict(model, device, volume):
    out = np.zeros(volume.shape, dtype=bool)
    for z in range(volume.shape[0]):
        det = model([image_tensor(volume[z]).to(device)])[0]
        if len(det["scores"]) and det["scores"].max() >= SCORE_THRESHOLD:
            k = int(det["scores"].argmax())
            out[z] = det["masks"][k, 0].cpu().numpy() >= 0.5
    return out


def overlap(pred, truth):
    inter, union, total = (pred & truth).sum(), (pred | truth).sum(), pred.sum() + truth.sum()
    return (2 * inter / total if total else 1.0), (inter / union if union else 1.0)


def tile(volume, pred, truth, label):
    z = int(np.argmax(truth.reshape(len(truth), -1).sum(1)))
    rgb = np.stack([volume[z]] * 3, -1).astype(np.uint8)
    p, t = pred[z], truth[z]
    rgb[t & ~p] = [0, 255, 0]
    rgb[p & ~t] = [255, 0, 0]
    rgb[p & t] = [255, 255, 0]
    ys, xs = np.where(t | p)
    cy, cx = int(ys.mean()), int(xs.mean())
    crop = Image.fromarray(rgb[max(cy - 40, 0):cy + 40, max(cx - 40, 0):cx + 40]).resize((200, 200), Image.NEAREST)
    canvas = Image.new("RGB", (200, 218), (25, 25, 25))
    ImageDraw.Draw(canvas).text((4, 3), label, fill=(230, 230, 230))
    canvas.paste(crop, (0, 18))
    return canvas


def main():
    OUT.mkdir(exist_ok=True)
    device = pick_device()
    loaded = {}
    for name, run in RUNS.items():
        m = models.build("mask_rcnn")
        m.load_state_dict(torch.load(ROOT / "rcnn_out" / run / "model.pt", map_location="cpu"))
        loaded[name] = m.to(device).eval()

    rows, tiles = [], []
    for brain, (template, labels) in BRAINS.items():
        for rat in REFERENCE_RATS:
            volume, truth = virtual_scan(template, labels, rat)
            for name, model in loaded.items():
                pred = predict(model, device, volume)
                dice, iou = overlap(pred, truth)
                rows.append({"brain": brain, "reference_rat": rat, "model": name,
                             "dice": round(float(dice), 3), "iou": round(float(iou), 3),
                             "slices_with_structure": int(truth.reshape(len(truth), -1).any(1).sum()),
                             "slices_detected": int(pred.reshape(len(pred), -1).any(1).sum())})
                print(rows[-1], flush=True)
                if rat == REFERENCE_RATS[0]:
                    tiles.append(tile(volume, pred, truth, f"{brain.split()[0]} - {name.replace('Mask R-CNN', 'MRCNN')}"))

    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    grid = Image.new("RGB", (200 * len(RUNS), 218 * len(BRAINS)), (25, 25, 25))
    for k, t in enumerate(tiles):
        grid.paste(t, ((k % len(RUNS)) * 200, (k // len(RUNS)) * 218))
    grid.save(OUT / "overlays.png")

    lines = ["# Public-data test", "",
             "Trained Mask R-CNN models run on two public rat brain templates, each resampled into CAMRI's",
             "scan geometry, scored against the template's own hypothalamus label (median over 3 views).",
             "See `public_atlas_test.py` for how the images were made and the limits of this test.", "",
             "| Public brain | Model | Dice | IoU |", "|---|---|---|---|"]
    for brain in BRAINS:
        for name in RUNS:
            sel = [r for r in rows if r["brain"] == brain and r["model"] == name]
            lines.append(f"| {brain} | {name} | {np.median([r['dice'] for r in sel]):.2f} | "
                         f"{np.median([r['iou'] for r in sel]):.2f} |")
    lines += ["", "For comparison, on CAMRI's own hand-traced rats: Mask R-CNN Dice 0.74 / IoU 0.59; "
              "+ augmentation 0.72 / 0.56 (see `RESULTS.md`).", "",
              "![overlays](overlays.png)", "",
              "Yellow = agreement, red = model only, green = label only (first view of each brain)."]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:-6]))


if __name__ == "__main__":
    main()
