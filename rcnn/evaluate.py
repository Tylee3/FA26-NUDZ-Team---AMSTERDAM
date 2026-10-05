"""Scores every trained model on the 9 hand-traced test rats, next to the atlas baseline.
Usage, from the repo root:  python3 -m rcnn.evaluate

Per slice, a model's single best detection counts if its score clears the threshold
(0.5; 0 for R-CNN's SVM margin). One hypothalamus per slice at most, so one detection.

Metrics, all against Tyler's manual traces ("Me") unless noted:
  slices found      share of the 45 structure-containing slices where the model detected something
  false alarms      detections on the 63 slices that don't contain the structure
  box IoU           overlap of the predicted box with the tight box around the trace,
                    averaged over the 45 structure slices (a miss counts as 0)
  3D IoU / Dice     the same 3D overlap as auto_seg_out/results.csv. Box-only models are
                    scored as filled rectangles, which also covers tissue around the
                    structure, so compare them against "atlas (as box)" rather than "atlas".

Writes rcnn_out/comparison.csv, comparison.md, comparison_grid.png, and per model:
results.csv, overlay.png, pred_XXX.nrrd (open in 3D Slicer with rcnn/slicer_compare.py).
"""

import argparse
import csv
import pickle

import numpy as np
import SimpleITK as sitk
from PIL import Image

from .common import (ATLAS_TEST_DIR, DEFAULT_SCORE_THRESHOLD, MASK_MODELS, MODELS, ROOT,
                     SCORE_THRESHOLDS, box_iou, load_split, mask_to_box, t2_path)

LABELS = {"atlas": "Atlas (Charley's method)", "atlas_box": "Atlas, as a box",
          "rcnn": "R-CNN", "fast_rcnn": "Fast R-CNN", "faster_rcnn": "Faster R-CNN",
          "rfcn": "R-FCN", "cascade_rcnn": "Cascade R-CNN", "mask_rcnn": "Mask R-CNN"}


def overlap(a, b):
    inter, union, total = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    return (inter / union if union else 1.0), (2 * inter / total if total else 1.0)


def filled_box(box, shape):
    m = np.zeros(shape, dtype=bool)
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m


def slice_predictions(model, preds, data):
    """Per test slice: (box or None, mask or None) for the model's single best detection."""
    thresh = SCORE_THRESHOLDS.get(model, DEFAULT_SCORE_THRESHOLD)
    by_file = {p["file"]: p for p in preds}
    out = {}
    for i, f in enumerate(data["files"]):
        p = by_file.get(str(f))
        if p is None:
            continue
        if len(p["scores"]) == 0 or p["scores"].max() < thresh:
            out[i] = (None, None)
            continue
        k = int(np.argmax(p["scores"]))
        box = p["boxes"][k]
        if model in MASK_MODELS and p["masks"] is not None and k < len(p["masks"]):
            out[i] = (box, p["masks"][k] >= 128)
        else:
            out[i] = (box, filled_box(box, data["masks"][i].shape))
    return out


def atlas_predictions(data, as_box):
    out, cache = {}, {}
    for i, (n, z) in enumerate(zip(data["subjects"], data["zs"])):
        if n not in cache:
            cache[n] = sitk.GetArrayFromImage(sitk.ReadImage(str(ATLAS_TEST_DIR / f"auto_seg_{n:03d}.nrrd"))) > 0
        m = cache[n][z]
        box = mask_to_box(m)
        out[i] = (box, filled_box(box, m.shape) if as_box else m)
    return out


def score(model, per_slice, data, out_dir, write_nrrd):
    rows, volumes = [], {}
    for n in sorted(set(data["subjects"].tolist())):
        idx = [i for i in np.where(data["subjects"] == n)[0]]
        if not all(i in per_slice for i in idx):
            continue
        idx.sort(key=lambda i: data["zs"][i])
        gt_me = np.stack([data["masks"][i] for i in idx])
        gt_other = np.stack([data["masks_other"][i] for i in idx])
        pred = np.stack([per_slice[i][1] if per_slice[i][1] is not None
                         else np.zeros(gt_me.shape[1:], bool) for i in idx])
        found = alarms = 0
        box_scores = []
        for j, i in enumerate(idx):
            gt_box, pred_box = mask_to_box(gt_me[j]), per_slice[i][0]
            if gt_box is not None:
                found += pred_box is not None
                box_scores.append(box_iou(pred_box, gt_box) if pred_box is not None else 0.0)
            elif pred_box is not None:
                alarms += 1
        iou_me, dice_me = overlap(pred, gt_me)
        iou_other, dice_other = overlap(pred, gt_other)
        rows.append({"rat": int(n), "slices": len(idx), "structure_slices": len(box_scores), "slices_found": found,
                     "false_alarms": alarms, "box_iou": float(np.mean(box_scores)),
                     "iou_vs_me": float(iou_me), "dice_vs_me": float(dice_me),
                     "iou_vs_other": float(iou_other), "dice_vs_other": float(dice_other)})
        volumes[int(n)] = (pred, gt_me, idx)
        if write_nrrd:
            img = sitk.GetImageFromArray(pred.astype(np.uint8))
            img.CopyInformation(sitk.ReadImage(str(t2_path(int(n)))))
            sitk.WriteImage(img, str(out_dir / f"pred_{int(n):03d}.nrrd"))
    if rows and out_dir is not None:
        with open(out_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return rows, volumes


def overlay_tile(data, idx, pred, gt, size=200):
    j = int(np.argmax(gt.reshape(len(gt), -1).sum(1)))
    gray = data["images"][idx[j]].astype(np.float32) / 255
    rgb = np.stack([gray] * 3, -1)
    p, g = pred[j], gt[j]
    rgb[g & ~p] = [0, 1, 0]
    rgb[p & ~g] = [1, 0, 0]
    rgb[p & g] = [1, 1, 0]
    ys, xs = np.where(g | p)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (gray.shape[0] // 2, gray.shape[1] // 2)
    half = 40
    y0, x0 = max(cy - half, 0), max(cx - half, 0)
    crop = (rgb[y0:y0 + 2 * half, x0:x0 + 2 * half] * 255).astype(np.uint8)
    return Image.fromarray(crop).resize((size, size), Image.NEAREST)


def grid(tiles, labels, path, cols, size=200):
    from PIL import ImageDraw
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * size, rows * (size + 18)), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    for k, (tile, label) in enumerate(zip(tiles, labels)):
        x, y = (k % cols) * size, (k // cols) * (size + 18)
        draw.text((x + 4, y + 3), label, fill=(230, 230, 230))
        canvas.paste(tile, (x, y + 18))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="rcnn_out")
    args = ap.parse_args()
    out_root = ROOT / args.out
    data = load_split("test")

    results, all_volumes = {}, {}
    for name, per_slice, out_dir in [("atlas", atlas_predictions(data, False), None),
                                     ("atlas_box", atlas_predictions(data, True), None)]:
        results[name], all_volumes[name] = score(name, per_slice, data, out_dir, False)
    for name in MODELS:
        pred_file = out_root / name / "predictions.pkl"
        if not pred_file.exists():
            continue
        with open(pred_file, "rb") as f:
            preds = pickle.load(f)
        per_slice = slice_predictions(name, preds, data)
        results[name], all_volumes[name] = score(name, per_slice, data, out_root / name, True)

    summary = []
    for name, rows in results.items():
        if not rows:
            continue
        structure = sum(r["structure_slices"] for r in rows)
        empty = sum(r["slices"] - r["structure_slices"] for r in rows)
        summary.append({
            "model": LABELS[name], "output": "mask" if name in MASK_MODELS | {"atlas"} else "box",
            "rats": len(rows),
            "slices_found": f"{sum(r['slices_found'] for r in rows)}/{structure}",
            "false_alarms": f"{sum(r['false_alarms'] for r in rows)}/{empty}",
            "mean_box_iou": round(float(np.mean([r["box_iou"] for r in rows])), 3),
            "median_3d_iou_vs_me": round(float(np.median([r["iou_vs_me"] for r in rows])), 3),
            "median_3d_dice_vs_me": round(float(np.median([r["dice_vs_me"] for r in rows])), 3),
            "median_3d_iou_vs_other": round(float(np.median([r["iou_vs_other"] for r in rows])), 3),
        })
    with open(out_root / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    cols = list(summary[0])
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(s[c]) for c in cols) + " |" for s in summary]
    (out_root / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    names = [n for n in results if n != "atlas_box" and results[n]]
    for name in names:
        if name == "atlas":
            continue
        rows = sorted(results[name], key=lambda r: r["iou_vs_me"])
        picks = [rows[-1], rows[len(rows) // 2], rows[0]] if len(rows) >= 3 else rows
        tiles = [overlay_tile(data, all_volumes[name][r["rat"]][2], all_volumes[name][r["rat"]][0],
                              all_volumes[name][r["rat"]][1]) for r in picks]
        labels = [f"rat {r['rat']}  3D IoU {r['iou_vs_me']:.2f}  box IoU {r['box_iou']:.2f}" for r in picks]
        grid(tiles, labels, out_root / name / "overlay.png", cols=len(tiles))

    rats = sorted(r["rat"] for r in results["atlas"] if all(r["rat"] in all_volumes[n] for n in names))
    tiles, labels = [], []
    for rat in rats:
        for name in names:
            pred, gt, idx = all_volumes[name][rat]
            tiles.append(overlay_tile(data, idx, pred, gt))
            labels.append(f"{LABELS[name]} - rat {rat}")
    if tiles:
        grid(tiles, labels, out_root / "comparison_grid.png", cols=len(names))
    print(f"\nwrote {out_root / 'comparison.csv'}, comparison.md, comparison_grid.png")


if __name__ == "__main__":
    main()
