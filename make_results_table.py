"""Builds RESULTS.md: Dice and IoU for every method, read straight from the result files,
so the table can be regenerated whenever a new run finishes.

Usage:  python3 make_results_table.py
"""

import csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent
TEST_RATS = [1, 3, 4, 5, 6, 7, 8, 9, 10]


def per_rat(path):
    with open(path) as f:
        return {int(r["rat"]): r for r in csv.DictReader(f)}


def medians(path):
    rows = per_rat(path)
    rows = [rows[n] for n in TEST_RATS if n in rows]
    med = lambda k: float(np.median([float(r[k]) for r in rows])) if rows and k in rows[0] else None
    found = alarms = None
    if rows and "slices_found" in rows[0]:
        found = f"{sum(int(r['slices_found']) for r in rows)}/{sum(int(r['structure_slices']) for r in rows)}"
        alarms = f"{sum(int(r['false_alarms']) for r in rows)}/{sum(int(r['slices']) - int(r['structure_slices']) for r in rows)}"
    return {"dice": med("dice_vs_me"), "iou": med("iou_vs_me"), "dice2": med("dice_vs_other"),
            "iou2": med("iou_vs_other"), "found": found, "alarms": alarms, "n": len(rows)}


def human_agreement():
    with np.load(ROOT / "rcnn_data" / "test.npz") as d:
        subjects, me, other = d["subjects"], d["masks"], d["masks_other"]
    dice, iou = [], []
    for n in TEST_RATS:
        a, b = me[subjects == n], other[subjects == n]
        inter, union = (a & b).sum(), (a | b).sum()
        dice.append(2 * inter / (a.sum() + b.sum()))
        iou.append(inter / union)
    return {"dice": float(np.median(dice)), "iou": float(np.median(iou)), "n": len(TEST_RATS)}


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def row(name, output, m, note=""):
    return (f"| {name} | {output} | {fmt(m.get('dice'))} | {fmt(m.get('iou'))} | {fmt(m.get('dice2'))} | "
            f"{fmt(m.get('iou2'))} | {m.get('found') or '—'} | {m.get('alarms') or '—'} | {note} |")


def main():
    lines = [
        "# Results",
        "",
        "Median Dice and IoU of every method on the 9 hand-traced test rats (1, 3–10), measured over the",
        "whole 3D volume. No method saw these rats during training. Labeler 1 is Tyler's traces;",
        "labeler 2 is the second, independent set. Regenerate with `python3 make_results_table.py`.",
        "",
        "| Method | Output | Dice (lab. 1) | IoU (lab. 1) | Dice (lab. 2) | IoU (lab. 2) | Slices found | False alarms | Notes |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    h = human_agreement()
    lines.append(f"| **Human vs. human** | outline | {fmt(h['dice'])} | {fmt(h['iou'])} | — | — | — | — | the realistic ceiling |")

    lines.append(row("**Waxholm atlas, rigid** (Charley's method)", "outline", medians(ROOT / "auto_seg_out/results.csv"),
                     "baseline; labels for training"))
    deform = {k: medians(ROOT / f"atlas_deformable_out/{k}/results.csv") for k in "abc"
              if (ROOT / f"atlas_deformable_out/{k}/results.csv").exists()}
    if deform:
        best = max(deform, key=lambda k: deform[k]["dice"])
        lines.append(row("Waxholm atlas, deformable", "outline", deform[best], f"best of {len(deform)} preset settings"))
    sigma = {v: medians(ROOT / f"sigma_out/{v}/results.csv") for v in
             ["plain", "mask", "waxholm_init", "waxholm_init_refine"] if (ROOT / f"sigma_out/{v}/results.csv").exists()}
    if sigma:
        best = max(sigma, key=lambda v: sigma[v]["dice"])
        lines.append(row("SIGMA atlas", "outline", sigma[best], f"best of {len(sigma)} variants ({best})"))

    models = [("R-CNN", "rcnn", "box"), ("Fast R-CNN", "fast_rcnn", "box"), ("Faster R-CNN", "faster_rcnn", "box"),
              ("R-FCN", "rfcn", "box"), ("Cascade R-CNN", "cascade_rcnn", "box"), ("Mask R-CNN", "mask_rcnn", "outline"),
              ("Mask R-CNN + augmentation", "mask_rcnn_augment", "outline"),
              ("Mask R-CNN + neighbor slices", "mask_rcnn_neighbors", "outline"),
              ("Mask R-CNN + neighbor slices + augmentation", "mask_rcnn_neighbors_augment", "outline"),
              ("Mask R-CNN on a zoomed view, box first (Faster R-CNN)", "zoom_mask_rcnn", "outline"),
              ("Mask R-CNN on a zoomed view, box first + augmentation", "zoom_mask_rcnn_augment", "outline"),
              ("3D U-Net", "unet3d", "outline"),
              ("3D U-Net + augmentation", "unet3d_augment", "outline"),
              ("Mesh R-CNN (3D shape from one slice)", "mesh_rcnn", "3D mesh"),
              ("Zoomed Mask R-CNN + mirror pass", "zoom_mask_rcnn_mirror", "outline"),
              ("Zoomed Mask R-CNN + augmentation + mirror pass", "zoom_mask_rcnn_augment_mirror", "outline"),
              ("3D U-Net + augmentation + mirror pass", "unet3d_augment_mirror", "outline"),
              ("**Vote of 5** (atlas, 2 zoomed, Mesh R-CNN, 3D U-Net; 3 needed)", "ensemble_vote", "outline"),
              ("**Vote of 5 + mirror pass**", "ensemble_vote_mirror", "outline")]
    for name, run, output in models:
        path = ROOT / "rcnn_out" / run / "results.csv"
        if path.exists():
            note = {"box": "scored as a filled box", "3D mesh": "mesh cut back into slices"}.get(output, "")
            lines.append(row(name, output, medians(path), note))
    if not (ROOT / "rcnn_out" / "mesh_rcnn" / "results.csv").exists():
        lines.append("| Mesh R-CNN | 3D mesh | — | — | — | — | — | — | runs on Colab, see `rcnn/mesh_pack.py` |")

    cv = ROOT / "rcnn_out" / "finetune_cv" / "results.csv"
    if cv.exists():
        with open(cv) as f:
            rows_cv = list(csv.DictReader(f))
        m = lambda run, k: float(np.median([float(r[k]) for r in rows_cv if r["run"] == run]))
        lines += [
            "",
            "**Preview: fine-tuned on hand traces** (zoomed Mask R-CNN, 3-fold cross-validation on the same 9 rats:",
            "fine-tuned on 6, scored on the 3 held out). The fair comparison with human vs. human is the score",
            "against the labeler the model did *not* learn from (bold). Details: `rcnn_out/finetune_cv/FINETUNE_CV.md`.",
            "",
            "| Fine-tuned on | Dice (lab. 1) | Dice (lab. 2) |",
            "|---|---|---|",
            f"| labeler 1's traces | {m('fine-tuned on labeler1', 'dice_vs_me'):.2f} | **{m('fine-tuned on labeler1', 'dice_vs_other'):.2f}** |",
            f"| labeler 2's traces | **{m('fine-tuned on labeler2', 'dice_vs_me'):.2f}** | {m('fine-tuned on labeler2', 'dice_vs_other'):.2f} |",
        ]

    lines += [
        "",
        "**How to read it:**",
        "- Box-only models are scored as filled rectangles, which also cover tissue around the structure.",
        "  Their fair reference is the Waxholm atlas scored the same way: Dice 0.64 / IoU 0.48.",
        "- With 9 test rats, differences of a few hundredths are ties.",
        "- All learned models were trained on Waxholm atlas labels for 53 other rats, so the atlas is their",
        "  ceiling until training moves to hand traces.",
        "",
        "Details: [Waxholm baseline](auto_seg_out/WAXHOLM_BASELINE.md) · six-model baseline and experiments:",
        "`rcnn_out/comparison.md` · SIGMA: `sigma_registration.py` docstring.",
    ]
    (ROOT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[6:-11]))


if __name__ == "__main__":
    main()
