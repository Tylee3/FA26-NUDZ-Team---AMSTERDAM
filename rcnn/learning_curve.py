"""Learning curve: does the best model (zoomed Mask R-CNN, rcnn/zoom.py) still improve with
more training rats, or has it flattened out?

Retrains it on random subsets of 13, 26 and 40 of the 53 atlas-labeled training rats, two
different random subsets per size, and scores each on the same 9 hand-traced test rats.
The 53-rat point is the existing rcnn_out/zoom_mask_rcnn run, not retrained.

Every run trains for the SAME number of steps as the 53-rat run (3180 iterations, learning
rate x0.1 at 70%), so smaller sets are not also undertrained: the only difference is how
many different rats the model sees. Each run takes about as long as the full run (~55 min).

Reading it: if Dice still rises from 40 to 53 rats, more atlas-labeled scans would help;
if it is flat, the labels (the atlas outline) are the limit, not the number of scans.

Usage (from the repo root):  python3 -m rcnn.learning_curve   (--smoke for a tiny check)
Writes rcnn_out/learning_curve/<run>/ per run (skipped if already done), then
rcnn_out/learning_curve/LEARNING_CURVE.md, results.csv and learning_curve.png.
"""

import argparse
import csv
import json
import pickle
import time

import numpy as np
import torch

from . import models
from .common import ROOT, load_split, pick_device
from .evaluate import score
from .train import Log, train_network
from .zoom import ZoomDataset, predict

SIZES = [13, 26, 40]
SEEDS = [0, 1]
FULL_ITERS = 3180
OUT = ROOT / "rcnn_out" / "learning_curve"


def subset(seed, size, rats):
    rng = np.random.default_rng(seed)
    return sorted(int(r) for r in rng.choice(rats, size=size, replace=False))


def run_one(name, rats, device, smoke):
    out_dir = OUT / name
    if (out_dir / "predictions.pkl").exists():
        print(f"skipping {name} (already done)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({"model": "mask_rcnn", "zoom": True, "train_rats": rats,
                                                     "iterations": FULL_ITERS}, indent=2))
    log = Log(out_dir / "train.log")
    torch.manual_seed(0)
    t0 = time.time()
    model = models.build("mask_rcnn").to(device)
    ds = ZoomDataset(limit=None, rats=set(rats))
    train_network("mask_rcnn", model, device, out_dir, log, 10, 10 if smoke else FULL_ITERS, None, False, False,
                  dataset=ds)
    preds = predict(model, device, 12 if smoke else None)
    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    log(f"done: {len(rats)} rats, {(time.time() - t0) / 60:.1f} min total")


def medians(results_csv):
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    return {k: float(np.median([float(r[k]) for r in rows])) for k in ["dice_vs_me", "iou_vs_me", "dice_vs_other"]}


def summarize():
    from .evaluate import slice_predictions
    test = load_split("test")
    rows = []
    for d in sorted(OUT.iterdir()):
        if not (d / "predictions.pkl").exists():
            continue
        with open(d / "predictions.pkl", "rb") as f:
            per_slice = slice_predictions("mask_rcnn", pickle.load(f), test)
        score("mask_rcnn", per_slice, test, d, False)
        cfg = json.loads((d / "config.json").read_text())
        rows.append({"run": d.name, "rats": len(cfg["train_rats"]), **medians(d / "results.csv")})
    full = ROOT / "rcnn_out" / "zoom_mask_rcnn" / "results.csv"
    if full.exists():
        rows.append({"run": "zoom_mask_rcnn (existing)", "rats": 53, **medians(full)})
    rows.sort(key=lambda r: (r["rats"], r["run"]))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    md = ["# Learning curve: zoomed Mask R-CNN", "",
          "Median 3D score on the 9 hand-traced test rats, by number of atlas-labeled training rats.",
          "Every run trains for the same 3180 steps; two random subsets per size.", "",
          "| Training rats | Run | Dice (lab. 1) | IoU (lab. 1) | Dice (lab. 2) |", "|---|---|---|---|---|"]
    md += [f"| {r['rats']} | {r['run']} | {r['dice_vs_me']:.3f} | {r['iou_vs_me']:.3f} | {r['dice_vs_other']:.3f} |"
           for r in rows]
    (OUT / "LEARNING_CURVE.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label, color in [("dice_vs_me", "vs labeler 1", "tab:blue"), ("dice_vs_other", "vs labeler 2", "tab:orange")]:
        xs = sorted({r["rats"] for r in rows})
        ax.scatter([r["rats"] for r in rows], [r[key] for r in rows], color=color, alpha=0.6)
        ax.plot(xs, [np.mean([r[key] for r in rows if r["rats"] == x]) for x in xs], color=color, label=label)
    ax.axhline(0.825, color="gray", ls="--", lw=1)
    ax.text(14, 0.828, "human vs. human", color="gray", fontsize=8)
    ax.set_xlabel("atlas-labeled training rats")
    ax.set_ylabel("median 3D Dice, 9 test rats")
    ax.set_title("Zoomed Mask R-CNN: more scans vs. score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "learning_curve.png", dpi=150)
    print(f"\nwrote {OUT / 'LEARNING_CURVE.md'}, results.csv, learning_curve.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    global OUT
    if args.smoke:
        OUT = ROOT / "rcnn_out_smoke" / "learning_curve"
    OUT.mkdir(parents=True, exist_ok=True)
    device = pick_device(args.device)
    rats = sorted(set(load_split("train")["subjects"].tolist()))
    for size in SIZES[:1] if args.smoke else SIZES:
        for seed in SEEDS[:1] if args.smoke else SEEDS:
            run_one(f"zoom_n{size:02d}_s{seed}", subset(seed, size, rats), device, args.smoke)
    summarize()


if __name__ == "__main__":
    main()
