"""Preview of training on hand traces, before NUDZ's new traces arrive: does fine-tuning the
best model (zoomed Mask R-CNN, already trained on 53 atlas-labeled rats) on a few hand-traced
rats lift it past the atlas, toward the human-vs-human agreement (Dice 0.82)?

Uses the 9 hand-traced rats we have, in 3 folds fixed in advance: [1, 3, 4], [5, 6, 7],
[8, 9, 10]. Per fold, the model is fine-tuned on the hand traces of the 6 other rats and
scored on the 3 held out, so every rat is scored once by a model that never saw it.
Done twice: once fine-tuning on labeler 1's traces, once on labeler 2's.

The fair comparison with human vs. human is the score against the labeler the model did
NOT learn from: a model trained on labeler 1 scored against labeler 2 is judged exactly
like labeler 1 is judged against labeler 2.

Recipe, fixed before looking at any result: start from rcnn_out/zoom_mask_rcnn/model.pt
(stage 1 stays the atlas-trained Faster R-CNN), same zoomed-window training as rcnn/zoom.py,
600 iterations, SGD lr 0.001 (a fifth of the original), x0.1 at 70%.

Caveat: 6 training rats is very few. No gain would not prove hand traces don't help; a gain
is a strong sign that the Oct 30 traces will.

Usage (from the repo root):  python3 -m rcnn.finetune_cv [--head edge_loss|aspp|gated_edge|best] [--neighbors | --warp]
(--smoke for a tiny check). With --head, it fine-tunes that edge variant of the zoomed model
(rcnn_out/zoom_mask_rcnn_<head>) and writes rcnn_out/finetune_cv_<head>/. With --neighbors,
it fine-tunes the 2.5D zoomed model (rcnn_out/zoom_mask_rcnn_neighbors) the same way and
writes rcnn_out/finetune_cv_neighbors/.
Writes rcnn_out/finetune_cv/<labeler>/fold<k>/ (skipped if done), results.csv and
FINETUNE_CV.md.

With --warp (Hoku's idea), every training window also gets a gentle random local warp
(common.elastic_warp), the trace warped identically, and the run writes finetune_cv_warp/.
Same base model, folds, seed and settings as the plain run, so warp is the only difference.
Rule fixed before the run: keep the warp only if it lifts the fair score (vs the labeler the
model did not learn from) by more than 0.01 for both labelers.

Round 6 (HYPO_TEST=14): the 14 traced rats in 3 folds, [1, 3, 4, 15, 25], [5, 6, 7, 35, 100],
[8, 9, 10, 125], so each fold fine-tunes on 9 or 10 rats instead of 6. Starts from
rcnn_out_14/zoom_mask_rcnn, which never saw any of the 14, and writes rcnn_out_14/finetune_cv/.
"""

import argparse
import csv
import json
import pickle
import time

import numpy as np
import torch

from . import models
from .common import ROOT, TEST14, TEST_RATS, load_split, out_root, pick_device
from .evaluate import score, slice_predictions
from .train import Log, train_network
from .zoom import ZoomDataset, predict

FOLDS = [[1, 3, 4], [5, 6, 7], [8, 9, 10]]
if TEST14:
    # Round 6, fixed before any result: the 5 new rats spread over the same 3 folds.
    FOLDS = [[1, 3, 4, 15, 25], [5, 6, 7, 35, 100], [8, 9, 10, 125]]
LABELERS = {"labeler1": "masks", "labeler2": "masks_other"}
ITERS = 600
LR = 0.001
HEADS = ["edge_loss", "aspp", "gated_edge"]
BASE_RUN = "zoom_mask_rcnn"
BASE_ROOT = out_root()
HEAD = None
NEIGHBORS = False
WARP = False
OUT = out_root() / "finetune_cv"


def best_head():
    """The edge variant of the zoomed model with the best median Dice vs labeler 1 (rule fixed
    before any of them was run)."""
    def med(h):
        with open(BASE_ROOT / f"zoom_mask_rcnn_{h}" / "results.csv") as f:
            rows = [r for r in csv.DictReader(f) if int(r["rat"]) in TEST_RATS]
        return float(np.median([float(r["dice_vs_me"]) for r in rows]))
    scores = {h: med(h) for h in HEADS}
    print("edge variants, median Dice vs labeler 1:", {h: round(v, 3) for h, v in scores.items()})
    return max(scores, key=scores.get)


def run_fold(labeler, key, k, held_out, device, smoke):
    out_dir = OUT / labeler / f"fold{k}"
    if (out_dir / "predictions.pkl").exists():
        print(f"skipping {labeler} fold {k} (already done)")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rats = [n for n in TEST_RATS if n not in held_out]
    (out_dir / "config.json").write_text(json.dumps(
        {"model": "mask_rcnn", "zoom": True, "head": HEAD, "neighbors": NEIGHBORS, "warp": WARP, "base": str((BASE_ROOT / BASE_RUN / "model.pt").relative_to(ROOT)), "finetune_on": labeler,
         "train_rats": train_rats, "held_out": held_out, "iterations": ITERS, "lr": LR}, indent=2))
    log = Log(out_dir / "train.log")
    torch.manual_seed(0)
    t0 = time.time()
    model = models.build("mask_rcnn", HEAD)
    model.load_state_dict(torch.load(BASE_ROOT / BASE_RUN / "model.pt", map_location="cpu"))
    model.to(device)
    ds = ZoomDataset(rats=set(train_rats), split="test", mask_key=key, neighbors=NEIGHBORS, warp=WARP)
    train_network("mask_rcnn", model, device, out_dir, log, 10, 10 if smoke else ITERS, None, False, False,
                  dataset=ds, lr=LR)
    preds = predict(model, device, limit=12 if smoke else None, neighbors=NEIGHBORS, smoke=smoke)
    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    log(f"done: fine-tuned on {labeler} traces of rats {train_rats}, {(time.time() - t0) / 60:.1f} min")


def held_out_rows(preds, test, held_out):
    per_slice = slice_predictions("mask_rcnn", preds, test)
    keep = {i: v for i, v in per_slice.items() if int(test["subjects"][i]) in held_out}
    rows, _ = score("mask_rcnn", keep, test, None, False)
    return rows


def summarize():
    test = load_split("test")
    rows = []
    with open(BASE_ROOT / BASE_RUN / "predictions.pkl", "rb") as f:
        base = held_out_rows(pickle.load(f), test, TEST_RATS)
    for r in base:
        rows.append({"run": "atlas-trained (no fine-tune)", **r})
    for labeler in LABELERS:
        for k, held_out in enumerate(FOLDS):
            p = OUT / labeler / f"fold{k}" / "predictions.pkl"
            if not p.exists():
                continue
            with open(p, "rb") as f:
                rows += [{"run": f"fine-tuned on {labeler}", **r} for r in held_out_rows(pickle.load(f), test, held_out)]
    human = []
    for n in TEST_RATS:
        a, b = test["masks"][test["subjects"] == n], test["masks_other"][test["subjects"] == n]
        human.append(2 * (a & b).sum() / (a.sum() + b.sum()))
    with open(OUT / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def med(run, key):
        vals = [r[key] for r in rows if r["run"] == run]
        return (f"{np.median(vals):.3f}", len(vals)) if vals else ("—", 0)

    md = [f"# Fine-tuning on hand traces: 3-fold {'(14 rats)' if TEST14 else 'preview'}", "",
          f"Median 3D Dice on the {len(TEST_RATS)} hand-traced rats; every rat scored by a model that never saw it.",
          "**Bold** = scored against the labeler the model did not learn from (the fair comparison with human vs. human).",
          "", "| Model | Dice vs labeler 1 | Dice vs labeler 2 | Rats |", "|---|---|---|---|",
          f"| Human vs. human (labeler 1 vs labeler 2) | {np.median(human):.3f} | {np.median(human):.3f} | {len(human)} |"]
    for run, bold in [("atlas-trained (no fine-tune)", None), ("fine-tuned on labeler1", "dice_vs_other"),
                      ("fine-tuned on labeler2", "dice_vs_me")]:
        (d1, n), (d2, _) = med(run, "dice_vs_me"), med(run, "dice_vs_other")
        if not n:
            continue
        d1 = f"**{d1}**" if bold == "dice_vs_me" else d1
        d2 = f"**{d2}**" if bold == "dice_vs_other" else d2
        md.append(f"| {run} | {d1} | {d2} | {n} |")
    md += ["", "Per rat (Dice vs labeler 1 / vs labeler 2):", "",
           "| Rat | Human vs. human | Atlas-trained | Fine-tuned on labeler 1 | Fine-tuned on labeler 2 |",
           "|---|---|---|---|---|"]
    for n, h in zip(TEST_RATS, human):
        cells = []
        for run in ["atlas-trained (no fine-tune)", "fine-tuned on labeler1", "fine-tuned on labeler2"]:
            r = [x for x in rows if x["run"] == run and x["rat"] == n]
            cells.append(f"{r[0]['dice_vs_me']:.2f} / {r[0]['dice_vs_other']:.2f}" if r else "—")
        md.append(f"| {n} | {h:.2f} | " + " | ".join(cells) + " |")
    (OUT / "FINETUNE_CV.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=None, help="edge variant to fine-tune (edge_loss / aspp / gated_edge), "
                                                  "or 'best' to pick by the rule in best_head()")
    ap.add_argument("--neighbors", action="store_true", help="fine-tune the 2.5D zoomed model")
    ap.add_argument("--warp", action="store_true", help="gentle random local warp of each training window")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    global OUT, HEAD, BASE_RUN, BASE_ROOT, NEIGHBORS, WARP
    if sum(map(bool, (args.head, args.neighbors, args.warp))) > 1:
        ap.error("--head, --neighbors and --warp are separate experiments")
    NEIGHBORS, WARP = args.neighbors, args.warp
    if args.smoke and (args.head or args.neighbors or TEST14):
        BASE_ROOT = out_root(smoke=True)
    HEAD = best_head() if args.head == "best" else args.head
    if HEAD:
        BASE_RUN, OUT = f"zoom_mask_rcnn_{HEAD}", out_root() / f"finetune_cv_{HEAD}"
        print(f"fine-tuning the zoomed model with the {HEAD} head")
    if NEIGHBORS:
        BASE_RUN, OUT = "zoom_mask_rcnn_neighbors", out_root() / "finetune_cv_neighbors"
        print("fine-tuning the 2.5D zoomed model")
    if WARP:
        OUT = out_root() / "finetune_cv_warp"
        print("fine-tuning with a gentle random local warp of each training window")
    if args.smoke:
        OUT = out_root(smoke=True) / OUT.name
    device = pick_device(args.device)
    for labeler, key in list(LABELERS.items())[:1 if args.smoke else None]:
        for k, held_out in enumerate(FOLDS[:1] if args.smoke else FOLDS):
            run_fold(labeler, key, k, held_out, device, args.smoke)
    summarize()


if __name__ == "__main__":
    main()
