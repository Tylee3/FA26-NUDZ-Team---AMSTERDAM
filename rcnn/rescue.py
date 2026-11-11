"""Round 8: end-slice rescue. No retraining; the fine-tuned fold models are reused.

Why: in round 7 every slice the fine-tuned models missed (9 of 69, on 8 of 14 rats) was the
first or last slice the hypothalamus appears on, never a gap in the middle. Either stage 1's
box score fell just under its 0.5 bar (4 slices) or stage 2 drew nothing (4 slices).

The rule, fixed before seeing any result: a slice the model left empty, directly next to a
slice it did find, gets a second try with both bars lowered to RESCUE (0.2):
  1. stage 1's own box, if its score is at least RESCUE, else
  2. the found neighbor's outline box (the hypothalamus barely moves from one slice to the next).
Stage 2's outline is accepted if its score is at least RESCUE. Only one slice out from the
found ones, so outlines cannot creep along the stack. The cost to watch: false alarms on the
slice just past the true end.

Run for all 9 fold-model sets (expert 1 / expert 2 / both, seeds 0-2), so each comparison is
paired: the same models, with and without the rule, and seeds show whether a change repeats.
0.1 and 0.3 are also scored, only to see whether the result hinges on the exact bar.

Writes rcnn_out_14/RESCUE.md (the summary), and rcnn_out_14/rescue_<member>/ for seed 0 at
0.2, in the usual format, so rcnn.evaluate scores them alongside everything else.

Usage (from the repo root):  HYPO_TEST=14 python3 -m rcnn.rescue   (--smoke: one model set, one fold)
"""

import argparse
import json
import pickle
import statistics

import numpy as np
import torch

from . import models
from .common import load_split, mask_to_box, out_root, pick_device
from .ensemble import write_run
from .evaluate import score
from .zoom import STAGE1, STAGE1_THRESHOLD, STAGE2_THRESHOLD, crop_tensor, paste, slice_lookup, window

RESCUE = 0.2
SWEEP = [0.1, 0.2, 0.3]
SEEDS = {0: "finetune_cv", 1: "finetune_cv_seed1", 2: "finetune_cv_seed2"}
MEMBERS = {"labeler1": "expert 1", "labeler2": "expert 2", "both": "both experts"}
WHOSE = {"labeler1": "expert 1's", "labeler2": "expert 2's", "both": "both experts'"}
# The expert each model never learned from: the fair score is the one against that expert.
FAIR = {"labeler1": "dice_vs_other", "labeler2": "dice_vs_me", "both": None}


@torch.no_grad()
def outline(model, data, lookup, i, box, device):
    """Stage 2 on the window around box: (score, probability map) of its best outline, or (0, None)."""
    win = window(box)
    det = model([crop_tensor(data, lookup, i, win, False).to(device)])[0]
    if not len(det["scores"]):
        return 0.0, None
    k = int(det["scores"].argmax())
    return float(det["scores"][k]), paste(det["masks"][k, 0].cpu().numpy(), win, data["images"][i].shape)


def run_folds(root, member, data, stage1, device, smoke):
    """Every held-out slice's masks for this model set: the plain rule, and each rescue bar."""
    lookup = slice_lookup(data)
    plain, rescued = {}, {r: {} for r in SWEEP}
    for k, fold in enumerate(sorted((root / member).glob("fold*"))[:1 if smoke else None]):
        held_out = set(json.loads((fold / "config.json").read_text())["held_out"])
        model = models.build("mask_rcnn")
        model.load_state_dict(torch.load(fold / "model.pt", map_location="cpu"))
        model.to(device).eval()
        idx = [i for i in range(len(data["files"])) if int(data["subjects"][i]) in held_out]
        empty = np.zeros(data["images"][0].shape, bool)
        first = {}
        for i in idx:
            p1 = stage1[str(data["files"][i])]
            s1 = float(p1["scores"].max()) if len(p1["scores"]) else 0.0
            s2, prob = outline(model, data, lookup, i, p1["boxes"][int(np.argmax(p1["scores"]))], device) \
                if s1 >= min(SWEEP) else (0.0, None)
            first[i] = (s1, s2, prob)
            ok = s1 >= STAGE1_THRESHOLD and s2 >= STAGE2_THRESHOLD and prob is not None
            plain[i] = prob >= 0.5 if ok else empty
        for i in idx:
            if plain[i].any():
                continue
            n, z = int(data["subjects"][i]), int(data["zs"][i])
            found = [j for j in (lookup.get((n, z - 1)), lookup.get((n, z + 1))) if j is not None and plain[j].any()]
            neighbor = None
            for r in SWEEP:
                s1, s2, prob = first[i]
                if not (found and s1 >= r and s2 >= r and prob is not None and (prob >= 0.5).any()):
                    if found and neighbor is None:
                        j = max(found, key=lambda j: plain[j].sum())
                        neighbor = outline(model, data, lookup, i, mask_to_box(plain[j]), device)
                    s2, prob = neighbor if found else (0.0, None)
                hit = found and s2 >= r and prob is not None
                rescued[r][i] = prob >= 0.5 if hit else empty
        for r in SWEEP:
            rescued[r].update({i: m for i, m in plain.items() if i not in rescued[r]})
        stored = {p["file"]: p for p in pickle.load(open(fold / "predictions.pkl", "rb"))}
        differ = sum(plain[i].any() != (len(stored[str(data["files"][i])]["scores"]) > 0) for i in idx)
        print(f"  {root.name}/{member}/{fold.name}: plain rule matches the stored predictions on "
              f"{len(idx) - differ}/{len(idx)} slices", flush=True)
    return plain, rescued


def summary(masks, data):
    rows, _ = score("mask_rcnn", {i: (mask_to_box(m), m) for i, m in masks.items()}, data, None, False)
    return {
        "dice_vs_me": statistics.median(r["dice_vs_me"] for r in rows),
        "dice_vs_other": statistics.median(r["dice_vs_other"] for r in rows),
        "mean_vs_me": statistics.mean(r["dice_vs_me"] for r in rows),
        "mean_vs_other": statistics.mean(r["dice_vs_other"] for r in rows),
        "found": sum(r["slices_found"] for r in rows), "of": sum(r["structure_slices"] for r in rows),
        "alarms": sum(r["false_alarms"] for r in rows), "rats": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    base = out_root()
    data = load_split("test")
    device = pick_device(args.device)
    with open(base / STAGE1 / "predictions.pkl", "rb") as f:
        stage1 = {p["file"]: p for p in pickle.load(f)}
    sets = [(0, "labeler1")] if args.smoke else [(s, m) for s in SEEDS for m in MEMBERS]
    results = {}
    for seed, member in sets:
        plain, rescued = run_folds(base / SEEDS[seed], member, data, stage1, device, args.smoke)
        results[seed, member] = {"plain": summary(plain, data), **{r: summary(rescued[r], data) for r in SWEEP}}
        p, q = results[seed, member]["plain"], results[seed, member][RESCUE]
        print(f"seed {seed} {member}: found {p['found']}->{q['found']}/{p['of']}, false alarms "
              f"{p['alarms']}->{q['alarms']}, median Dice vs expert 1 {p['dice_vs_me']:.3f}->{q['dice_vs_me']:.3f}, "
              f"vs expert 2 {p['dice_vs_other']:.3f}->{q['dice_vs_other']:.3f}", flush=True)
        if seed == 0 and not args.smoke:
            write_run(base, f"rescue_{member}", rescued[RESCUE], data, {
                "model": "mask_rcnn", "zoom": True, "rescue": RESCUE, "member": member,
                "label": f"Mask R-CNN + zoom, fine-tuned on {WHOSE[member]} traces + end-slice rescue (3-fold)"})
    if not args.smoke:
        write_summary(base / "RESCUE.md", results)


def write_summary(path, results):
    def cell(v, member, key):
        return f"**{v:.3f}**" if FAIR[member] == key else f"{v:.3f}"

    lines = ["# Round 8: end-slice rescue (no retraining)", "",
             f"Same fold models with and without the rule (bar {RESCUE}, fixed in advance). Median 3D Dice over "
             "the 14 traced rats, each scored by a fold that never saw it. **Bold** = the fair score, against the "
             "expert the model did not learn from. Human vs. human is 0.824.", "",
             "| Seed | Trained on | Slices found | False alarms | Dice vs expert 1 | Dice vs expert 2 "
             "| Mean Dice vs expert 1 | Mean Dice vs expert 2 |", "|---|---|---|---|---|---|---|---|"]
    for (seed, member), res in results.items():
        p, q = res["plain"], res[RESCUE]
        lines.append(f"| {seed} | {MEMBERS[member]} | {p['found']} → {q['found']} of {p['of']} | {p['alarms']} → {q['alarms']} "
                     f"| {cell(p['dice_vs_me'], member, 'dice_vs_me')} → {cell(q['dice_vs_me'], member, 'dice_vs_me')} "
                     f"| {cell(p['dice_vs_other'], member, 'dice_vs_other')} → {cell(q['dice_vs_other'], member, 'dice_vs_other')} "
                     f"| {p['mean_vs_me']:.3f} → {q['mean_vs_me']:.3f} | {p['mean_vs_other']:.3f} → {q['mean_vs_other']:.3f} |")
    fair = [res[RESCUE][FAIR[m]] - res["plain"][FAIR[m]] for (s, m), res in results.items() if FAIR[m]]
    lines += ["", f"Fair median change, averaged over the 3 seeds and both single-expert models: {statistics.mean(fair):+.3f} "
              f"(keep rule: more than +0.01; seed noise is about 0.007).", "",
              "The rule mostly reaches the weakest rats, which a median barely registers, so the mean is shown too.", "",
              "Does the exact bar matter? Fair median change averaged the same way, per bar:", "",
              "| Bar | " + " | ".join(str(r) for r in SWEEP) + " |", "|---|" + "---|" * len(SWEEP)]
    per_bar = [statistics.mean(res[r][FAIR[m]] - res["plain"][FAIR[m]] for (s, m), res in results.items() if FAIR[m])
               for r in SWEEP]
    lines.append("| Change | " + " | ".join(f"{v:+.3f}" for v in per_bar) + " |")
    lines += ["", "Per rat, seed 0, trained on expert 2, Dice vs expert 1 (fair), without → with:", "",
              "| Rat | Without | With |", "|---|---|---|"]
    if (0, "labeler2") in results:
        res = results[0, "labeler2"]
        for a, b in zip(res["plain"]["rats"], res[RESCUE]["rats"]):
            mark = " ←" if abs(a["dice_vs_me"] - b["dice_vs_me"]) > 0.005 else ""
            lines.append(f"| {a['rat']} | {a['dice_vs_me']:.2f} | {b['dice_vs_me']:.2f}{mark} |")
    path.write_text("\n".join(lines) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
