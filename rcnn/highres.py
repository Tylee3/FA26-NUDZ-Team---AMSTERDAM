"""Round 9: the R-CNN models on the 5 high-resolution rats (rcnn_data_hr/, built by
rcnn.build_highres). No training; every model was trained on thick-protocol rats only, so
this asks how well each one carries over to NUDZ's finer protocol.

Per version of the data (1mm slices like the training scans, and the native 0.2 mm slices):
  stage 1   the atlas-trained Faster R-CNN (rcnn_out_14/faster_rcnn) on every slice
  stage 2   each zoomed model on the window around stage 1's box, exactly as rcnn/zoom.py:
              2D:   atlas-trained, and fine-tuned on expert 1 / expert 2 / both
              2.5D: atlas-trained, and fine-tuned on expert 1 / expert 2
A fine-tuned model is 3 fold models; none of them saw these rats, so all 3 are used and
their probability maps averaged before the 0.5 cut (the same for 2D and 2.5D, so the two
stay comparable). The 2.5D models' neighbour slices are 1 mm away, as in training: the next
slice on the 1mm data, 5 slices over on the 0.2 mm data.

Scores: 3D Dice / IoU against expert 1 (Jaison, all 5 rats) and expert 2 (rats 50 and 60).
Writes rcnn_out_hr/<version>/<run>/ in the usual format and rcnn_out_hr/HIGHRES.md, which
also picks up the MedSAM2 runs (rcnn/medsam2.py) written to the same folders.

Usage (from the repo root):  HYPO_TEST=14 python3 -m rcnn.highres   (--smoke: rat 50, 1mm, 2 models)
"""

import argparse
import json
import pickle

import numpy as np

from .common import ROOT, mask_to_box

OUT_HR = ROOT / "rcnn_out_hr"
STEPS = {"1mm": 1, "02mm": 5}
NAMES = {"1mm": "1 mm slices (like the training scans)", "02mm": "0.2 mm slices (native)"}


def load_hr(version):
    from .build_highres import HR_DIR
    with np.load(HR_DIR / f"test_{version}.npz") as f:
        return {k: f[k] for k in f.files}


def specs():
    """(run, label, model files, 2.5D?) for every zoomed model that exists."""
    from .common import out_root
    r = out_root()
    who = {"labeler1": "expert 1", "labeler2": "expert 2", "both": "both experts"}
    out = [("zoom_2d", "2D, atlas-trained", [r / "zoom_mask_rcnn/model.pt"], False)]
    out += [(f"ft_2d_{m}", f"2D, fine-tuned on {who[m]}", [r / f"finetune_cv/{m}/fold{k}/model.pt" for k in range(3)], False)
            for m in who]
    out += [("zoom_25d", "2.5D, atlas-trained", [r / "zoom_mask_rcnn_neighbors/model.pt"], True)]
    out += [(f"ft_25d_{m}", f"2.5D, fine-tuned on {who[m]}",
             [r / f"finetune_cv_neighbors/{m}/fold{k}/model.pt" for k in range(3)], True) for m in ("labeler1", "labeler2")]
    return out


def crops(data, lookup, i, win, neighbors, step):
    import torch
    from .common import image_tensor
    from .zoom import cut

    def crop(k):
        return cut(data["images"][k], win, "bilinear").clip(0, 255).astype(np.uint8)
    if not neighbors:
        return image_tensor(crop(i))
    n, z = int(data["subjects"][i]), int(data["zs"][i])
    ks = [lookup.get((n, z - step), i), i, lookup.get((n, z + step), i)]
    return torch.from_numpy(np.stack([crop(k) for k in ks])).float().div(255.0)


def stage1(data, device, idx):
    import torch
    from . import models
    from .common import image_tensor, out_root
    model = models.build("faster_rcnn")
    model.load_state_dict(torch.load(out_root() / "faster_rcnn/model.pt", map_location="cpu"))
    model.to(device).eval()
    preds = {}
    with torch.no_grad():
        for i in idx:
            out = model([image_tensor(data["images"][i]).to(device)])[0]
            preds[i] = (out["boxes"].cpu().numpy(), out["scores"].cpu().numpy())
    return preds


def stage2(paths, neighbors, step, data, s1, device, idx):
    """Slice index -> mask: the average probability of the models, cut at 0.5."""
    import torch
    from . import models
    from .zoom import STAGE1_THRESHOLD, STAGE2_THRESHOLD, paste, slice_lookup, window
    lookup = slice_lookup(data)
    probs = {i: np.zeros(data["images"][i].shape, np.float32) for i in idx}
    for path in paths:
        model = models.build("mask_rcnn")
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.to(device).eval()
        with torch.no_grad():
            for i in idx:
                boxes, scores = s1[i]
                if not len(scores) or scores.max() < STAGE1_THRESHOLD:
                    continue
                win = window(boxes[int(np.argmax(scores))])
                det = model([crops(data, lookup, i, win, neighbors, step).to(device)])[0]
                if len(det["scores"]) and det["scores"].max() >= STAGE2_THRESHOLD:
                    m = det["masks"][int(det["scores"].argmax()), 0].cpu().numpy()
                    probs[i] += paste(m, win, data["images"][i].shape) / len(paths)
    return {i: p >= 0.5 for i, p in probs.items()}


def overlap(a, b):
    inter, union, total = (a & b).sum(), (a | b).sum(), a.sum() + b.sum()
    return (inter / union if union else 1.0), (2 * inter / total if total else 1.0)


def score_hr(masks, data):
    """Per rat: Dice / IoU vs expert 1, vs expert 2 (None without one), slices found, false alarms."""
    rows = []
    for n in sorted(set(data["subjects"].tolist())):
        idx = sorted(np.where(data["subjects"] == n)[0], key=lambda i: data["zs"][i])
        if not all(i in masks for i in idx):
            continue
        pred = np.stack([masks[i] for i in idx])
        me, other = data["masks"][idx], data["masks_other"][idx]
        iou1, dice1 = overlap(pred, me)
        iou2, dice2 = overlap(pred, other) if data["has_other"][idx[0]] else (None, None)
        has = me.reshape(len(idx), -1).any(1)
        hit = pred.reshape(len(idx), -1).any(1)
        rows.append({"rat": int(n), "dice_vs_me": dice1, "iou_vs_me": iou1, "dice_vs_other": dice2, "iou_vs_other": iou2,
                     "found": int((has & hit).sum()), "of": int(has.sum()), "alarms": int((~has & hit).sum())})
    return rows


def write_masks(version, run, masks, data, config):
    from .ensemble import write_run
    write_run(OUT_HR / version, run, masks, data, config)


def read_masks(d, data):
    by_file = {p["file"]: p for p in pickle.load(open(d / "predictions.pkl", "rb"))}
    empty = np.zeros(data["images"][0].shape, bool)
    return {i: (by_file[str(f)]["masks"][0] >= 128 if by_file[str(f)]["masks"] is not None else empty)
            for i, f in enumerate(data["files"]) if str(f) in by_file}


def summarize():
    """HIGHRES.md: every run in rcnn_out_hr/<version>/, R-CNN and MedSAM2 alike."""
    def med(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        return f"{np.median(vals):.3f}" if vals else "—"

    md = ["# Round 9: the 5 high-resolution rats", "",
          "Median 3D Dice / IoU. Expert 1 = Jaison (all 5 rats); expert 2 traced rats 50 and 60 only, so the "
          "expert 2 columns are 2 rats. No model trained on any of these rats or on this scan protocol. "
          "Fine-tuned R-CNN rows average their 3 fold models.", ""]
    for version in STEPS:
        if not (OUT_HR / version).exists():
            continue
        data = load_hr(version)
        human = score_hr({i: m for i, m in enumerate(data["masks_other"])}, data)
        human = [r for r in human if r["dice_vs_other"] is not None]
        md += [f"## {NAMES[version]}", "",
               "| Model | Dice vs expert 1 | IoU vs expert 1 | Dice vs expert 2 | IoU vs expert 2 | Slices found | False alarms |",
               "|---|---|---|---|---|---|---|",
               f"| Expert 2 vs expert 1 (human vs human, rats {', '.join(str(r['rat']) for r in human)}) | "
               f"{med(human, 'dice_vs_me')} | {med(human, 'iou_vs_me')} | — | — | | |"]
        per_rat = {}
        for d in sorted((OUT_HR / version).iterdir()):
            cfg = d / "config.json"
            if not (d / "predictions.pkl").exists() or not cfg.exists() or "label" not in json.loads(cfg.read_text()):
                continue
            label = json.loads(cfg.read_text())["label"]
            rows = score_hr(read_masks(d, data), data)
            per_rat[label] = rows
            md.append(f"| {label} | {med(rows, 'dice_vs_me')} | {med(rows, 'iou_vs_me')} | {med(rows, 'dice_vs_other')} | "
                      f"{med(rows, 'iou_vs_other')} | {sum(r['found'] for r in rows)}/{sum(r['of'] for r in rows)} | "
                      f"{sum(r['alarms'] for r in rows)} |")
        rats = sorted(set(data["subjects"].tolist()))
        md += ["", "Per rat, Dice vs expert 1:", "", "| Model | " + " | ".join(f"Rat {n}" for n in rats) + " |",
               "|---|" + "---|" * len(rats)]
        for label, rows in per_rat.items():
            by = {r["rat"]: r["dice_vs_me"] for r in rows}
            md.append(f"| {label} | " + " | ".join(f"{by[n]:.2f}" if n in by else "—" for n in rats) + " |")
        md.append("")
    (OUT_HR / "HIGHRES.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


def main():
    from .common import pick_device
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    for version, step in STEPS.items():
        if args.smoke and version != "1mm":
            continue
        data = load_hr(version)
        idx = [i for i in range(len(data["files"])) if not args.smoke or data["subjects"][i] == 50]
        folder = version + ("_smoke" if args.smoke else "")
        (OUT_HR / folder).mkdir(parents=True, exist_ok=True)
        s1 = stage1(data, device, idx)
        with open(OUT_HR / folder / "stage1.pkl", "wb") as f:
            pickle.dump({str(data["files"][i]): {"boxes": b, "scores": s} for i, (b, s) in s1.items()}, f)
        for run, label, paths, neighbors in specs()[:2 if args.smoke else None]:
            missing = [p for p in paths if not p.exists()]
            if missing:
                print(f"{version} {run}: skipped, missing {missing[0].relative_to(ROOT)}")
                continue
            masks = stage2(paths, neighbors, step, data, s1, device, idx)
            write_masks(folder, run, masks, data,
                        {"model": "mask_rcnn", "zoom": True, "neighbors": neighbors, "models": [str(p.relative_to(ROOT)) for p in paths],
                         "label": f"R-CNN {label}"})
            rows = score_hr(masks, data)
            print(f"{version} {run}: median Dice vs expert 1 {np.median([r['dice_vs_me'] for r in rows]):.3f}", flush=True)
    if not args.smoke:
        summarize()


if __name__ == "__main__":
    main()
