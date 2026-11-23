"""Round 9: MedSAM2 (bowang-lab/MedSAM2, SAM2.1 Tiny fine-tuned on medical scans), as
released, no training. Weights: third_party/MedSAM2/checkpoints/MedSAM2_latest.pt, licensed
for research and education only.

MedSAM2 reads a scan like a video: given a box on one slice, it outlines the structure there,
then carries the outline slice by slice to both ends of the stack (two passes, forward and
backward, as in the authors' medsam2_infer_3D_CT.py; the largest connected piece is kept).

Two prompts, fixed before any result:
  auto    the atlas-trained Faster R-CNN's box (stage 1 of the zoomed model) on the slice
          where it is most confident; fully automatic, so the fair score
  expert  a box around expert 1's trace on its largest slice: the best case, what MedSAM2's
          outline is worth once it knows exactly where to look. It borrows expert 1's
          location, so its score against expert 2 is the fairer one
Two views:
  full    the whole slice, scaled to MedSAM2's 512 x 512 input
  zoom    a square window 4x the box's longer side (at least 64 px) around the prompt box,
          the same window on every slice, scaled to 512 x 512: the hypothalamus fills ~8x
          more of the input (Dr. Ankit's worry that it is too small for the mask decoder)

Data: the 14 thick test rats (rcnn_data_14, scored with rcnn.evaluate alongside every other
model) and the 5 high-resolution rats in both versions (rcnn_data_hr, scored in
rcnn_out_hr/HIGHRES.md). The high-resolution stage 1 boxes come from rcnn.highres, so that
runs first.

Runs in its own environment (.venv-medsam2), so the R-CNN environment is untouched:
  HYPO_TEST=14 .venv-medsam2/bin/python -m rcnn.medsam2 [--data thick|highres]   (--smoke: rat 1, auto + full)
"""

import argparse
import pickle

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from .common import ROOT, load_split, mask_to_box, out_root, pick_device

MEDSAM2 = ROOT / "third_party" / "MedSAM2"
CHECKPOINT = MEDSAM2 / "checkpoints" / "MedSAM2_latest.pt"
CONFIG = "configs/sam2.1_hiera_t512.yaml"
INPUT = 512
ZOOM, MIN_SIDE, MARGIN = 4.0, 64, 2
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
PROMPTS = {"auto": "the atlas-trained detector's box", "expert": "a box around expert 1's trace"}
VIEWS = {"full": "whole slice", "zoom": "zoomed window"}


def frames(vol, device):
    """(D, H, W) uint8 -> (D, 3, 512, 512) normalised float tensor, as the authors prepare CT."""
    out = np.stack([np.array(Image.fromarray(s).convert("RGB").resize((INPUT, INPUT))).transpose(2, 0, 1) for s in vol])
    t = torch.from_numpy(out).float().div(255.0).to(device)
    return (t - torch.tensor(MEAN, device=device)[:, None, None]) / torch.tensor(STD, device=device)[:, None, None]


def segment(predictor, vol, key, box, device):
    """MedSAM2 on one stack: box on slice key, propagated both ways; largest 3D piece kept."""
    d, h, w = vol.shape
    seg = np.zeros(vol.shape, bool)
    with torch.inference_mode():
        state = predictor.init_state(frames(vol, device), h, w)
        for reverse in (False, True):
            predictor.add_new_points_or_box(inference_state=state, frame_idx=key, obj_id=1, box=np.asarray(box, np.float32))
            for z, _, logits in predictor.propagate_in_video(state, reverse=reverse):
                seg[z] |= (logits[0, 0] > 0).cpu().numpy()
            predictor.reset_state(state)
    if seg.any():
        lab, _ = ndimage.label(seg)
        seg = lab == np.argmax(np.bincount(lab.ravel())[1:]) + 1
    return seg


def square(box):
    x0, y0, x1, y1 = box
    side = int(round(max(MIN_SIDE, ZOOM * max(x1 - x0, y1 - y0))))
    return int(round((x0 + x1) / 2 - side / 2)), int(round((y0 + y1) / 2 - side / 2)), side


def run_rat(predictor, vol, key, box, view, device):
    if view == "full":
        return segment(predictor, vol, key, box, device)
    x, y, side = square(box)
    padded = np.pad(vol, ((0, 0), (side, side), (side, side)))
    crop = padded[:, y + side:y + 2 * side, x + side:x + 2 * side]
    seg = segment(predictor, crop, key, [box[0] - x, box[1] - y, box[2] - x, box[3] - y], device)
    full = np.zeros((vol.shape[0], vol.shape[1] + 2 * side, vol.shape[2] + 2 * side), bool)
    full[:, y + side:y + 2 * side, x + side:x + 2 * side] = seg
    return full[:, side:-side, side:-side]


def prompt_box(prompt, idx, data, s1):
    """(position in the stack, box) for one rat."""
    if prompt == "expert":
        areas = [data["masks"][i].sum() for i in idx]
        j = int(np.argmax(areas))
        x0, y0, x1, y1 = mask_to_box(data["masks"][idx[j]])
        return j, [x0 - MARGIN, y0 - MARGIN, x1 - 1 + MARGIN, y1 - 1 + MARGIN]
    best = [(s1[str(data["files"][i])]["scores"].max() if len(s1[str(data["files"][i])]["scores"]) else -1, j)
            for j, i in enumerate(idx)]
    _, j = max(best)
    p = s1[str(data["files"][idx[j]])]
    return j, p["boxes"][int(np.argmax(p["scores"]))].tolist()


def run_dataset(predictor, data, s1, prompts, views, device, rats=None):
    """{(prompt, view): {slice index: mask}} over every rat."""
    out = {(p, v): {} for p in prompts for v in views}
    for n in sorted(set(data["subjects"].tolist())):
        if rats and n not in rats:
            continue
        idx = sorted(np.where(data["subjects"] == n)[0], key=lambda i: data["zs"][i])
        vol = data["images"][idx]
        for p in prompts:
            key, box = prompt_box(p, idx, data, s1)
            for v in views:
                seg = run_rat(predictor, vol, key, box, v, device)
                out[p, v].update({i: seg[j] for j, i in enumerate(idx)})
        print(f"  rat {n} done", flush=True)
    return out


def label(p, v):
    return f"MedSAM2 (no training), {PROMPTS[p]}, {VIEWS[v]}"


def main():
    from sam2.build_sam import build_sam2_video_predictor_npz

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", choices=["all", "thick", "highres"], default="all")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    device = pick_device(args.device)
    torch.manual_seed(2024)
    predictor = build_sam2_video_predictor_npz(CONFIG, str(CHECKPOINT), device=str(device))
    prompts, views = (["auto"], ["full"]) if args.smoke else (list(PROMPTS), list(VIEWS))

    if args.data != "highres":
        run_thick(predictor, prompts, views, device, args.smoke)
    if args.data != "thick" and not args.smoke:
        run_highres(predictor, prompts, views, device)


def run_thick(predictor, prompts, views, device, smoke):
    from .ensemble import write_run
    from .evaluate import score
    data = load_split("test")
    with open(out_root() / "faster_rcnn" / "predictions.pkl", "rb") as f:
        s1 = {p["file"]: p for p in pickle.load(f)}
    print("14 thick rats", flush=True)
    results = run_dataset(predictor, data, s1, prompts, views, device, rats=[1] if smoke else None)
    for (p, v), masks in results.items():
        rows, _ = score("mask_rcnn", {i: (mask_to_box(m), m) for i, m in masks.items()}, data, None, False)
        med = lambda k: np.median([r[k] for r in rows])
        print(f"{label(p, v)}: median IoU / Dice vs expert 1 {med('iou_vs_me'):.3f} / {med('dice_vs_me'):.3f}, "
              f"vs expert 2 {med('iou_vs_other'):.3f} / {med('dice_vs_other'):.3f}", flush=True)
        if not smoke:
            write_run(out_root(), f"medsam2_{p}_{v}", masks, data, {"model": "mask_rcnn", "prompt": p, "view": v, "label": label(p, v)})


def run_highres(predictor, prompts, views, device):
    from . import highres
    for version in highres.STEPS:
        stage1 = highres.OUT_HR / version / "stage1.pkl"
        if not stage1.exists():
            print(f"high-res {version}: skipped, run rcnn.highres first for its stage 1 boxes")
            continue
        hr = highres.load_hr(version)
        with open(stage1, "rb") as f:
            s1 = pickle.load(f)
        print(f"high-res {version}", flush=True)
        for (p, v), masks in run_dataset(predictor, hr, s1, prompts, views, device).items():
            highres.write_masks(version, f"medsam2_{p}_{v}", masks, hr, {"model": "medsam2", "prompt": p, "view": v, "label": label(p, v)})
    highres.summarize()


if __name__ == "__main__":
    main()
