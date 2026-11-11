"""Two cheap add-ons to the hand-trace fine-tune, with the recipe fixed in advance:

  mirror pass   the fine-tuned model also looks at the left-right mirrored window; the two
                probability maps (mirrored back) are averaged before thresholding at 0.5,
                as in rcnn/ensemble.py.
  average of 3  per fold, the models fine-tuned on labeler 1, labeler 2 and both, each with
                the mirror pass; their probabilities are averaged, then thresholded at 0.5.

Every rat is still scored only by the fold models that never saw it. Needs finetune_cv's
labeler1, labeler2 and both folds (python3 -m rcnn.finetune_cv, then --both).

Writes, each in the usual format so rcnn.evaluate scores them:
  <out>/finetune_both_mirror/    the model fine-tuned on both labelers, mirror pass
  <out>/finetune_avg3_mirror/    the average of the 3 fine-tuned models, mirror pass

Usage (from the repo root):  HYPO_TEST=14 python3 -m rcnn.finetune_combine   (--smoke: first fold, first test rat)
"""

import argparse
import pickle

import numpy as np
import torch

from . import models
from .common import image_tensor, load_split, out_root, pick_device
from .ensemble import write_run
from .finetune_cv import FOLDS
from .zoom import STAGE1, STAGE1_THRESHOLD, STAGE2_THRESHOLD, cut, paste, window

MEMBERS = ["labeler1", "labeler2", "both"]


@torch.no_grad()
def mirrored_probs(model_path, stage1, test, idx, device):
    """Test slice index -> probability map from one zoomed model, plain and mirrored, averaged."""
    model = models.build("mask_rcnn")
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device).eval()
    out = {}
    for i in idx:
        gray = test["images"][i]
        p1 = stage1[str(test["files"][i])]
        prob = np.zeros(gray.shape, np.float32)
        if len(p1["scores"]) and p1["scores"].max() >= STAGE1_THRESHOLD:
            win = window(p1["boxes"][int(np.argmax(p1["scores"]))])
            crop = cut(gray, win, "bilinear").clip(0, 255).astype(np.uint8)
            for mirrored in (False, True):
                c = crop[:, ::-1].copy() if mirrored else crop
                det = model([image_tensor(c).to(device)])[0]
                if len(det["scores"]) and det["scores"].max() >= STAGE2_THRESHOLD:
                    m = det["masks"][int(det["scores"].argmax()), 0].cpu().numpy()
                    prob += paste(m[:, ::-1].copy() if mirrored else m, win, gray.shape) / 2
        out[i] = prob
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    root = out_root(args.smoke)
    test = load_split("test")
    device = pick_device(args.device)
    with open(root / STAGE1 / "predictions.pkl", "rb") as f:
        stage1 = {p["file"]: p for p in pickle.load(f)}
    n = 12 if args.smoke else len(test["files"])
    both, avg = {}, {}
    for k, held_out in enumerate(FOLDS[:1] if args.smoke else FOLDS):
        idx = [i for i in range(n) if int(test["subjects"][i]) in held_out]
        probs = {}
        for member in MEMBERS:
            path = root / "finetune_cv" / member / f"fold{k}" / "model.pt"
            if path.exists():
                probs[member] = mirrored_probs(path, stage1, test, idx, device)
            print(f"fold {k}: {member} {'done' if member in probs else 'missing, left out'}", flush=True)
        if "both" not in probs:
            raise SystemExit(f"fold {k} has no 'both' model: run python3 -m rcnn.finetune_cv --both first")
        for i in idx:
            both[i] = probs["both"][i] >= 0.5
            avg[i] = np.mean([p[i] for p in probs.values()], axis=0) >= 0.5
    common = {"model": "mask_rcnn", "zoom": True, "mirror": True, "folds": FOLDS}
    write_run(root, "finetune_both_mirror", both, test,
              {**common, "label": "Mask R-CNN + zoom, fine-tuned on both labelers' traces + mirror pass (3-fold)"})
    write_run(root, "finetune_avg3_mirror", avg, test,
              {**common, "members": MEMBERS,
               "label": "Average of the 3 fine-tuned models (labeler 1, labeler 2, both) + mirror pass (3-fold)"})


if __name__ == "__main__":
    main()
