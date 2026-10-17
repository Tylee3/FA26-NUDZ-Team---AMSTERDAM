"""Box first, then mask on a zoomed view (Karina's idea): a box detector finds the
hypothalamus, a window about twice the box is cut out and scaled up to 256x256, and a Mask
R-CNN trained only on such windows draws the outline there. The result is pasted back
into the full slice.

Stage 1 is the Faster R-CNN already trained in rcnn_out/faster_rcnn (it found the most
slices, 39/45, with no false alarms); its saved test predictions are reused.
Stage 2 trains on windows around the training labels with random shifts and sizes, so it
learns to cope with stage 1's imperfect boxes, plus windows from slices without the
structure, so it can answer "nothing here". Same training settings as every other model.

One expectation to state up front: Mask R-CNN always draws its outline on a 28x28 grid per
box, so zooming adds detail to what the network sees rather than to the outline's grid.

Usage (from the repo root):  python3 -m rcnn.zoom          (--smoke for a tiny check)
Writes rcnn_out/zoom_mask_rcnn/ in the usual format, so rcnn.evaluate scores it.
"""

import argparse
import json
import pickle
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import models
from .common import ROOT, image_tensor, load_split, mask_to_box, pick_device
from .train import Log, train_network

CROP = 256
CONTEXT = 2.0
MIN_SIDE = 48
STAGE1 = "faster_rcnn"
STAGE1_THRESHOLD = 0.5
STAGE2_THRESHOLD = 0.5


def window(box, scale=CONTEXT, shift=(0.0, 0.0)):
    x0, y0, x1, y1 = box
    side = max(MIN_SIDE, scale * max(x1 - x0, y1 - y0))
    cx, cy = (x0 + x1) / 2 + shift[0], (y0 + y1) / 2 + shift[1]
    side = int(round(side))
    return int(round(cx - side / 2)), int(round(cy - side / 2)), side


def cut(arr, win, mode):
    """Square window out of a 2D array (zero-padded at the edges), scaled to CROP x CROP."""
    x, y, side = win
    pad = side
    padded = np.pad(arr, pad)
    patch = padded[y + pad:y + pad + side, x + pad:x + pad + side].astype(np.float32)
    t = torch.from_numpy(patch)[None, None]
    out = F.interpolate(t, size=(CROP, CROP), mode=mode, **({"align_corners": False} if mode == "bilinear" else {}))
    return out[0, 0].numpy()


def paste(prob, win, shape):
    """CROP x CROP probability map back into a full-size slice."""
    x, y, side = win
    up = F.interpolate(torch.from_numpy(prob)[None, None], size=(side, side), mode="bilinear",
                       align_corners=False)[0, 0].numpy()
    full = np.zeros(shape, dtype=np.float32)
    h, w = shape
    sx0, sy0 = max(0, -x), max(0, -y)
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(w, x + side), min(h, y + side)
    full[dy0:dy1, dx0:dx1] = up[sy0:sy0 + (dy1 - dy0), sx0:sx0 + (dx1 - dx0)]
    return full


class ZoomDataset(torch.utils.data.Dataset):
    def __init__(self, limit=None):
        self.data = load_split("train")
        n = len(self.data["files"]) if limit is None else limit
        self.items = list(range(n))
        boxes = [mask_to_box(m) for m in self.data["masks"][:n]]
        known = np.array([b for b in boxes if b is not None])
        self.typical = known.mean(axis=0)
        self.boxes = boxes

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        gray, mask = self.data["images"][i], self.data["masks"][i]
        box = self.boxes[i] if self.boxes[i] is not None else self.typical
        size = max(box[2] - box[0], box[3] - box[1])
        scale = float(torch.empty(()).uniform_(1.6, 2.4))
        shift = (float(torch.empty(()).uniform_(-0.25, 0.25)) * size, float(torch.empty(()).uniform_(-0.25, 0.25)) * size)
        win = window(box, scale, shift)
        img = cut(gray, win, "bilinear").clip(0, 255).astype(np.uint8)
        m = cut(mask.astype(np.uint8), win, "nearest") > 0.5
        target = {"image_id": torch.tensor(i)}
        b = mask_to_box(m)
        if b is None:
            target.update(boxes=torch.zeros((0, 4)), labels=torch.zeros((0,), dtype=torch.int64),
                          masks=torch.zeros((0, CROP, CROP), dtype=torch.uint8))
        else:
            target.update(boxes=torch.from_numpy(b[None]), labels=torch.ones((1,), dtype=torch.int64),
                          masks=torch.from_numpy(m[None].astype(np.uint8)))
        return image_tensor(img), target


@torch.no_grad()
def predict(model, device, limit=None):
    data = load_split("test")
    with open(ROOT / "rcnn_out" / STAGE1 / "predictions.pkl", "rb") as f:
        stage1 = {p["file"]: p for p in pickle.load(f)}
    model.eval()
    n = len(data["files"]) if limit is None else limit
    preds = []
    for i in range(n):
        f, gray = str(data["files"][i]), data["images"][i]
        p1 = stage1[f]
        out = {"file": f, "subject": int(data["subjects"][i]), "z": int(data["zs"][i]),
               "boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "masks": None}
        if len(p1["scores"]) and p1["scores"].max() >= STAGE1_THRESHOLD:
            win = window(p1["boxes"][int(np.argmax(p1["scores"]))])
            crop = cut(gray, win, "bilinear").clip(0, 255).astype(np.uint8)
            det = model([image_tensor(crop).to(device)])[0]
            if len(det["scores"]) and det["scores"].max() >= STAGE2_THRESHOLD:
                k = int(det["scores"].argmax())
                full = paste(det["masks"][k, 0].cpu().numpy(), win, gray.shape)
                b = mask_to_box(full >= 0.5)
                if b is not None:
                    out.update(boxes=b[None], scores=np.array([float(det["scores"][k])], np.float32),
                               masks=(full[None] * 255).clip(0, 255).astype(np.uint8))
        preds.append(out)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    out_dir = ROOT / ("rcnn_out_smoke" if args.smoke else "rcnn_out") / "zoom_mask_rcnn"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"model": "mask_rcnn", "zoom": True, "stage1": STAGE1, "crop": CROP, "context": CONTEXT}, indent=2))
    log = Log(out_dir / "train.log")
    device = pick_device(args.device)
    torch.manual_seed(0)
    limit_train, limit_test, max_iters = (16, 12, 10) if args.smoke else (None, None, None)
    t0 = time.time()
    model = models.build("mask_rcnn").to(device)
    train_network("mask_rcnn", model, device, out_dir, log, 10, max_iters, None, False, False,
                  dataset=ZoomDataset(limit_train))
    preds = predict(model, device, limit_test)
    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    log(f"done: {len(preds)} test slices predicted, {(time.time() - t0) / 60:.1f} min total")


if __name__ == "__main__":
    main()
