"""3D U-Net: segments a whole 12-slice scan at once, so every prediction sees the slices
above and below it, instead of one 2D slice at a time like the R-CNN family.

Input: each rat's 12 slices, cut to a fixed 128x128 in-plane window (x 66-194, y 98-226 of
the 256x256 slice). The window comes from the training labels alone (every training
structure lies in x 95-165, y 117-207) plus a margin, and it makes training about 4x faster.
Predictions outside it are zero.

Network: 4 levels (16-32-64-128 channels), 3x3x3 convolutions + instance norm. The first two
poolings shrink only in-plane, because the slices are 10x thicker (1 mm) than the in-plane
pixels (0.1 mm). Loss = binary cross-entropy + soft Dice. Adam, lr 1e-3, x0.1 for the last
30% of training, 150 passes over the 53 training rats, batch 2. --augment applies the same
flips/rotations/shifts/zoom/brightness as the R-CNN runs, identically on every slice of a
volume. Training labels are the atlas outlines (rcnn_data/train.npz), like every other model.

Usage (from the repo root):  python3 -m rcnn.unet3d [--augment] [--smoke]
Writes rcnn_out/unet3d[_augment]/ in the usual format, so rcnn.evaluate scores it.
"""

import argparse
import json
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .common import ROOT, augment, load_split, mask_to_box, pick_device
from .train import Log

Y0, Y1, X0, X1 = 98, 226, 66, 194
EPOCHS = 150
BATCH = 2
LR = 1e-3


def volumes(split):
    d = load_split(split)
    out = []
    for n in sorted(set(d["subjects"].tolist())):
        idx = sorted(np.where(d["subjects"] == n)[0], key=lambda i: d["zs"][i])
        out.append({"rat": int(n), "idx": idx, "image": np.stack([d["images"][i] for i in idx]),
                    "mask": np.stack([d["masks"][i] for i in idx]), "files": [str(d["files"][i]) for i in idx]})
    return out


def block(cin, cout):
    return nn.Sequential(nn.Conv3d(cin, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01),
                         nn.Conv3d(cout, cout, 3, padding=1), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.01))


class UNet3D(nn.Module):
    POOLS = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]

    def __init__(self, c=16):
        super().__init__()
        chans = [c, 2 * c, 4 * c, 8 * c]
        self.enc = nn.ModuleList([block(1, chans[0]), block(chans[0], chans[1]), block(chans[1], chans[2]),
                                  block(chans[2], chans[3])])
        self.up = nn.ModuleList([nn.ConvTranspose3d(chans[i + 1], chans[i + 1], self.POOLS[i], stride=self.POOLS[i])
                                 for i in range(3)])
        self.dec = nn.ModuleList([block(chans[i + 1] + chans[i], chans[i]) for i in range(3)])
        self.head = nn.Conv3d(chans[0], 1, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < 3:
                skips.append(x)
                x = F.max_pool3d(x, self.POOLS[i])
        for i in reversed(range(3)):
            x = self.dec[i](torch.cat([self.up[i](x), skips[i]], dim=1))
        return self.head(x)


def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3, 4))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4)) + 1)
    return bce + dice.mean()


def sample(vol, do_augment):
    img = torch.from_numpy(vol["image"][:, Y0:Y1, X0:X1]).float().div(255)
    mask = vol["mask"][:, Y0:Y1, X0:X1]
    if do_augment:
        img, mask = augment(img, mask)
    return img[None], torch.from_numpy(mask.astype(np.float32))[None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    run = "unet3d" + ("_augment" if args.augment else "")
    out_dir = ROOT / ("rcnn_out_smoke" if args.smoke else "rcnn_out") / run
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(
        {"model": "unet3d", "augment": args.augment, "epochs": EPOCHS, "window_yx": [Y0, Y1, X0, X1]}, indent=2))
    log = Log(out_dir / "train.log")
    device = pick_device(args.device)
    torch.manual_seed(0)
    t0 = time.time()

    train = volumes("train")
    if args.smoke:
        train = train[:4]
    model = UNet3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    per_epoch = (len(train) + BATCH - 1) // BATCH
    total = 10 if args.smoke else EPOCHS * per_epoch
    drop_at = int(0.7 * total)
    log(f"training {run}: {len(train)} volumes, {total} iterations on {device}")
    it = 0
    model.train()
    while it < total:
        order = torch.randperm(len(train)).tolist()
        for k in range(0, len(order), BATCH):
            pairs = [sample(train[j], args.augment) for j in order[k:k + BATCH]]
            x = torch.stack([p[0] for p in pairs]).to(device)
            y = torch.stack([p[1] for p in pairs]).to(device)
            for g in opt.param_groups:
                g["lr"] = LR * (0.1 if it >= drop_at else 1.0)
            loss = loss_fn(model(x), y)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at iteration {it}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % 20 == 0 or it == total:
                per_it = (time.time() - t0) / it
                log(f"  iter {it}/{total}  loss={loss.item():.3f}  {per_it:.2f}s/iter  ~{per_it * (total - it) / 60:.0f} min left")
            if it >= total:
                break
    torch.save(model.state_dict(), out_dir / "model.pt")

    model.eval()
    preds = []
    with torch.no_grad():
        for vol in volumes("test")[: (1 if args.smoke else None)]:
            x = torch.from_numpy(vol["image"][:, Y0:Y1, X0:X1]).float().div(255)[None, None].to(device)
            prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
            full = np.zeros(vol["image"].shape, dtype=np.float32)
            full[:, Y0:Y1, X0:X1] = prob
            for z, f in enumerate(vol["files"]):
                b = mask_to_box(full[z] >= 0.5)
                p = {"file": f, "subject": vol["rat"], "z": z, "boxes": np.zeros((0, 4), np.float32),
                     "scores": np.zeros((0,), np.float32), "masks": None}
                if b is not None:
                    p.update(boxes=b[None], scores=np.array([1.0], np.float32),
                             masks=(full[z][None] * 255).astype(np.uint8))
                preds.append(p)
    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    log(f"done: {len(preds)} test slices predicted, {(time.time() - t0) / 60:.1f} min total")


if __name__ == "__main__":
    main()
