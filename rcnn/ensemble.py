"""Combining the best methods, with the recipe fixed in advance (so the 9 test rats are not
used to pick it):

  vote          a pixel is hypothalamus when at least 3 of these 5 say so: the Waxholm atlas,
                zoomed Mask R-CNN, zoomed Mask R-CNN + augmentation, Mesh R-CNN,
                3D U-Net + augmentation.
  mirror pass   each learned model also looks at the left-right mirrored image; its two
                probability maps (mirrored back) are averaged before thresholding at 0.5.
                Applied to the zoomed models (on stage 2: the same stage-1 box, the window
                mirrored) and the 3D U-Net (its 128x128 window mirrored). Not to Mesh R-CNN,
                whose output is a yes/no mesh rather than a probability, nor to the atlas.

Writes, each in the usual format so rcnn.evaluate scores them:
  rcnn_out/ensemble_vote/                   the vote of the 5 as they are
  rcnn_out/ensemble_vote_mirror/            the vote with the mirror pass on its 3 members
  rcnn_out/<member>_mirror/                 each mirrored member on its own

Usage (from the repo root):  python3 -m rcnn.ensemble   (--smoke: first test rat only)
"""

import argparse
import json
import pickle

import numpy as np
import torch

from . import models
from .common import ROOT, image_tensor, load_split, mask_to_box, pick_device
from .evaluate import atlas_predictions, slice_predictions
from .unet3d import UNet3D, X0, X1, Y0, Y1, volumes
from .zoom import STAGE1, STAGE1_THRESHOLD, STAGE2_THRESHOLD, cut, paste, window

OUT = ROOT / "rcnn_out"
MEMBERS = [("zoom_mask_rcnn", "mask_rcnn"), ("zoom_mask_rcnn_augment", "mask_rcnn"), ("mesh_rcnn", "mesh_rcnn"),
           ("unet3d_augment", "unet3d")]
VOTES_NEEDED = 3


def stored_masks(test):
    """Per member: test slice index -> bool mask, from its saved predictions."""
    out = {"atlas": {i: m for i, (_, m) in atlas_predictions(test, False).items()}}
    for run, base in MEMBERS:
        with open(OUT / run / "predictions.pkl", "rb") as f:
            per_slice = slice_predictions(base, pickle.load(f), test)
        out[run] = {i: (m if m is not None else np.zeros(test["masks"][i].shape, bool)) for i, (_, m) in per_slice.items()}
    return out


@torch.no_grad()
def zoom_mirror(run, test, device, n):
    """Zoomed Mask R-CNN probabilities, plain and mirrored, averaged. Same stage-1 box."""
    model = models.build("mask_rcnn")
    model.load_state_dict(torch.load(OUT / run / "model.pt", map_location="cpu"))
    model.to(device).eval()
    with open(OUT / STAGE1 / "predictions.pkl", "rb") as f:
        stage1 = {p["file"]: p for p in pickle.load(f)}
    out = {}
    for i in range(n):
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
        out[i] = prob >= 0.5
    return out


@torch.no_grad()
def unet_mirror(run, test, device, n):
    model = UNet3D()
    model.load_state_dict(torch.load(OUT / run / "model.pt", map_location="cpu"))
    model.to(device).eval()
    lookup = {str(f): i for i, f in enumerate(test["files"])}
    out = {}
    for vol in volumes("test"):
        idx = [lookup[f] for f in vol["files"]]
        if any(i >= n for i in idx):
            continue
        x = torch.from_numpy(vol["image"][:, Y0:Y1, X0:X1]).float().div(255)[None, None].to(device)
        prob = (torch.sigmoid(model(x)) + torch.sigmoid(model(x.flip(-1))).flip(-1))[0, 0].cpu().numpy() / 2
        full = np.zeros(vol["image"].shape, np.float32)
        full[:, Y0:Y1, X0:X1] = prob
        for z, i in enumerate(idx):
            out[i] = full[z] >= 0.5
    return out


def vote(member_masks, test):
    out = {}
    for i in member_masks["atlas"]:
        if all(i in m for m in member_masks.values()):
            out[i] = sum(m[i].astype(np.uint8) for m in member_masks.values()) >= VOTES_NEEDED
    return out


def write_run(root, name, masks, test, config):
    d = root / name
    d.mkdir(parents=True, exist_ok=True)
    preds = []
    for i, m in sorted(masks.items()):
        p = {"file": str(test["files"][i]), "subject": int(test["subjects"][i]), "z": int(test["zs"][i]),
             "boxes": np.zeros((0, 4), np.float32), "scores": np.zeros((0,), np.float32), "masks": None}
        b = mask_to_box(m)
        if b is not None:
            p.update(boxes=b[None], scores=np.array([1.0], np.float32), masks=m[None].astype(np.uint8) * 255)
        preds.append(p)
    with open(d / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    (d / "config.json").write_text(json.dumps(config, indent=2))
    print(f"wrote {d} ({sum(len(p['scores']) for p in preds)} slices with an outline)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    test = load_split("test")
    n = 12 if args.smoke else len(test["files"])
    out_root = ROOT / "rcnn_out_smoke" if args.smoke else OUT
    device = pick_device(args.device)
    members = {k: {i: m for i, m in v.items() if i < n} for k, v in stored_masks(test).items()}
    names = list(members)
    mirrored = dict(members)
    for run, base in MEMBERS:
        if base == "mesh_rcnn":
            continue
        mirrored[run] = (zoom_mirror if base == "mask_rcnn" else unet_mirror)(run, test, device, n)
        config = json.loads((OUT / run / "config.json").read_text())
        write_run(out_root, f"{run}_mirror", mirrored[run], test, {**config, "mirror": True})
    common = {"model": "ensemble", "members": names, "votes_needed": VOTES_NEEDED}
    write_run(out_root, "ensemble_vote", vote(members, test), test, common)
    write_run(out_root, "ensemble_vote_mirror", vote(mirrored, test), test, {**common, "mirror": True})


if __name__ == "__main__":
    main()
