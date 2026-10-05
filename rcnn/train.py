"""Trains one model on the atlas-labeled training rats, then predicts on the 9 hand-traced
test rats. Usage, from the repo root:

    python3 -m rcnn.train --model mask_rcnn

Same plain settings for every model, adapted from torchvision's detection fine-tuning
tutorial: SGD, lr 0.005, momentum 0.9, weight decay 5e-4, batch 2, 10 epochs, lr x0.1 for
the last 3 epochs, 100-iteration warmup. No augmentation: that's the "alterations" step.

Writes rcnn_out/<model>/model.pt, predictions.pkl and train.log.
"""

import argparse
import pickle
import time

import numpy as np
import torch

from . import models, rcnn_classic
from .common import MASK_MODELS, ROOT, SliceDataset, collate, load_split, pick_device

EPOCHS = 10
LR_DROP_EPOCH = 7
BATCH = 2
LR = 0.005
WARMUP_ITERS = 100


class Log:
    def __init__(self, path):
        self.f = open(path, "a")

    def __call__(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def to_device(images, targets, device):
    images = [im.to(device) for im in images]
    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
    return images, targets


def train_network(name, model, device, out_dir, log, epochs, max_iters, limit):
    ds = SliceDataset("train", with_masks=name in MASK_MODELS,
                      with_proposals=name == "fast_rcnn", limit=limit)
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=5e-4)
    total = max_iters or epochs * len(loader)
    drop_at = LR_DROP_EPOCH * len(loader)

    def lr_at(it):
        warm = min(1.0, (it + 1) / WARMUP_ITERS) * (1 - 0.001) + 0.001
        return LR * warm * (0.1 if it >= drop_at else 1.0)

    model.train()
    it, t0 = 0, time.time()
    log(f"training {name}: {len(ds)} slices, {total} iterations on {device}")
    while it < total:
        for batch in loader:
            images, targets = batch[0], batch[1]
            images, targets = to_device(images, targets, device)
            for g in opt.param_groups:
                g["lr"] = lr_at(it)
            if name == "fast_rcnn":
                losses = model(images, targets, [p.to(device) for p in batch[2]])
            else:
                losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss became {loss.item()} at iteration {it}: {losses}")
            opt.zero_grad()
            loss.backward()
            opt.step()
            it += 1
            if it % 20 == 0 or it == total:
                per_it = (time.time() - t0) / it
                parts = " ".join(f"{k}={v.item():.3f}" for k, v in losses.items())
                log(f"  iter {it}/{total}  loss={loss.item():.3f} ({parts})  "
                    f"{per_it:.2f}s/iter  ~{per_it * (total - it) / 60:.0f} min left")
            if it >= total:
                break
    torch.save(model.state_dict(), out_dir / "model.pt")



@torch.no_grad()
def predict_network(name, model, device, limit):
    data = load_split("test")
    ds = SliceDataset("test", with_proposals=name == "fast_rcnn", limit=limit)
    model.eval()
    preds = []
    for i in range(len(ds)):
        item = ds[i]
        img = item[0].to(device)
        out = model([img], None, [item[2].to(device)])[0] if name == "fast_rcnn" else model([img])[0]
        p = {"file": str(data["files"][i]), "subject": int(data["subjects"][i]), "z": int(data["zs"][i]),
             "boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy(), "masks": None}
        if "masks" in out:
            top = out["masks"][:5, 0].cpu().numpy()
            p["masks"] = (top * 255).astype(np.uint8)
        preds.append(p)
    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["rcnn", *models.BUILDERS])
    ap.add_argument("--device", default=None)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--smoke", action="store_true", help="tiny run to check the model works end to end")
    args = ap.parse_args()

    out_root = ROOT / ("rcnn_out_smoke" if args.smoke else "rcnn_out")
    out_dir = out_root / args.model
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / "train.log")
    device = pick_device(args.device)
    torch.manual_seed(0)
    limit_train, limit_test, max_iters = (16, 12, 10) if args.smoke else (None, None, None)
    t0 = time.time()

    if args.model == "rcnn":
        extractor, scaler, svm, reg = rcnn_classic.train(out_dir, device, limit=limit_train, log=log)
        preds = rcnn_classic.predict(extractor, scaler, svm, reg, limit=limit_test)
    else:
        model = models.build(args.model).to(device)
        train_network(args.model, model, device, out_dir, log, args.epochs, max_iters, limit_train)
        preds = predict_network(args.model, model, device, limit_test)

    with open(out_dir / "predictions.pkl", "wb") as f:
        pickle.dump(preds, f)
    log(f"done: {len(preds)} test slices predicted, {(time.time() - t0) / 60:.1f} min total")


if __name__ == "__main__":
    main()
