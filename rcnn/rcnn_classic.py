"""The original R-CNN (Girshick et al. 2014), which is a pipeline rather than one network:

1. Selective Search proposes ~200-1000 candidate regions per slice.
2. Each candidate is cropped (with 16px of context, as in the paper), warped to 224x224,
   and run through the CNN on its own, giving one 2048-d feature vector per region.
   This repeated per-region pass is why R-CNN is slow and what Fast R-CNN fixed.
3. A linear SVM scores each region as hypothalamus vs not (positives = ground-truth boxes,
   negatives = candidates overlapping them by < 0.3 IoU, as in the paper).
4. A ridge regressor nudges each kept box toward the true one (trained on candidates
   overlapping the true box by >= 0.6, lambda = 1000, as in the paper).

Simplifications vs the paper, stated so they're not mistaken for the method: the CNN is
not fine-tuned on regions (frozen COCO-pretrained ResNet-50 features, which the paper also
reports as a variant), negatives are randomly sampled rather than hard-mined, and test
time uses the first 300 Selective Search candidates per slice.
"""

import pickle

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torchvision.ops import box_iou, nms, roi_align

from .common import load_split, mask_to_box
from .models import faster_base
from .selective_search import proposals_for

WARP = 224
CONTEXT = 16
TEST_PROPOSALS = 300
NEGATIVES_PER_SLICE = 20


class RegionFeatures:
    def __init__(self, device):
        body = faster_base().backbone.body
        self.net = nn.Sequential(body.conv1, body.bn1, body.relu, body.maxpool, body.layer1,
                                 body.layer2, body.layer3, body.layer4,
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten()).eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    @torch.no_grad()
    def __call__(self, gray, boxes, batch=64):
        if len(boxes) == 0:
            return np.zeros((0, 2048), dtype=np.float32)
        img = torch.from_numpy(gray).float().div(255).to(self.device)[None, None].repeat(1, 3, 1, 1)
        b = torch.as_tensor(boxes, dtype=torch.float32, device=self.device)
        pad_x = CONTEXT * (b[:, 2] - b[:, 0]) / (WARP - 2 * CONTEXT)
        pad_y = CONTEXT * (b[:, 3] - b[:, 1]) / (WARP - 2 * CONTEXT)
        padded = torch.stack([b[:, 0] - pad_x, b[:, 1] - pad_y, b[:, 2] + pad_x, b[:, 3] + pad_y], 1)
        feats = []
        for chunk in padded.split(batch):
            rois = torch.cat([torch.zeros(len(chunk), 1, device=self.device), chunk], 1)
            crops = roi_align(img, rois, (WARP, WARP), spatial_scale=1.0, sampling_ratio=2, aligned=True)
            feats.append(self.net((crops - self.mean) / self.std).cpu())
        return torch.cat(feats).numpy()


def encode(gt, props):
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    px, py = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    gw, gh = gt[2] - gt[0], gt[3] - gt[1]
    gx, gy = gt[0] + gw / 2, gt[1] + gh / 2
    return np.stack([(gx - px) / pw, (gy - py) / ph, np.log(gw / pw), np.log(gh / ph)], 1)


def decode(deltas, props):
    pw, ph = props[:, 2] - props[:, 0], props[:, 3] - props[:, 1]
    px, py = props[:, 0] + pw / 2, props[:, 1] + ph / 2
    cx, cy = px + deltas[:, 0] * pw, py + deltas[:, 1] * ph
    w, h = pw * np.exp(np.clip(deltas[:, 2], -4, 4)), ph * np.exp(np.clip(deltas[:, 3], -4, 4))
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def train(out_dir, device, limit=None, log=print):
    data = load_split("train")
    n = len(data["files"]) if limit is None else limit
    extractor = RegionFeatures(device)
    rng = np.random.default_rng(0)
    cls_x, cls_y, reg_x, reg_y = [], [], [], []

    for i in range(n):
        gray, gt = data["images"][i], mask_to_box(data["masks"][i])
        props = proposals_for(str(data["files"][i]), gray)
        if gt is None:
            neg = props[rng.choice(len(props), min(NEGATIVES_PER_SLICE, len(props)), replace=False)]
            cls_x.append(extractor(gray, neg))
            cls_y.append(np.zeros(len(neg)))
            continue
        ious = box_iou(torch.from_numpy(props), torch.from_numpy(gt[None]))[:, 0].numpy()
        neg_pool = np.where(ious < 0.3)[0]
        neg = props[rng.choice(neg_pool, min(NEGATIVES_PER_SLICE, len(neg_pool)), replace=False)]
        reg = props[ious >= 0.6]
        feats = extractor(gray, np.concatenate([gt[None], neg, reg]))
        cls_x.append(feats[: 1 + len(neg)])
        cls_y.append(np.r_[1.0, np.zeros(len(neg))])
        if len(reg):
            reg_x.append(feats[1 + len(neg):])
            reg_y.append(encode(gt, reg))
        if (i + 1) % 100 == 0:
            log(f"  features: {i + 1}/{n} slices")

    cls_x, cls_y = np.concatenate(cls_x), np.concatenate(cls_y)
    scaler = StandardScaler().fit(cls_x)
    svm = LinearSVC(C=0.01, class_weight="balanced", max_iter=5000).fit(scaler.transform(cls_x), cls_y)
    regressor = None
    if reg_x:
        reg_x, reg_y = np.concatenate(reg_x), np.concatenate(reg_y)
        regressor = Ridge(alpha=1000.0).fit(scaler.transform(reg_x), reg_y)
    log(f"  SVM: {int(cls_y.sum())} positives, {int((cls_y == 0).sum())} negatives; "
        f"box regressor: {0 if regressor is None else len(reg_x)} examples")
    with open(out_dir / "rcnn_classic.pkl", "wb") as f:
        pickle.dump({"scaler": scaler, "svm": svm, "regressor": regressor}, f)
    return extractor, scaler, svm, regressor


def predict(extractor, scaler, svm, regressor, limit=None):
    data = load_split("test")
    n = len(data["files"]) if limit is None else limit
    preds = []
    for i in range(n):
        gray = data["images"][i]
        props = proposals_for(str(data["files"][i]), gray)[:TEST_PROPOSALS]
        x = scaler.transform(extractor(gray, props))
        scores = svm.decision_function(x)
        boxes = decode(regressor.predict(x), props) if regressor is not None else props
        h, w = gray.shape
        boxes = np.clip(boxes, 0, [w, h, w, h]).astype(np.float32)
        keep = nms(torch.from_numpy(boxes), torch.from_numpy(scores.astype(np.float32)), 0.3)[:100].numpy()
        preds.append({"file": str(data["files"][i]), "subject": int(data["subjects"][i]),
                      "z": int(data["zs"][i]), "boxes": boxes[keep], "scores": scores[keep].astype(np.float32),
                      "masks": None})
    return preds
