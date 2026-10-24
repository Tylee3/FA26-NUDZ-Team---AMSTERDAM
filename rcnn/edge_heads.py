"""Edge-focused mask heads (a teammate's three ideas from the Notion to-do list), shared by
Cascade Mask R-CNN and the zoomed Mask R-CNN so both get identical versions.

Every variant keeps Mask R-CNN's mask head (4 convolutions on 14x14 RoI features, then a 2x
upsampling to a 28x28 outline) and adds one thing:

  plain       nothing added: the baseline, written the same way as the variants.
  edge_loss   multi-task edge loss (after SAM-BoMu / ECASeg): a second 1x1 output on the 28x28
              features predicts the outline's border, trained against a border map made by
              running a Sobel filter over the target mask. Edge loss = BCE + Dice, weight 1.
  aspp        an ASPP block (after DeepLab-v3+) before the mask head: parallel 3x3 convolutions
              dilated at rates 1, 2 and 3, a 1x1 convolution and a global-average branch,
              merged back to 256 channels. The rates are scaled down from DeepLab's 6/12/18
              because this head sees a 14x14 grid, where a rate of 18 would only sample padding.
              Added as a residual whose last layer starts at zero, so training starts from the
              COCO-pretrained head unchanged.
  gated_edge  a gated edge branch (after ECASeg / SAM-EG): a thin parallel branch (2 x 64-channel
              convolutions) on the same RoI features predicts the border (trained with the same
              edge loss) and produces a gate that scales the mask features before the final
              upsampling: features x (1 + tanh(gate)). The gate starts at zero (no change).

Edge target: the 28x28 target mask, binarized, through a Sobel filter; every pixel with a
non-zero gradient is border (a band about 2 pixels wide at 28x28).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss, project_masks_on_boxes
from torchvision.ops import MultiScaleRoIAlign

VARIANTS = ["plain", "edge_loss", "aspp", "gated_edge"]
EDGE_WEIGHT = 1.0
ASPP_RATES = (1, 2, 3)
SOBEL_X = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])


def edge_targets(masks):
    """(N, M, M) soft target masks -> (N, M, M) border maps in {0, 1}."""
    m = (masks >= 0.5).float()[:, None]
    kx = SOBEL_X.to(m)[None, None]
    gx = F.conv2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), kx)
    gy = F.conv2d(F.pad(m, (1, 1, 1, 1), mode="replicate"), kx.transpose(2, 3))
    return ((gx.abs() + gy.abs()) > 0).float()[:, 0]


def edge_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return bce + dice


class ASPP(nn.Module):
    def __init__(self, ch=256):
        super().__init__()
        self.branches = nn.ModuleList([nn.Conv2d(ch, ch, 1)] +
                                      [nn.Conv2d(ch, ch, 3, padding=r, dilation=r) for r in ASPP_RATES])
        self.pooled = nn.Conv2d(ch, ch, 1)
        self.merge = nn.Conv2d(ch * (len(ASPP_RATES) + 2), ch, 1)
        nn.init.zeros_(self.merge.weight)
        nn.init.zeros_(self.merge.bias)

    def forward(self, x):
        outs = [F.relu(b(x)) for b in self.branches]
        g = F.relu(self.pooled(x.mean(dim=(2, 3), keepdim=True)))
        outs.append(g.expand(-1, -1, x.shape[2], x.shape[3]))
        return x + self.merge(torch.cat(outs, 1))


class MaskBranch(nn.Module):
    """RoI pooling + mask head (+ the variant's addition) + 28x28 predictor, with its own loss."""

    def __init__(self, variant, head=None, num_classes=2):
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant = variant
        self.pool = MultiScaleRoIAlign(["0", "1", "2", "3"], 14, 2)
        self.head = head if head is not None else MaskRCNNHeads(256, [256, 256, 256, 256], 1)
        self.predictor = MaskRCNNPredictor(256, 256, num_classes)
        self.aspp = ASPP() if variant == "aspp" else None
        if variant == "edge_loss":
            self.edge_out = nn.Conv2d(256, 1, 1)
        if variant == "gated_edge":
            self.edge_branch = nn.Sequential(nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
                                             nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
            self.edge_up = nn.Sequential(nn.ConvTranspose2d(64, 64, 2, 2), nn.ReLU(), nn.Conv2d(64, 1, 1))
            self.gate = nn.Conv2d(64, 256, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def logits(self, features, boxes, image_sizes):
        roi = self.pool(features, boxes, image_sizes)
        x = self.aspp(roi) if self.aspp is not None else roi
        x = self.head(x)
        edge = None
        if self.variant == "gated_edge":
            e = self.edge_branch(roi)
            edge = self.edge_up(e)[:, 0]
            x = x * (1 + torch.tanh(self.gate(e)))
        up = F.relu(self.predictor.conv5_mask(x))
        mask = self.predictor.mask_fcn_logits(up)
        if self.variant == "edge_loss":
            edge = self.edge_out(up)[:, 0]
        return mask, edge

    def loss(self, features, boxes, image_sizes, gt_masks, matched_idxs):
        """boxes: positive proposals per image; matched_idxs: which target object each one is."""
        mask, edge = self.logits(features, boxes, image_sizes)
        size = mask.shape[-1]
        targets = [project_masks_on_boxes(m, b, i, size) for m, b, i in zip(gt_masks, boxes, matched_idxs)]
        targets = torch.cat(targets) if targets else mask.new_zeros((0, size, size))
        if targets.numel() == 0:
            return {"loss_mask": mask.sum() * 0}
        out = {"loss_mask": F.binary_cross_entropy_with_logits(mask[:, 1], targets)}
        if edge is not None:
            out["loss_edge"] = EDGE_WEIGHT * edge_loss(edge, edge_targets(targets))
        return out

    def infer(self, features, boxes, image_sizes):
        """28x28 hypothalamus probabilities per detected box, as (N, 1, 28, 28) per image."""
        counts = [len(b) for b in boxes]
        if sum(counts) == 0:
            return [features["0"].new_zeros((0, 1, 28, 28)) for _ in boxes]
        mask, _ = self.logits(features, boxes, image_sizes)
        return list(torch.sigmoid(mask[:, 1:2]).split(counts))


class EdgeRoIHeads(RoIHeads):
    """torchvision's RoIHeads (box part unchanged) with the mask part handed to a MaskBranch."""

    def forward(self, features, proposals, image_shapes, targets=None):
        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        box_features = self.box_head(self.box_roi_pool(features, proposals, image_shapes))
        class_logits, box_regression = self.box_predictor(box_features)
        result, losses = [], {}
        if self.training:
            loss_cls, loss_box = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
            losses = {"loss_classifier": loss_cls, "loss_box_reg": loss_box}
            pos = [torch.where(lab > 0)[0] for lab in labels]
            losses.update(self.mask_branch.loss(features, [p[k] for p, k in zip(proposals, pos)], image_shapes,
                                                [t["masks"] for t in targets],
                                                [m[k] for m, k in zip(matched_idxs, pos)]))
        else:
            boxes, scores, labs = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            probs = self.mask_branch.infer(features, boxes, image_shapes)
            result = [{"boxes": b, "labels": lab, "scores": s, "masks": p}
                      for b, s, lab, p in zip(boxes, scores, labs, probs)]
        return result, losses


def use_edge_heads(mask_rcnn, variant):
    """Swaps a torchvision Mask R-CNN's mask part for a MaskBranch, keeping its pretrained head."""
    rh = mask_rcnn.roi_heads
    branch = MaskBranch(variant, head=rh.mask_head)
    rh.__class__ = EdgeRoIHeads
    rh.mask_branch = branch
    rh.mask_roi_pool = rh.mask_head = rh.mask_predictor = None
    return mask_rcnn
