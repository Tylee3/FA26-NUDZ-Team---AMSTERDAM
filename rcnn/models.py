"""The R-CNN-family models, built on torchvision.

Every model starts from the same COCO-pretrained ResNet-50 (from torchvision's Faster
R-CNN weights) and gets a fresh 2-class output layer (background, hypothalamus).

- faster_rcnn, mask_rcnn: torchvision's own implementations.
- fast_rcnn:    Faster R-CNN with its Region Proposal Network removed; boxes come from
                Selective Search instead, as in the 2015 paper.
- cascade_rcnn: three box heads in sequence, each trained at a stricter IoU threshold
                (0.5, 0.6, 0.7) and refining the previous head's boxes (Cai & Vasconcelos 2018).
- rfcn:         position-sensitive score maps + position-sensitive RoI pooling, so the
                per-region step is just an average over a 7x7 grid (Dai et al. 2016).
                Single-scale, stride-16 features from ResNet layer3; its RPN head is new
                (the COCO one was trained on different features). No OHEM.
- rcnn (2014) is in rcnn_classic.py: it isn't an end-to-end network.

fast_rcnn, cascade_rcnn and rfcn are our own implementations from torchvision parts,
not reference code.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (FasterRCNN_ResNet50_FPN_Weights,
                                          MaskRCNN_ResNet50_FPN_Weights,
                                          fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn)
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import AnchorGenerator, RegionProposalNetwork, RPNHead
from torchvision.ops import MultiScaleRoIAlign, ps_roi_align
from torchvision.ops import boxes as box_ops

NUM_CLASSES = 2
IMG_SIZE = 512  # slices are 256px; every model upsamples 2x, the same for all


def faster_base():
    return fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
                                   min_size=IMG_SIZE, max_size=IMG_SIZE)


def build_faster_rcnn():
    m = faster_base()
    m.roi_heads.box_predictor = FastRCNNPredictor(1024, NUM_CLASSES)
    return m


def build_mask_rcnn():
    m = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1,
                              min_size=IMG_SIZE, max_size=IMG_SIZE)
    m.roi_heads.box_predictor = FastRCNNPredictor(1024, NUM_CLASSES)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, NUM_CLASSES)
    return m


def sample_rois(proposals, targets, matcher, sampler, coder):
    """Adds ground-truth boxes to the proposals, labels each proposal by IoU, samples a
    fixed-size foreground/background mix, and computes box-regression targets."""
    out_props, out_labels, out_regs = [], [], []
    for props, t in zip(proposals, targets):
        gt = t["boxes"]
        props = torch.cat([props, gt])
        if gt.numel() == 0:
            labels = torch.zeros(len(props), dtype=torch.int64, device=props.device)
            matched_gt = props.new_tensor([[0.0, 0.0, 1.0, 1.0]]).expand(len(props), 4)
        else:
            matched_idx = matcher(box_ops.box_iou(gt, props))
            clamped = matched_idx.clamp(min=0)
            labels = t["labels"][clamped].to(torch.int64)
            labels[matched_idx == det_utils.Matcher.BELOW_LOW_THRESHOLD] = 0
            labels[matched_idx == det_utils.Matcher.BETWEEN_THRESHOLDS] = -1
            matched_gt = gt[clamped]
        pos, neg = sampler([labels])
        keep = torch.where(pos[0] | neg[0])[0]
        props, labels, matched_gt = props[keep], labels[keep], matched_gt[keep]
        out_props.append(props)
        out_labels.append(labels)
        out_regs.append(coder.encode_single(matched_gt, props))
    return out_props, out_labels, out_regs


def postprocess_boxes(boxes_list, scores_list, sizes, score_thresh, nms_thresh, top_k):
    out = []
    for boxes, scores, size in zip(boxes_list, scores_list, sizes):
        boxes = box_ops.clip_boxes_to_image(boxes, size)
        keep = torch.where(scores > score_thresh)[0]
        boxes, scores = boxes[keep], scores[keep]
        keep = box_ops.remove_small_boxes(boxes, 1e-2)
        boxes, scores = boxes[keep], scores[keep]
        keep = box_ops.nms(boxes, scores, nms_thresh)[:top_k]
        out.append({"boxes": boxes[keep], "scores": scores[keep],
                    "labels": torch.ones(len(keep), dtype=torch.int64, device=boxes.device)})
    return out


class FastRCNN(nn.Module):
    """Faster R-CNN's backbone and RoI head, fed Selective Search proposals instead of an RPN."""

    def __init__(self):
        super().__init__()
        base = build_faster_rcnn()
        self.transform, self.backbone, self.roi_heads = base.transform, base.backbone, base.roi_heads

    def forward(self, images, targets=None, proposals=None):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        image_list, targets = self.transform(images, targets)
        scaled = []
        for p, (oh, ow), (nh, nw) in zip(proposals, original_sizes, image_list.image_sizes):
            scaled.append(p * p.new_tensor([nw / ow, nh / oh, nw / ow, nh / oh]))
        features = self.backbone(image_list.tensors)
        detections, losses = self.roi_heads(features, scaled, image_list.image_sizes, targets)
        if self.training:
            return losses
        return self.transform.postprocess(detections, image_list.image_sizes, original_sizes)


class CascadeRCNN(nn.Module):
    STAGE_IOUS = (0.5, 0.6, 0.7)
    STAGE_LOSS_WEIGHTS = (1.0, 0.5, 0.25)
    STAGE_CODER_WEIGHTS = ((10.0, 10.0, 5.0, 5.0), (20.0, 20.0, 10.0, 10.0), (30.0, 30.0, 15.0, 15.0))

    def __init__(self):
        super().__init__()
        base = faster_base()
        self.transform, self.backbone, self.rpn = base.transform, base.backbone, base.rpn
        self.box_roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], 7, 2)
        pretrained_head = base.roi_heads.box_head.state_dict()
        self.heads = nn.ModuleList()
        self.predictors = nn.ModuleList()
        for _ in self.STAGE_IOUS:
            head = TwoMLPHead(256 * 7 * 7, 1024)
            head.load_state_dict(pretrained_head)
            self.heads.append(head)
            self.predictors.append(FastRCNNPredictor(1024, NUM_CLASSES))
        self.coders = [det_utils.BoxCoder(w) for w in self.STAGE_CODER_WEIGHTS]
        self.matchers = [det_utils.Matcher(t, t, allow_low_quality_matches=False) for t in self.STAGE_IOUS]
        self.sampler = det_utils.BalancedPositiveNegativeSampler(512, 0.25)

    def _stage(self, features, proposals, sizes, s):
        x = self.heads[s](self.box_roi_pool(features, proposals, sizes))
        return self.predictors[s](x)

    def _decode(self, deltas, proposals, sizes, s):
        fg = deltas.view(len(deltas), NUM_CLASSES, 4)[:, 1]
        boxes = self.coders[s].decode_single(fg, torch.cat(proposals))
        return [box_ops.clip_boxes_to_image(b, size)
                for b, size in zip(boxes.split([len(p) for p in proposals]), sizes)]

    def forward(self, images, targets=None):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        image_list, targets = self.transform(images, targets)
        sizes = image_list.image_sizes
        features = self.backbone(image_list.tensors)
        proposals, rpn_losses = self.rpn(image_list, features, targets)

        if self.training:
            losses = dict(rpn_losses)
            for s in range(len(self.STAGE_IOUS)):
                proposals, labels, regs = sample_rois([p.detach() for p in proposals], targets,
                                                      self.matchers[s], self.sampler, self.coders[s])
                logits, deltas = self._stage(features, proposals, sizes, s)
                cls_loss, box_loss = fastrcnn_loss(logits, deltas, labels, regs)
                losses[f"stage{s + 1}_cls"] = cls_loss * self.STAGE_LOSS_WEIGHTS[s]
                losses[f"stage{s + 1}_box"] = box_loss * self.STAGE_LOSS_WEIGHTS[s]
                proposals = self._decode(deltas.detach(), proposals, sizes, s)
            return losses

        stage_scores = []
        for s in range(len(self.STAGE_IOUS)):
            logits, deltas = self._stage(features, proposals, sizes, s)
            stage_scores.append(F.softmax(logits, -1)[:, 1])
            refined = self._decode(deltas, proposals, sizes, s)
            if s < len(self.STAGE_IOUS) - 1:
                proposals = refined
        scores = torch.stack(stage_scores).mean(0).split([len(p) for p in proposals])
        detections = postprocess_boxes(refined, scores, sizes, 0.05, 0.5, 100)
        return self.transform.postprocess(detections, sizes, original_sizes)


class RFCN(nn.Module):
    K = 7

    def __init__(self):
        super().__init__()
        base = faster_base()
        body = base.backbone.body
        self.transform = base.transform
        self.stem = nn.Sequential(body.conv1, body.bn1, body.relu, body.maxpool,
                                  body.layer1, body.layer2, body.layer3)  # stride 16, 1024 ch
        anchors = AnchorGenerator(sizes=((32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
        self.rpn = RegionProposalNetwork(
            anchors, RPNHead(1024, anchors.num_anchors_per_location()[0]),
            fg_iou_thresh=0.7, bg_iou_thresh=0.3, batch_size_per_image=256, positive_fraction=0.5,
            pre_nms_top_n=dict(training=6000, testing=6000),
            post_nms_top_n=dict(training=300, testing=300), nms_thresh=0.7)
        self.reduce = nn.Sequential(nn.Conv2d(1024, 512, 1), nn.ReLU(inplace=True))
        self.cls_map = nn.Conv2d(512, self.K * self.K * NUM_CLASSES, 1)
        self.box_map = nn.Conv2d(512, self.K * self.K * 4, 1)
        self.matcher = det_utils.Matcher(0.5, 0.5, allow_low_quality_matches=False)
        self.sampler = det_utils.BalancedPositiveNegativeSampler(128, 0.25)
        self.coder = det_utils.BoxCoder((10.0, 10.0, 5.0, 5.0))

    def forward(self, images, targets=None):
        original_sizes = [tuple(img.shape[-2:]) for img in images]
        image_list, targets = self.transform(images, targets)
        feat = self.stem(image_list.tensors)
        proposals, rpn_losses = self.rpn(image_list, {"0": feat}, targets)
        x = self.reduce(feat)
        cls_map, box_map = self.cls_map(x), self.box_map(x)

        if self.training:
            proposals, labels, regs = sample_rois(proposals, targets, self.matcher, self.sampler, self.coder)
        logits = ps_roi_align(cls_map, proposals, self.K, spatial_scale=1 / 16, sampling_ratio=2).mean(dim=(2, 3))
        deltas = ps_roi_align(box_map, proposals, self.K, spatial_scale=1 / 16, sampling_ratio=2).mean(dim=(2, 3))

        if self.training:
            labels, regs = torch.cat(labels), torch.cat(regs)
            pos = torch.where(labels > 0)[0]
            box_loss = F.smooth_l1_loss(deltas[pos], regs[pos], beta=1 / 9, reduction="sum") / labels.numel()
            return {**rpn_losses, "rfcn_cls": F.cross_entropy(logits, labels), "rfcn_box": box_loss}

        counts = [len(p) for p in proposals]
        boxes = self.coder.decode_single(deltas, torch.cat(proposals)).split(counts)
        scores = F.softmax(logits, -1)[:, 1].split(counts)
        detections = postprocess_boxes(boxes, scores, image_list.image_sizes, 0.05, 0.3, 100)
        return self.transform.postprocess(detections, image_list.image_sizes, original_sizes)


BUILDERS = {
    "faster_rcnn": build_faster_rcnn,
    "mask_rcnn": build_mask_rcnn,
    "fast_rcnn": FastRCNN,
    "cascade_rcnn": CascadeRCNN,
    "rfcn": RFCN,
}


def build(name):
    return BUILDERS[name]()
