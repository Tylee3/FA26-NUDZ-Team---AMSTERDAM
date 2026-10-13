"""Shared paths, dataset loading and box helpers for the R-CNN baseline."""

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "rcnn_data"
SLICES_DIR = ROOT / "roboflow_slices"
T2_DIR = ROOT / "T2 for segmentation"
MANUAL_DIR = ROOT / "Manual Segments" / "Segmented nrrd"
ATLAS_TRAIN_DIR = ROOT / "atlas_labels"
ATLAS_TEST_DIR = ROOT / "auto_seg_out"

# The 9 hand-traced rats are the test set. Rat 2's "manual" file is itself atlas-derived.
TEST_RATS = [1, 3, 4, 5, 6, 7, 8, 9, 10]
PILOT_RATS = set(range(1, 11))

MODELS = ["rcnn", "fast_rcnn", "faster_rcnn", "rfcn", "cascade_rcnn", "mask_rcnn"]
MASK_MODELS = {"mask_rcnn"}

# One detection per slice counts as "found it" above this score. R-CNN's SVM outputs a
# margin rather than a probability, so its natural cut-off is 0.
SCORE_THRESHOLDS = {"rcnn": 0.0}
DEFAULT_SCORE_THRESHOLD = 0.5


def t2_path(n: int) -> Path:
    return T2_DIR / f"sub-{n:03d}_ses-1_acq-RARE_T2w.nii"


def pick_device(name=None):
    if name:
        return torch.device(name)
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def load_split(split: str) -> dict:
    with np.load(DATA_DIR / f"{split}.npz", allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def mask_to_box(mask: np.ndarray):
    """Tight box around a binary mask as [x0, y0, x1, y1] (x1/y1 exclusive), or None."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], dtype=np.float32)


def box_iou(a, b) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def image_tensor(gray_uint8: np.ndarray) -> torch.Tensor:
    """Grayscale slice -> 3-channel float tensor, since the pretrained networks expect RGB."""
    t = torch.from_numpy(gray_uint8).float().div(255.0)
    return t.unsqueeze(0).repeat(3, 1, 1)


def neighbor_tensor(images, index, i) -> torch.Tensor:
    """The slice below, the slice itself and the slice above as the 3 channels, so the
    network sees where the structure is tapering off. At the ends of a stack, the slice
    itself fills in for the missing neighbor."""
    n, z = index["subject"][i], index["z"][i]
    below = index["lookup"].get((n, z - 1), i)
    above = index["lookup"].get((n, z + 1), i)
    stack = np.stack([images[below], images[i], images[above]])
    return torch.from_numpy(stack).float().div(255.0)


def augment(img: torch.Tensor, mask: np.ndarray):
    """Training-time "alterations": left-right flip (the brain is left-right symmetric),
    a small rotation, shift and zoom, and a brightness/contrast change. The mask moves with
    the image, and the box is recomputed from the moved mask afterwards."""
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as TF

    m = torch.from_numpy(mask[None].astype(np.uint8))
    if torch.rand(()) < 0.5:
        img, m = TF.horizontal_flip(img), TF.horizontal_flip(m)
    angle = float(torch.empty(()).uniform_(-10, 10))
    shift = [int(torch.randint(-10, 11, ())), int(torch.randint(-10, 11, ()))]
    scale = float(torch.empty(()).uniform_(0.9, 1.1))
    img = TF.affine(img, angle=angle, translate=shift, scale=scale, shear=[0.0],
                    interpolation=InterpolationMode.BILINEAR)
    m = TF.affine(m, angle=angle, translate=shift, scale=scale, shear=[0.0],
                  interpolation=InterpolationMode.NEAREST)
    contrast = float(torch.empty(()).uniform_(0.85, 1.15))
    brightness = float(torch.empty(()).uniform_(-0.08, 0.08))
    img = (img * contrast + brightness).clamp(0, 1)
    return img, m[0].numpy().astype(bool)


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, with_masks: bool = False, with_proposals: bool = False,
                 limit=None, neighbors: bool = False, augment: bool = False):
        self.data = load_split(split)
        self.with_masks = with_masks
        self.with_proposals = with_proposals
        self.neighbors = neighbors
        self.augment = augment
        self.n = len(self.data["files"]) if limit is None else min(limit, len(self.data["files"]))
        self.index = {
            "subject": self.data["subjects"], "z": self.data["zs"],
            "lookup": {(int(s), int(z)): k for k, (s, z) in enumerate(zip(self.data["subjects"], self.data["zs"]))},
        }

    def __len__(self):
        return self.n

    def image(self, i) -> torch.Tensor:
        if self.neighbors:
            return neighbor_tensor(self.data["images"], self.index, i)
        return image_tensor(self.data["images"][i])

    def __getitem__(self, i):
        gray = self.data["images"][i]
        mask = self.data["masks"][i]
        img = self.image(i)
        if self.augment:
            img, mask = augment(img, mask)
        box = mask_to_box(mask)
        target = {"image_id": torch.tensor(i)}
        if box is None:
            target["boxes"] = torch.zeros((0, 4), dtype=torch.float32)
            target["labels"] = torch.zeros((0,), dtype=torch.int64)
            if self.with_masks:
                target["masks"] = torch.zeros((0, *mask.shape), dtype=torch.uint8)
        else:
            target["boxes"] = torch.from_numpy(box[None])
            target["labels"] = torch.ones((1,), dtype=torch.int64)
            if self.with_masks:
                target["masks"] = torch.from_numpy(mask[None].astype(np.uint8))
        if self.with_proposals:
            from .selective_search import proposals_for
            props = torch.from_numpy(proposals_for(str(self.data["files"][i]), gray))
            return img, target, props
        return img, target


def collate(batch):
    return tuple(zip(*batch))
