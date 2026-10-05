"""Selective Search region proposals, the external proposal step that R-CNN and Fast R-CNN
depend on (Faster R-CNN and later replaced it with a learned Region Proposal Network).

Proposals are cached per slice in rcnn_data/ss_cache/, so they're computed once.
"""

from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np

from .common import DATA_DIR, load_split

CACHE_DIR = DATA_DIR / "ss_cache"
MAX_PROPOSALS = 1000
MIN_SIDE = 4


def proposals_for(file_name: str, gray: np.ndarray) -> np.ndarray:
    """[N, 4] float32 boxes (x0, y0, x1, y1) in the slice's own pixel coordinates."""
    path = CACHE_DIR / (Path(file_name).stem + ".npy")
    if path.exists():
        return np.load(path)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
    ss.setBaseImage(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))
    ss.switchToSelectiveSearchFast()
    boxes = [(x, y, x + w, y + h) for x, y, w, h in ss.process() if w >= MIN_SIDE and h >= MIN_SIDE]
    arr = np.array(boxes[:MAX_PROPOSALS], dtype=np.float32).reshape(-1, 4)
    np.save(path, arr)
    return arr


def _work(args):
    proposals_for(*args)


def main():
    jobs = []
    for split in ("train", "test"):
        data = load_split(split)
        jobs += [(str(f), img) for f, img in zip(data["files"], data["images"])]
    with Pool() as pool:
        pool.map(_work, jobs, chunksize=8)
    counts = [len(np.load(p)) for p in CACHE_DIR.glob("*.npy")]
    print(f"{len(counts)} slices, proposals per slice: median {int(np.median(counts))}, min {min(counts)}")


if __name__ == "__main__":
    main()
