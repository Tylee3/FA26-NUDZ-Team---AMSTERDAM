"""Builds the high-resolution test set: NUDZ's 5 hand-traced isotropic-protocol rats
(50, 60, 75, 90, 110; 144x144x64 at 0.2 mm), traced by expert 1 and, for 50 and 60, expert 2.

Every model so far learned on the thick protocol: coronal slices, 0.1 mm in-plane, 1 mm
apart. The isotropic scans are stored as horizontal (axial) slices, so each scan and its
traces are resampled onto a coronal grid in the thick protocol's orientation, 256x256 at
0.1 mm in-plane, centred on the scan, over the central 12 mm front to back (the same span
as a thick scan; chosen without looking at the traces). Two versions:

  1mm   each slice is the average of 5 consecutive 0.2 mm slices, like a thick slice;
        the truth on a slice is the voxels traced on at least 3 of the 5
  02mm  every 0.2 mm slice, truth as traced

Images are scaled to 0-255 per scan with the same 1st-99th percentile window as
extract_slices_for_roboflow.py. None of these rats is in any training set (training only
uses thick-protocol rats).

Writes rcnn_data_hr/test_1mm.npz and test_02mm.npz in the layout of rcnn_data/test.npz, plus
has_other (per slice: does this rat have an expert 2 trace). Rats without one carry a copy
of expert 1's trace in masks_other so the arrays line up; scoring skips them for expert 2.

Usage (from the repo root):  python3 -m rcnn.build_highres
"""

import numpy as np
import SimpleITK as sitk

from .common import MANUAL_DIR, ROOT, t2_path

HR_RATS = [50, 60, 75, 90, 110]
HR_DIR = ROOT / "rcnn_data_hr"
SIZE, PIXEL, SPAN_MM, THIN = 256, 0.1, 12.0, 0.2
VARIANTS = {"1mm": 5, "02mm": 1}
THICK_DIRECTION = sitk.ReadImage(str(t2_path(1))).GetDirection()


def coronal_grid(scan):
    """A 256 x 256 x 60 grid (0.1 x 0.1 x 0.2 mm) in the thick protocol's orientation,
    centred on the scan."""
    n = int(round(SPAN_MM / THIN))
    centre = np.array(scan.TransformContinuousIndexToPhysicalPoint([(s - 1) / 2 for s in scan.GetSize()]))
    spacing = np.array([PIXEL, PIXEL, THIN])
    half = (np.array([SIZE, SIZE, n]) - 1) / 2 * spacing
    origin = centre - np.array(THICK_DIRECTION).reshape(3, 3) @ half
    grid = sitk.Image(SIZE, SIZE, n, sitk.sitkFloat32)
    grid.SetSpacing(spacing.tolist())
    grid.SetDirection(THICK_DIRECTION)
    grid.SetOrigin(origin.tolist())
    return grid


def resample(img, grid, mask):
    interp = sitk.sitkNearestNeighbor if mask else sitk.sitkLinear
    out = sitk.GetArrayFromImage(sitk.Resample(img, grid, sitk.Transform(), interp, 0, sitk.sitkFloat32))
    return out > 0 if mask else out


def to_uint8(vol):
    nonzero = vol[vol > 0]
    lo, hi = np.percentile(nonzero, [1, 99])
    return ((np.clip(vol, lo, max(hi, lo + 1)) - lo) / (max(hi, lo + 1) - lo) * 255).astype(np.uint8)


def trace(who, n):
    return next((MANUAL_DIR / who).glob(f"{n}_Segmentation*.nrrd"), None)


def main():
    HR_DIR.mkdir(exist_ok=True)
    arrays = {v: {k: [] for k in ("images", "masks", "masks_other", "has_other", "subjects", "zs", "files")}
              for v in VARIANTS}
    for n in HR_RATS:
        scan = sitk.ReadImage(str(t2_path(n)), sitk.sitkFloat32)
        grid = coronal_grid(scan)
        img = resample(scan, grid, False)
        me = resample(sitk.ReadImage(str(trace("Me", n))), grid, True)
        other_path = trace("Other", n)
        other = resample(sitk.ReadImage(str(other_path)), grid, True) if other_path else me
        for v, k in VARIANTS.items():
            m = len(img) // k
            vol = to_uint8(img[:m * k].reshape(m, k, SIZE, SIZE).mean(1))
            a = me[:m * k].reshape(m, k, SIZE, SIZE).mean(1) >= 0.5
            b = other[:m * k].reshape(m, k, SIZE, SIZE).mean(1) >= 0.5
            out = arrays[v]
            for z in range(m):
                out["images"].append(vol[z])
                out["masks"].append(a[z])
                out["masks_other"].append(b[z])
                out["has_other"].append(other_path is not None)
                out["subjects"].append(n)
                out["zs"].append(z)
                out["files"].append(f"hr{n:03d}_{v}_z{z:03d}")
            found = [z for z in range(m) if a[z].any()]
            print(f"rat {n} {v}: {m} slices, expert 1 on {found[0]}-{found[-1]} ({len(found)} slices)"
                  + (", expert 2 too" if other_path else "")
                  + (" - TRACE TOUCHES THE EDGE OF THE SPAN" if found[0] == 0 or found[-1] == m - 1 else ""))
    for v, out in arrays.items():
        np.savez_compressed(HR_DIR / f"test_{v}.npz", **{k: np.stack(x) if k in ("images", "masks", "masks_other")
                                                         else np.array(x) for k, x in out.items()})
        print(f"wrote {HR_DIR / f'test_{v}.npz'} ({len(out['files'])} slices)")


if __name__ == "__main__":
    main()
