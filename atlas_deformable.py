"""Experiment: refine the rigid atlas registration with a deformable (B-spline) step, so the
atlas outline can bend to fit each rat's shape instead of only being rotated and shifted.

Same rigid step as atlas_registration_pipeline.py, then a B-spline transform on a coarse
control-point grid, optionally with the similarity metric restricted to a margin around
the rigid hypothalamus (so tissue far away can't pull the fit off). The affine attempt
taught us that extra freedom can make registration worse, so every setting is scored
against the hand traces next to the rigid result before anything uses it.

Usage:  python3 atlas_deformable.py            (all three preset settings, 9 test rats)
        python3 atlas_deformable.py --config b 11 12 13   (one setting, any rats)

Writes atlas_deformable_out/<config>/auto_seg_XXX.nrrd and results.csv.

Result (2026-10-13, 9 test rats, median 3D IoU vs Tyler's traces): rigid 0.599; a 0.526,
b 0.593, c 0.598. No setting beat rigid. Without a margin the fit is pulled off by distant
tissue (as with affine); with a margin the outline barely moves, likely because the Waxholm
template (ex vivo) and NUDZ's in vivo T2 scans differ too much in contrast for a local fit.
Rigid stays the baseline; an in vivo T2 template (SIGMA) is the next thing to try.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from atlas_registration_pipeline import (ATLAS_DIR, ATLAS_LABELS, ATLAS_MRI, HYPOTHALAMUS_LABEL,
                                         register_rigid, score_against)

TEST_RATS = [1, 3, 4, 5, 6, 7, 8, 9, 10]

# Fixed in advance, not tuned on the test rats: (control-point grid per axis, metric margin in mm or None)
CONFIGS = {
    "a": {"mesh": [3, 3, 1], "margin_mm": None},
    "b": {"mesh": [4, 4, 2], "margin_mm": 2.0},
    "c": {"mesh": [8, 8, 2], "margin_mm": 2.0},
}


def t2_path(n):
    return ATLAS_DIR / "T2 for segmentation" / f"sub-{n:03d}_ses-1_acq-RARE_T2w.nii"


def margin_mask(label_on_fixed, margin_mm):
    spacing = label_on_fixed.GetSpacing()
    radius = [max(1, int(round(margin_mm / s))) for s in spacing]
    return sitk.Cast(sitk.BinaryDilate(label_on_fixed, radius), sitk.sitkUInt8)


def register_deformable(fixed, moving, labels, mesh, margin_mm):
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    rigid, _ = register_rigid(fixed_f, moving_f)
    moving_rigid = sitk.Resample(moving_f, fixed_f, rigid, sitk.sitkLinear, 0.0)

    bspline = sitk.BSplineTransformInitializer(fixed_f, mesh, order=3)
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    if margin_mm:
        rigid_label = sitk.BinaryThreshold(
            sitk.Resample(labels, fixed, rigid, sitk.sitkNearestNeighbor, 0, labels.GetPixelID()),
            HYPOTHALAMUS_LABEL, HYPOTHALAMUS_LABEL, 1, 0)
        reg.SetMetricFixedMask(margin_mask(rigid_label, margin_mm))
        reg.SetMetricSamplingStrategy(reg.NONE)
    else:
        reg.SetMetricSamplingStrategy(reg.RANDOM)
        reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, numberOfIterations=100,
                             maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=500,
                             costFunctionConvergenceFactor=1e7)
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SetInitialTransform(bspline, inPlace=True)
    reg.Execute(fixed_f, moving_rigid)

    # The B-spline maps fixed -> rigidly-moved atlas; the rigid maps that -> atlas.
    # SimpleITK applies the LAST transform in the list FIRST, so this is rigid(bspline(x)).
    composite = sitk.CompositeTransform([rigid, bspline])
    warped = sitk.Resample(labels, fixed, composite, sitk.sitkNearestNeighbor, 0, labels.GetPixelID())
    mask = sitk.BinaryThreshold(warped, HYPOTHALAMUS_LABEL, HYPOTHALAMUS_LABEL, 1, 0)
    return sitk.Cast(mask, sitk.sitkUInt8)


def manual_path(n, who):
    return next((ATLAS_DIR / "Manual Segments" / "Segmented nrrd" / who).glob(f"{n}_Segmentation*.nrrd"), None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", type=int, default=TEST_RATS)
    ap.add_argument("--config", choices=list(CONFIGS), action="append")
    args = ap.parse_args()
    moving = sitk.ReadImage(str(ATLAS_MRI))
    labels = sitk.ReadImage(str(ATLAS_LABELS))

    for name in args.config or list(CONFIGS):
        cfg = CONFIGS[name]
        out_dir = ATLAS_DIR / "atlas_deformable_out" / name
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        for n in args.subjects:
            fixed = sitk.ReadImage(str(t2_path(n)))
            mask = register_deformable(fixed, moving, labels, cfg["mesh"], cfg["margin_mm"])
            sitk.WriteImage(mask, str(out_dir / f"auto_seg_{n:03d}.nrrd"))
            row = {"rat": n}
            for who in ("Me", "Other"):
                p = manual_path(n, who)
                if p is not None:
                    row[f"iou_vs_{who.lower()}"], row[f"dice_vs_{who.lower()}"] = score_against(mask, p)
            rows.append(row)
            print(f"[{name}] rat {n}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "rat"), flush=True)
        with open(out_dir / "results.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        scored = [r for r in rows if "iou_vs_me" in r]
        if scored:
            print(f"[{name}] mesh {cfg['mesh']}, margin {cfg['margin_mm']}: median IoU vs Me "
                  f"{np.median([r['iou_vs_me'] for r in scored]):.3f}, Dice "
                  f"{np.median([r['dice_vs_me'] for r in scored]):.3f}\n", flush=True)


if __name__ == "__main__":
    main()
