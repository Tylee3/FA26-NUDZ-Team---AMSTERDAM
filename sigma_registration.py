"""Registers the SIGMA rat atlas (Barriere et al. 2019, CC-BY 4.0) to each rat with the same
rigid recipe as the Waxholm baseline, and scores it the same way, so the two atlases can
be compared side by side.

SIGMA's in vivo template is a T2-weighted scan of living rats, like NUDZ's; the Waxholm
template is an ex vivo scan. The question is whether that closer match aligns better.

Usage:  python3 sigma_registration.py --variant mask            (the 9 hand-traced rats)
        python3 sigma_registration.py 11 12 13 --variant mask   (any rats, e.g. training rats)
Variants are listed above VARIANTS below; waxholm_init* need sigma_to_waxholm.py run first.

Writes sigma_out/<variant>/auto_seg_XXX.nrrd, results.csv and prints the per-slice table.

Setup: download sigma_wistar_rat_brain_templatesandatlases_version_2.0.zip from
https://zenodo.org/records/10635831 (CC-BY 4.0, cite Barriere et al., Nat Commun 2019) and
unzip SIGMA_InVivo_Anatomical_Brain_{template,mask}.nii.gz and
SIGMA_InVivo_Anatomical_Brain_Atlas.{nii.gz,txt} into sigma/.

Result (2026-10-13, 9 test rats, median vs labeler 1; Waxholm rigid = Dice 0.749 / IoU 0.599):
  plain               0.637 / 0.467  (better than Waxholm on 1/9 rats)
  mask                0.625 / 0.455  (1/9; one rat fails completely)
  waxholm_init        0.703 / 0.542  (3/9; best SIGMA variant)
  waxholm_init_refine 0.681 / 0.516  (2/9; one rat fails completely)
No variant beats Waxholm. With Waxholm's alignment (waxholm_init) the remaining gap is the
label itself: SIGMA's region 48 is ~30% smaller (18.8 vs 26.8 mm^3) and never reaches the
front slice (slice 3) that the hand traces include. Aligning to SIGMA's in vivo image did
not align better than Waxholm, even when started from Waxholm's alignment.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from atlas_registration_pipeline import register_rigid, score_against

ROOT = Path(__file__).parent
SIGMA_DIR = ROOT / "sigma"
SIGMA_TEMPLATE = SIGMA_DIR / "SIGMA_InVivo_Anatomical_Brain_template.nii.gz"
SIGMA_ATLAS = SIGMA_DIR / "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz"
SIGMA_LABELS = SIGMA_DIR / "SIGMA_InVivo_Anatomical_Brain_Atlas.txt"
OUT_DIR = ROOT / "sigma_out"
TEST_RATS = [1, 3, 4, 5, 6, 7, 8, 9, 10]

# SIGMA label ids that make up the hypothalamus target; set after reading the label list.
HYPOTHALAMUS_IDS = [48]  # "Hypothalamic region, unspecified", same id and name as Waxholm


def t2_path(n):
    return ROOT / "T2 for segmentation" / f"sub-{n:03d}_ses-1_acq-RARE_T2w.nii"


def manual_path(n, who):
    return next((ROOT / "Manual Segments" / "Segmented nrrd" / who).glob(f"{n}_Segmentation*.nrrd"), None)


# Fixed in advance (not tuned on the test rats):
#   plain               same rigid recipe as Waxholm, against SIGMA's in vivo template
#   mask                plain, but the match is measured only inside SIGMA's brain mask
#                       (SIGMA's template has no skull; NUDZ's scans do)
#   waxholm_init        SIGMA's outline carried through the rat's Waxholm alignment plus the
#                       one-off SIGMA-to-Waxholm atlas alignment (sigma_to_waxholm.py)
#   waxholm_init_refine waxholm_init, then a rigid touch-up against SIGMA's in vivo template
#                       (brain-only match)
VARIANTS = ["plain", "mask", "waxholm_init", "waxholm_init_refine"]
SIGMA_MASK = SIGMA_DIR / "SIGMA_InVivo_Anatomical_Brain_mask.nii.gz"


def rigid(fixed_f, moving_f, moving_mask=None, moving_initial=None):
    """Charley's rigid recipe (as register_rigid), optionally brain-only and/or on top of a
    fixed starting transform. With a starting transform, the total mapping is
    moving_initial(rigid(x))."""
    if moving_initial is None:
        initial = sitk.CenteredTransformInitializer(fixed_f, moving_f, sitk.Euler3DTransform(),
                                                    sitk.CenteredTransformInitializerFilter.GEOMETRY)
    else:
        initial = sitk.Euler3DTransform()
        initial.SetCenter(fixed_f.TransformContinuousIndexToPhysicalPoint([(n - 1) / 2 for n in fixed_f.GetSize()]))
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(learningRate=2.0, minStep=1e-4,
                                                 numberOfIterations=200, relaxationFactor=0.5)
    reg.SetOptimizerScalesFromPhysicalShift()
    if moving_mask is not None:
        reg.SetMetricMovingMask(moving_mask)
    if moving_initial is not None:
        reg.SetMovingInitialTransform(moving_initial)
    reg.SetInitialTransform(initial, inPlace=True)
    reg.Execute(fixed_f, moving_f)
    return initial


def transform_for(variant, fixed, sigma_f, sigma_mask, wax_f, wax_to_sigma):
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    if variant == "plain":
        return register_rigid(fixed_f, sigma_f)[0]
    if variant == "mask":
        return rigid(fixed_f, sigma_f, moving_mask=sigma_mask)
    wax_rigid = register_rigid(fixed_f, wax_f)[0]
    start = sitk.CompositeTransform([wax_to_sigma, wax_rigid])  # applies wax_rigid first
    if variant == "waxholm_init":
        return start
    touch_up = rigid(fixed_f, sigma_f, moving_mask=sigma_mask, moving_initial=start)
    return sitk.CompositeTransform([wax_to_sigma, wax_rigid, touch_up])  # touch_up first


def hypothalamus_mask(fixed, transform, atlas):
    warped = sitk.Resample(atlas, fixed, transform, sitk.sitkNearestNeighbor, 0, atlas.GetPixelID())
    arr = np.isin(sitk.GetArrayFromImage(warped), HYPOTHALAMUS_IDS).astype(np.uint8)
    mask = sitk.GetImageFromArray(arr)
    mask.CopyInformation(fixed)
    return mask


def per_slice(mask, manual):
    a = sitk.GetArrayFromImage(mask) > 0
    m = sitk.GetArrayFromImage(sitk.Resample(manual, mask, sitk.Transform(), sitk.sitkNearestNeighbor,
                                             0, manual.GetPixelID())) > 0
    out = {}
    for z in range(a.shape[0]):
        if a[z].any() or m[z].any():
            out[z] = (2 * (a[z] & m[z]).sum() / (a[z].sum() + m[z].sum()), int(m[z].sum()), int(a[z].sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("subjects", nargs="*", type=int, default=TEST_RATS)
    ap.add_argument("--variant", choices=VARIANTS, default="plain")
    args = ap.parse_args()
    out_dir = OUT_DIR / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    sigma_f = sitk.Cast(sitk.ReadImage(str(SIGMA_TEMPLATE)), sitk.sitkFloat32)
    sigma_mask = sitk.Cast(sitk.ReadImage(str(SIGMA_MASK)) > 0, sitk.sitkUInt8)
    atlas = sitk.ReadImage(str(SIGMA_ATLAS))
    wax_f = wax_to_sigma = None
    if args.variant.startswith("waxholm_init"):
        from atlas_registration_pipeline import ATLAS_MRI
        wax_f = sitk.Cast(sitk.ReadImage(str(ATLAS_MRI)), sitk.sitkFloat32)
        wax_to_sigma = sitk.ReadTransform(str(SIGMA_DIR / "waxholm_to_sigma.tfm"))

    rows, slices = [], {}
    for n in args.subjects:
        fixed = sitk.ReadImage(str(t2_path(n)))
        transform = transform_for(args.variant, fixed, sigma_f, sigma_mask, wax_f, wax_to_sigma)
        mask = hypothalamus_mask(fixed, transform, atlas)
        sitk.WriteImage(mask, str(out_dir / f"auto_seg_{n:03d}.nrrd"))
        row = {"rat": n}
        for who in ("Me", "Other"):
            p = manual_path(n, who)
            if p is not None:
                row[f"iou_vs_{who.lower()}"], row[f"dice_vs_{who.lower()}"] = score_against(mask, p)
                if who == "Me":
                    for z, v in per_slice(mask, sitk.ReadImage(str(p))).items():
                        slices.setdefault(z, []).append(v)
        rows.append(row)
        print(f"rat {n}: " + "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k != "rat"), flush=True)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scored = [r for r in rows if "iou_vs_me" in r]
    if scored:
        print(f"\n[{args.variant}] median vs labeler 1: Dice {np.median([r['dice_vs_me'] for r in scored]):.3f}, "
              f"IoU {np.median([r['iou_vs_me'] for r in scored]):.3f}")
        print("slice (1-based) | rats | median 2D Dice | manual px | atlas px")
        for z in sorted(slices):
            v = slices[z]
            print(f"{z + 1} | {len(v)} | {np.median([x[0] for x in v]):.2f} | "
                  f"{int(np.median([x[1] for x in v]))} | {int(np.median([x[2] for x in v]))}")


if __name__ == "__main__":
    main()
