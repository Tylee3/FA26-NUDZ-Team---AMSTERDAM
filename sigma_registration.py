"""Registers the SIGMA rat atlas (Barriere et al. 2019, CC-BY 4.0) to each rat with the same
rigid recipe as the Waxholm baseline, and scores it the same way, so the two atlases can
be compared side by side.

SIGMA's in vivo template is a T2-weighted scan of living rats, like NUDZ's; the Waxholm
template is an ex vivo scan. The question is whether that closer match aligns better.

Usage:  python3 sigma_registration.py            (the 9 hand-traced rats)
        python3 sigma_registration.py 11 12 13   (any rats, e.g. to relabel training rats)

Writes sigma_out/auto_seg_XXX.nrrd, sigma_out/results.csv and prints the per-slice table.

Setup: download sigma_wistar_rat_brain_templatesandatlases_version_2.0.zip from
https://zenodo.org/records/10635831 (CC-BY 4.0, cite Barriere et al., Nat Commun 2019) and
unzip SIGMA_InVivo_Anatomical_Brain_{template,mask}.nii.gz and
SIGMA_InVivo_Anatomical_Brain_Atlas.{nii.gz,txt} into sigma/.

Result (2026-10-13, 9 test rats, same rigid recipe as Waxholm): median Dice 0.637 / IoU
0.467 vs Waxholm's 0.749 / 0.599. SIGMA's outline is about the right size on the middle
slices but placed toward the back (missing on slice 3, spilling onto slice 8); a whole-slice
shift does not fix it, so the offset is partial and the cause is in the alignment.
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


def hypothalamus_mask(fixed, moving, atlas):
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)
    rigid, _ = register_rigid(fixed_f, moving_f)
    warped = sitk.Resample(atlas, fixed, rigid, sitk.sitkNearestNeighbor, 0, atlas.GetPixelID())
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
    args = ap.parse_args()
    if not HYPOTHALAMUS_IDS:
        raise SystemExit("Set HYPOTHALAMUS_IDS from SIGMA_InVivo_Anatomical_Brain_Atlas.txt first.")
    OUT_DIR.mkdir(exist_ok=True)
    moving = sitk.ReadImage(str(SIGMA_TEMPLATE))
    atlas = sitk.ReadImage(str(SIGMA_ATLAS))

    rows, slices = [], {}
    for n in args.subjects:
        fixed = sitk.ReadImage(str(t2_path(n)))
        mask = hypothalamus_mask(fixed, moving, atlas)
        sitk.WriteImage(mask, str(OUT_DIR / f"auto_seg_{n:03d}.nrrd"))
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

    with open(OUT_DIR / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    scored = [r for r in rows if "iou_vs_me" in r]
    if scored:
        print(f"\nmedian vs labeler 1: Dice {np.median([r['dice_vs_me'] for r in scored]):.3f}, "
              f"IoU {np.median([r['iou_vs_me'] for r in scored]):.3f}")
        print("slice (1-based) | rats | median 2D Dice | manual px | atlas px")
        for z in sorted(slices):
            v = slices[z]
            print(f"{z + 1} | {len(v)} | {np.median([x[0] for x in v]):.2f} | "
                  f"{int(np.median([x[1] for x in v]))} | {int(np.median([x[2] for x in v]))}")


if __name__ == "__main__":
    main()
