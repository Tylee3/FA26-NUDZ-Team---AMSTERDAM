"""Reproduces Charley's 3D Slicer atlas-registration workflow (waxholm.py) in scriptable
SimpleITK, so it can loop over subjects instead of being driven by hand in the Slicer GUI.

Design history: an earlier version of this script used AFFINE registration and found it
unreliable (median IoU ~0.10 across the 10 pilot subjects, some near zero) -- the Mattes MI
metric's global optimum often had near-zero real overlap with the hypothalamus, because
affine's extra degrees of freedom (scale, shear) gave the optimizer too much room to
wander on this small, thick-sliced (12-slice), limited-FOV data.

That version was reimplementing the wrong thing. Reading Charley's actual waxholm.py
(github.com/yokunerukosbelt/autoseg) shows the real Slicer module uses RIGID registration
(Euler3D: rotation + translation only, no scale/shear) by default, GEOMETRY-based centered
init, single-shot (no multi-start, no multi-resolution pyramid). Matching that recipe
exactly here gets median IoU ~0.58 across the 10 pilot subjects (range 0.31-0.67) in one
deterministic run -- no ensembling needed. Rigid's smaller search space is the fix, not
cleverer optimization.

Side note for whoever maintains the Slicer module: waxholm.py's "Use Affine Registration"
checkbox computes an affine refinement (do_affine branch) but then resamples the output
mask using `final_rigid`, not the refined `final_tx` -- the checkbox has no effect on the
actual output. Worth a PR if anyone wants affine to actually do something there.
"""

import argparse
import csv
from pathlib import Path

import SimpleITK as sitk

ATLAS_DIR = Path(__file__).parent
ATLAS_MRI = ATLAS_DIR / "WAXHOLM_SPACE_OF_THE_SPRAGUE_DAWLEY_V1_01.nii"
ATLAS_LABELS = ATLAS_DIR / "WAXHOLM_SPACE_ATLAS_OF_THE_SPRAGUE_DAWLEY_RAT_BRAIN_V4.label.nii"
HYPOTHALAMUS_LABEL = 48


def register_rigid(fixed_f: sitk.Image, moving_f: sitk.Image):
    initial_tx = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )
    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(50)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.2, seed=1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0, minStep=1e-4, numberOfIterations=200, relaxationFactor=0.5,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetInitialTransform(initial_tx, inPlace=False)
    final_transform = reg.Execute(fixed_f, moving_f)
    return final_transform, reg.GetMetricValue()


def hypothalamus_mask_for_subject(subject_t2_path: Path):
    fixed = sitk.ReadImage(str(subject_t2_path))
    moving = sitk.ReadImage(str(ATLAS_MRI))
    labels = sitk.ReadImage(str(ATLAS_LABELS))
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    transform, metric = register_rigid(fixed_f, moving_f)

    resampled_labels = sitk.Resample(labels, fixed, transform, sitk.sitkNearestNeighbor, 0, labels.GetPixelID())
    hypo_mask = sitk.BinaryThreshold(resampled_labels, HYPOTHALAMUS_LABEL, HYPOTHALAMUS_LABEL, 1, 0)
    return sitk.Cast(hypo_mask, sitk.sitkUInt8), metric


def score_against(mask: sitk.Image, manual_path: Path):
    manual = sitk.ReadImage(str(manual_path))
    manual_on_grid = sitk.Resample(
        manual, mask, sitk.Transform(), sitk.sitkNearestNeighbor, 0, manual.GetPixelID()
    )
    a = sitk.GetArrayFromImage(mask) > 0
    b = sitk.GetArrayFromImage(manual_on_grid) > 0
    inter = int((a & b).sum())
    union = int((a | b).sum())
    total = int(a.sum() + b.sum())
    iou = inter / union if union else 1.0
    dice = 2 * inter / total if total else 1.0
    return iou, dice


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subjects", nargs="+", help="subject numbers, e.g. 1 3 5")
    parser.add_argument("--out-dir", default="auto_seg_out")
    args = parser.parse_args()

    out_dir = ATLAS_DIR / args.out_dir
    out_dir.mkdir(exist_ok=True)

    rows = []
    for n in args.subjects:
        n = int(n)
        subject_id = f"{n:03d}"
        t2_path = ATLAS_DIR / "T2 for segmentation" / f"sub-{subject_id}_ses-1_acq-RARE_T2w.nii"
        if not t2_path.exists():
            t2_path = ATLAS_DIR / f"sub-{subject_id}_ses-1_acq-RARE_T2w.nii"
        if not t2_path.exists():
            print(f"[rat {n}] no T2 file found, skipping")
            continue

        print(f"[rat {n}] registering atlas to {t2_path.name} (rigid, single-shot)...")
        mask, metric = hypothalamus_mask_for_subject(t2_path)
        out_path = out_dir / f"auto_seg_{subject_id}.nrrd"
        sitk.WriteImage(mask, str(out_path))
        print(f"[rat {n}] done, metric={metric:.4f}, wrote {out_path.name}")

        row = {"rat": n, "mi_metric": metric}

        me_glob = list((ATLAS_DIR / "Manual Segments" / "Segmented nrrd" / "Me").glob(f"{n}_Segmentation*.nrrd"))
        other_glob = list((ATLAS_DIR / "Manual Segments" / "Segmented nrrd" / "Other").glob(f"{n}_Segmentation*.nrrd"))

        if me_glob:
            iou, dice = score_against(mask, me_glob[0])
            row["iou_vs_me"], row["dice_vs_me"] = iou, dice
            print(f"[rat {n}]   vs Me:    IoU={iou:.4f}  Dice={dice:.4f}")
        if other_glob:
            iou, dice = score_against(mask, other_glob[0])
            row["iou_vs_other"], row["dice_vs_other"] = iou, dice
            print(f"[rat {n}]   vs Other: IoU={iou:.4f}  Dice={dice:.4f}")

        rows.append(row)

    if rows:
        csv_path = out_dir / "results.csv"
        fieldnames = ["rat", "mi_metric", "iou_vs_me", "dice_vs_me", "iou_vs_other", "dice_vs_other"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {csv_path}")


if __name__ == "__main__":
    main()
