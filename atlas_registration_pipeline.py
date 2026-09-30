"""Reproduces Charley's 3D Slicer atlas-registration workflow (waxholm.py) in scriptable
SimpleITK, so it can loop over subjects instead of being driven by hand in the Slicer GUI.

Design note (from debugging on rat 3): a plain global affine registration with Mattes
Mutual Information is NOT reliable on this data. The MI metric is dominated by skull/tissue
matching elsewhere in the thick-sliced, limited-FOV subject volume, so the metric's global
optimum can land at zero overlap with the real hypothalamus while a "worse" local optimum
sits right on top of it (verified empirically: best-by-metric had IoU 0.0, best-by-IoU had
IoU 0.21, across 12 random restarts of a naive global registration).

Fix: use the untouched centered (MOMENTS) initial alignment to seed a dilated region of
interest around where the atlas hypothalamus roughly lands, then restrict the registration
metric to that ROI and multi-start a *local* affine refinement inside it. With the ROI
restriction, best-by-metric tracks best-by-real-overlap much more closely (IoU 0.31 vs 0.21
for the naive global version, on rat 3), and is far more consistent across seeds.

Even within the ROI, though, best-by-metric is noisy trial-to-trial (verified across 25
restarts: the single best-metric trial scored IoU 0.16 while a middling-metric trial scored
IoU 0.40). Picking one "winning" transform isn't reliable. What is reliable: taking the
top-5 trials by metric and voxel-wise majority-voting their masks together (a standard
multi-atlas-style consensus trick) — this consistently lands around IoU 0.33 / Dice 0.50 on
rat 3, close to the oracle best-of-25 and far more stable than trusting any single trial.

This is still meaningfully worse than Charley's Slicer output for rat 3 (IoU 0.49, Dice
0.66) — that result likely benefits from a human visually confirming/nudging the
registration before accepting it. Treat this script's numbers as "how good can pure
automation get without a human in the loop," not a replacement for the manual workflow.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ATLAS_DIR = Path(__file__).parent
ATLAS_MRI = ATLAS_DIR / "WAXHOLM_SPACE_OF_THE_SPRAGUE_DAWLEY_V1_01.nii"
ATLAS_LABELS = ATLAS_DIR / "WAXHOLM_SPACE_ATLAS_OF_THE_SPRAGUE_DAWLEY_RAT_BRAIN_V4.label.nii"
HYPOTHALAMUS_LABEL = 48
NUM_TRIALS = 25
TOP_K_FOR_VOTE = 5
ROI_DILATION_XY_VOXELS = 15
ROI_DILATION_Z_VOXELS = 4


def keep_largest_component(mask: sitk.Image) -> sitk.Image:
    labeled = sitk.ConnectedComponent(mask)
    if sitk.GetArrayFromImage(labeled).max() == 0:
        return mask
    relabeled = sitk.RelabelComponent(labeled, sortByObjectSize=True)
    return sitk.Cast(sitk.Equal(relabeled, 1), sitk.sitkUInt8)


def register_local(fixed_f, moving_f, roi_mask, seed, jitter_scale):
    base_init = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    t = sitk.AffineTransform(base_init)
    if seed > 0:
        rng = np.random.default_rng(1000 + seed)
        params = np.array(t.GetParameters())
        params[9:12] += rng.uniform(-2, 2, size=3)
        params[0:9] = np.eye(3).flatten() + rng.uniform(-jitter_scale, jitter_scale, size=9)
        t.SetParameters(params.tolist())

    reg = sitk.ImageRegistrationMethod()
    reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    reg.SetMetricFixedMask(roi_mask)
    reg.SetMetricSamplingStrategy(reg.RANDOM)
    reg.SetMetricSamplingPercentage(0.9, seed=seed + 1)
    reg.SetInterpolator(sitk.sitkLinear)
    reg.SetOptimizerAsGradientDescent(
        learningRate=0.5, numberOfIterations=300,
        convergenceMinimumValue=1e-7, convergenceWindowSize=20,
    )
    reg.SetOptimizerScalesFromPhysicalShift()
    reg.SetShrinkFactorsPerLevel([2, 1])
    reg.SetSmoothingSigmasPerLevel([1, 0])
    reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    reg.SetInitialTransform(t, inPlace=False)
    final = reg.Execute(fixed_f, moving_f)
    return final, reg.GetMetricValue()


def hypothalamus_mask_for_subject(subject_t2_path: Path, num_trials=NUM_TRIALS):
    fixed = sitk.ReadImage(str(subject_t2_path))
    moving = sitk.ReadImage(str(ATLAS_MRI))
    labels = sitk.ReadImage(str(ATLAS_LABELS))
    fixed_f = sitk.Cast(fixed, sitk.sitkFloat32)
    moving_f = sitk.Cast(moving, sitk.sitkFloat32)

    seed_init = sitk.CenteredTransformInitializer(
        fixed_f, moving_f, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.MOMENTS,
    )
    seed_resampled = sitk.Resample(labels, fixed, seed_init, sitk.sitkNearestNeighbor, 0, labels.GetPixelID())
    seed_hypo = sitk.BinaryThreshold(seed_resampled, HYPOTHALAMUS_LABEL, HYPOTHALAMUS_LABEL, 1, 0)
    roi = sitk.BinaryDilate(seed_hypo, [ROI_DILATION_XY_VOXELS, ROI_DILATION_XY_VOXELS, ROI_DILATION_Z_VOXELS])

    trials = []
    for trial in range(num_trials):
        if trial == 0:
            jitter_scale = 0.0
        else:
            jitter_scale = np.random.default_rng(1000 + trial).uniform(0.03, 0.12)
        transform, metric = register_local(fixed_f, moving_f, roi, trial, jitter_scale)
        resampled_labels = sitk.Resample(labels, fixed, transform, sitk.sitkNearestNeighbor, 0, labels.GetPixelID())
        hypo = sitk.BinaryThreshold(resampled_labels, HYPOTHALAMUS_LABEL, HYPOTHALAMUS_LABEL, 1, 0)
        trials.append((metric, sitk.GetArrayFromImage(hypo) > 0))

    trials.sort(key=lambda r: r[0])
    top_k = trials[:min(TOP_K_FOR_VOTE, len(trials))]
    stack = np.stack([arr for _, arr in top_k])
    vote_arr = stack.sum(axis=0) >= (len(top_k) // 2 + 1)

    hypo_mask = sitk.GetImageFromArray(vote_arr.astype(np.uint8))
    hypo_mask.CopyInformation(fixed)
    hypo_mask = keep_largest_component(hypo_mask)
    best_metric = top_k[0][0]
    return hypo_mask, best_metric


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
    parser.add_argument("--trials", type=int, default=NUM_TRIALS)
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

        print(f"[rat {n}] registering atlas to {t2_path.name} ({args.trials} local-refinement restarts)...")
        mask, best_metric = hypothalamus_mask_for_subject(t2_path, num_trials=args.trials)
        out_path = out_dir / f"auto_seg_{subject_id}.nrrd"
        sitk.WriteImage(mask, str(out_path))
        print(f"[rat {n}] done, best metric={best_metric:.4f}, wrote {out_path.name}")

        row = {"rat": n, "mi_metric": best_metric}

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
