"""Extracts 2D PNG slices from the T2w .nii volumes for Roboflow upload.

Roboflow doesn't take NIfTI directly, so each subject's 3D volume needs to become a
stack of 2D images first.

The 132-subject set is NOT one consistent acquisition: 63 subjects (including all of
the pilot 10) are thick-slice (256x256x12, 1.0mm z-spacing); 69 subjects are a finer
isotropic protocol (144x144x64, 0.2mm z-spacing). Both cover almost the same physical
field of view (~26x26x12mm) -- the isotropic protocol just slices it 5x finer. Extracting
every one of its 64 slices would produce ~5,172 images total, most of them near-duplicates
of their z-neighbors. By default this script subsamples the isotropic subjects so their
output slice spacing matches the thick-slice protocol's ~1mm (use --no-subsample to get
every native slice instead).

Each slice is normalized independently per-VOLUME (not per-slice) using a 1st-99th
percentile window over that volume's nonzero voxels, so brightness is stable across a
subject's own slice stack instead of flickering slice-to-slice.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image

TARGET_Z_SPACING_MM = 1.0


def normalize_volume_to_uint8(arr):
    nonzero = arr[arr > 0]
    if nonzero.size == 0:
        lo, hi = 0, 1
    else:
        lo, hi = np.percentile(nonzero, [1, 99])
        if hi <= lo:
            hi = lo + 1
    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255).astype(np.uint8)


def process_subject(nii_path: Path, out_dir: Path, subsample: bool):
    img = sitk.ReadImage(str(nii_path))
    size = img.GetSize()
    spacing = img.GetSpacing()
    z_spacing = spacing[2]

    arr = sitk.GetArrayFromImage(img)  # (z, y, x)
    norm = normalize_volume_to_uint8(arr)

    stride = 1
    if subsample and z_spacing < TARGET_Z_SPACING_MM * 0.6:
        stride = max(1, round(TARGET_Z_SPACING_MM / z_spacing))

    subject_id = nii_path.stem.split("_")[0]  # e.g. "sub-003"
    protocol = "thick" if size == (256, 256, 12) else "iso" if stride == 1 else f"iso-stride{stride}"

    rows = []
    for z in range(0, arr.shape[0], stride):
        out_name = f"{subject_id}_z{z:03d}.png"
        Image.fromarray(norm[z]).save(out_dir / out_name)
        rows.append({
            "subject": subject_id,
            "source_file": nii_path.name,
            "size": "x".join(map(str, size)),
            "spacing_mm": ",".join(f"{s:.4f}" for s in spacing),
            "protocol": protocol,
            "slice_z": z,
            "output_file": out_name,
        })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="T2 for segmentation")
    parser.add_argument("--output-dir", default="roboflow_slices")
    parser.add_argument("--subjects", nargs="*", default=None, help="e.g. 1 3 50 (default: all)")
    parser.add_argument("--no-subsample", action="store_true", help="extract every native slice, no z-subsampling")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)

    files = sorted(input_dir.glob("*.nii"))
    if args.subjects:
        wanted = {f"sub-{int(n):03d}" for n in args.subjects}
        files = [f for f in files if f.stem.split("_")[0] in wanted]

    all_rows = []
    skipped = []
    for f in files:
        try:
            rows = process_subject(f, out_dir, subsample=not args.no_subsample)
            all_rows.extend(rows)
            print(f"{f.name}: wrote {len(rows)} slices ({rows[0]['protocol']})")
        except Exception as e:
            skipped.append((f.name, str(e)))
            print(f"{f.name}: SKIPPED ({e})")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["subject", "source_file", "size", "spacing_mm", "protocol", "slice_z", "output_file"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n{len(all_rows)} images written to {out_dir}/")
    print(f"Manifest: {manifest_path}")
    if skipped:
        print(f"\n{len(skipped)} subject(s) skipped:")
        for name, err in skipped:
            print(f"  {name}: {err}")


if __name__ == "__main__":
    main()
