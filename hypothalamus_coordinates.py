"""Hypothalamus coordinates for every source, in one shared frame (Waxholm space, mm), so they
can be compared directly. Answers NUDZ's request (2026-10-15) to check whether the
coordinates of the Waxholm atlas, the manual traces and SIGMA match.

WHS_SD_rat_atlas_v4.label itself holds no coordinates: it is the lookup table (id, color,
name). The coordinates are in the atlas label volume. Each rat's scan sits in its own
scanner frame, so its points are first carried into Waxholm space by that rat's rigid
registration (the same one the Waxholm baseline uses). Points are mapped directly, with no
resampling, so the comparison is not limited by any voxel size.

Sources:
  waxholm_atlas       label 48 of the Waxholm atlas (native Waxholm space)
  sigma_atlas         label 48 of the SIGMA in vivo atlas, via the atlas-to-atlas fit (sigma_to_waxholm.py)
  manual_me / other   each rat's hand trace (rat 2's file is atlas-derived, flagged)
  sigma_on_rat        SIGMA registered to the rat on its own (sigma_out/plain)

Writes hypothalamus_coordinates/: waxholm_labels.json (the .label table as JSON),
summary.json, COORDINATES.md, and voxels/<source>.json (every voxel's [x, y, z] in mm).
"""

import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from atlas_registration_pipeline import ATLAS_LABELS, ATLAS_MRI, register_rigid

ROOT = Path(__file__).parent
OUT = ROOT / "hypothalamus_coordinates"
LABEL_FILE = ROOT / "WHS_SD_rat_atlas_v4.label"
SIGMA_ATLAS = ROOT / "sigma" / "SIGMA_InVivo_Anatomical_Brain_Atlas.nii.gz"
WAX_TO_SIGMA = ROOT / "sigma" / "waxholm_to_sigma.tfm"
RATS = range(1, 11)
ATLAS_DERIVED = {2}
HYPOTHALAMUS = 48


def label_table():
    rows = []
    for line in LABEL_FILE.read_text().splitlines():
        m = re.match(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(\d)\s+(\d)\s+"(.*)"', line)
        if m:
            idx, r, g, b, a, vis, msh, name = m.groups()
            rows.append({"id": int(idx), "name": name, "rgb": [int(r), int(g), int(b)],
                         "alpha": float(a), "visible": vis == "1", "mesh_visible": msh == "1"})
    return rows


def mask_points(img, value=None):
    """Physical [x, y, z] (mm) of every voxel in the mask (or equal to value), plus voxel volume."""
    arr = sitk.GetArrayFromImage(img)
    sel = (np.rint(arr) == value) if value is not None else (arr > 0)
    zyx = np.argwhere(sel)
    pts = np.array([img.TransformIndexToPhysicalPoint((int(i), int(j), int(k))) for k, j, i in zyx])
    return pts.reshape(-1, 3), float(np.prod(img.GetSpacing()))


def moved(points, transform):
    return np.array([transform.TransformPoint(tuple(p)) for p in points]).reshape(-1, 3)


def stats(points, voxel_volume):
    lo, hi = points.min(axis=0), points.max(axis=0)
    r = lambda v: [round(float(x), 3) for x in v]
    return {"voxels": int(len(points)), "volume_mm3": round(len(points) * voxel_volume, 2),
            "centroid_mm": r(points.mean(axis=0)), "bbox_min_mm": r(lo), "bbox_max_mm": r(hi),
            "extent_mm": r(hi - lo)}


def manual_file(n, who):
    return next((ROOT / "Manual Segments" / "Segmented nrrd" / who).glob(f"{n}_Segmentation*.nrrd"), None)


def main():
    OUT.mkdir(exist_ok=True)
    (OUT / "voxels").mkdir(exist_ok=True)
    (OUT / "waxholm_labels.json").write_text(json.dumps(label_table(), indent=1))

    sources = {}

    def add(name, points, voxel_volume, meta=None):
        sources[name] = {**(meta or {}), **stats(points, voxel_volume)}
        (OUT / "voxels" / f"{name}.json").write_text(json.dumps([[round(float(c), 3) for c in p] for p in points]))

    wax_pts, wax_vv = mask_points(sitk.ReadImage(str(ATLAS_LABELS)), HYPOTHALAMUS)
    add("waxholm_atlas", wax_pts, wax_vv, {"source": "Waxholm atlas label 48"})

    sigma_pts, sigma_vv = mask_points(sitk.ReadImage(str(SIGMA_ATLAS)), HYPOTHALAMUS)
    wax_to_sigma = sitk.ReadTransform(str(WAX_TO_SIGMA))
    add("sigma_atlas", moved(sigma_pts, wax_to_sigma.GetInverse()), sigma_vv,
        {"source": "SIGMA in vivo atlas label 48, mapped into Waxholm space",
         "note": "volume is SIGMA's own; the atlas-to-atlas fit enlarges it ~10% when mapped, since SIGMA's brain is smaller"})

    wax_template = sitk.Cast(sitk.ReadImage(str(ATLAS_MRI)), sitk.sitkFloat32)
    for n in RATS:
        scan = sitk.ReadImage(str(ROOT / "T2 for segmentation" / f"sub-{n:03d}_ses-1_acq-RARE_T2w.nii"))
        rat_to_wax = register_rigid(sitk.Cast(scan, sitk.sitkFloat32), wax_template)[0]
        meta = {"rat": n, "atlas_derived_file": n in ATLAS_DERIVED}
        for who, key in (("Me", "manual_me"), ("Other", "manual_other")):
            p = manual_file(n, who)
            if p is not None:
                pts, vv = mask_points(sitk.ReadImage(str(p)))
                add(f"rat{n:02d}_{key}", moved(pts, rat_to_wax), vv, {**meta, "source": f"hand trace ({who})"})
        sigma_rat = ROOT / "sigma_out" / "plain" / f"auto_seg_{n:03d}.nrrd"
        if sigma_rat.exists():
            pts, vv = mask_points(sitk.ReadImage(str(sigma_rat)))
            add(f"rat{n:02d}_sigma_on_rat", moved(pts, rat_to_wax), vv,
                {**meta, "source": "SIGMA registered to the rat on its own"})
        print(f"rat {n} done", flush=True)

    ref = np.array(sources["waxholm_atlas"]["centroid_mm"])
    for s in sources.values():
        d = np.array(s["centroid_mm"]) - ref
        s["offset_from_waxholm_atlas_mm"] = [round(float(x), 3) for x in d]
        s["distance_from_waxholm_atlas_mm"] = round(float(np.linalg.norm(d)), 3)
    (OUT / "summary.json").write_text(json.dumps(
        {"frame": "Waxholm space, mm (SimpleITK physical coordinates)", "sources": sources}, indent=1))

    lines = [
        "# Hypothalamus coordinates in Waxholm space",
        "",
        "Every source mapped into the same frame (Waxholm space, mm). Offsets are each centroid minus",
        "the Waxholm atlas centroid. Generated by `hypothalamus_coordinates.py`; full numbers in `summary.json`.",
        "",
        "| Source | Volume (mm³) | Centroid x, y, z (mm) | Size x, y, z (mm) | Offset from Waxholm (mm) | Distance (mm) |",
        "|---|---|---|---|---|---|",
    ]
    for name, s in sources.items():
        flag = " (atlas-derived file)" if s.get("atlas_derived_file") and "manual" in name else ""
        fmt3 = lambda v: ", ".join(f"{x:.2f}" for x in v)
        lines.append(f"| {name}{flag} | {s['volume_mm3']:.1f} | {fmt3(s['centroid_mm'])} | {fmt3(s['extent_mm'])} | "
                     f"{fmt3(s['offset_from_waxholm_atlas_mm'])} | {s['distance_from_waxholm_atlas_mm']:.2f} |")
    traced = [n for n in RATS if n not in ATLAS_DERIVED]
    def med(key):
        v = [sources[f"rat{n:02d}_{key}"]["distance_from_waxholm_atlas_mm"] for n in traced if f"rat{n:02d}_{key}" in sources]
        return float(np.median(v)), len(v)
    def mean_offset(key):
        v = [sources[f"rat{n:02d}_{key}"]["offset_from_waxholm_atlas_mm"] for n in traced if f"rat{n:02d}_{key}" in sources]
        return ", ".join(f"{x:+.2f}" for x in np.mean(v, axis=0))
    me, other, sig = med("manual_me"), med("manual_other"), med("sigma_on_rat")
    summary = [
        "## Do they match?",
        "",
        f"On the {me[1]} hand-traced rats (rat 2's file is atlas-derived and left out):",
        "",
        f"- **Hand traces vs. Waxholm atlas:** centroids a median {me[0]:.2f} mm apart (labeler 1) and {other[0]:.2f} mm "
        "(labeler 2), well under one CAMRI slice (1 mm). They match closely.",
        f"- **SIGMA registered on its own vs. Waxholm atlas:** a median {sig[0]:.2f} mm apart, with a consistent mean "
        f"offset of ({mean_offset('sigma_on_rat')}) mm. The y axis runs front to back, the direction CAMRI's 1 mm slices are "
        "cut, so this is the partial-slice shift seen in the per-slice Dice table.",
        f"- **SIGMA atlas vs. Waxholm atlas** (atlas to atlas): {sources['sigma_atlas']['distance_from_waxholm_atlas_mm']:.2f} mm apart. "
        f"SIGMA's region is smaller ({sources['sigma_atlas']['volume_mm3']:.1f} vs. {sources['waxholm_atlas']['volume_mm3']:.1f} mm³).",
        "",
        "Axes: x is left–right (centroids sit near 0, the midline), y front–back, z top–bottom.",
        "Sizes along y in the rat scans are limited by the 1 mm slice spacing.",
        "",
    ]
    lines[5:5] = summary
    (OUT / "COORDINATES.md").write_text("\n".join(lines) + "\n")
    print("\n".join(summary))


if __name__ == "__main__":
    main()
