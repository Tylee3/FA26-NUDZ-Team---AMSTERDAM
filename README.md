# FA26 NUDZ Team — Hypothalamus MRI Segmentation

IFSA Prague Challenge Project, Spring 2026. Partner org: NUDZ (Czech National Institute
of Mental Health, nano-neuromedicine lab). Goal: automatic hypothalamus segmentation in
rat brain MRI, benchmarked against Waxholm atlas registration, extended to human MRI,
assessed for mobile deployment.

## What's in this repo

- `atlas_registration_pipeline.py` — scriptable SimpleITK reimplementation of Charley
  Batte's `autoseg` 3D Slicer module (registers the Waxholm atlas to a subject T2w MRI,
  extracts the hypothalamus label). See the docstring for why plain global registration
  doesn't work here and what the ROI-restricted, ensembled version does instead.
- `IoU_charley.py`, `nrrd_to_minc_andI0U.py`, `visualise_nrrd.py` — Charley's original
  IoU/Dice + visualization scripts.
- `auto_seg_out/results.csv`, `batch_bars.png`, `batch_overlay.png` — results from running
  the pipeline across all 10 pilot subjects: median IoU ~0.10 vs. manual traces, well
  below the human-supervised Slicer baseline (rat 3, IoU 0.49). Automated atlas
  registration isn't reliable without a human confirming it — worth factoring into how
  much weight the team puts on the registration approach vs. a learned model.
- `WAXHOLM_SPACE_*.nii`, `WHS_SD_rat_atlas_v4.label` — the public Waxholm rat brain atlas
  (label 48 = hypothalamus), needed to run the pipeline script.
- `instructions.docx` — Charley's step-by-step for the 3D Slicer module.
- `PROJECT_SUMMARY.md` — working notes on data completeness and where the team stands.
- `deadline_calendar.html` / `Hypothalamus_Segmentation_Deadline_Calendar.docx` — team
  deadline calendar and proposal task breakdown (rubric due Sun Oct 11, 23:59).
- `extract_slices_for_roboflow.py` — converts the T2w `.nii` volumes into 2D PNG slices
  for Roboflow upload (Roboflow doesn't take NIfTI directly). Writes `roboflow_slices/`
  (gitignored — regenerate locally, don't commit it) plus a `manifest.csv` mapping each
  PNG back to its source subject/slice.

### Heads up: the 132-subject set is two different scan protocols

63 subjects (all of the pilot 10 among them) are thick-slice: 256x256x12, 1mm z-spacing.
The other 69 are a finer isotropic protocol: 144x144x64, 0.2mm spacing. They're
interleaved by subject number, not a clean split (e.g. 1-43 and 97-132 are thick-slice;
44-96 and 105-120 are isotropic) — check `roboflow_slices/manifest.csv`'s `protocol`
column for any given subject rather than assuming from the number.

Both cover almost the same physical field of view (~26x26x12mm), so `extract_slices_for_roboflow.py`
subsamples the isotropic subjects by default (every 5th slice) to match the thick-slice
protocol's ~1mm spacing — otherwise the isotropic subjects alone would contribute ~4,400
near-duplicate images. The isotropic protocol's slices also render in a different in-plane
orientation than the thick-slice ones (rotated, not mirrored — verified as a genuine
anterior-to-posterior coronal-style sequence in both, just stored with different axis
conventions). Worth a heads-up to whoever's annotating so it doesn't read as a data error.

## What's NOT in this repo

The raw T2 scans (`T2 for segmentation/`, ~330MB, 132 subjects) and the manual/traced
segmentations (`Manual Segments/`, two independent labelers across the 10 pilot rats)
are NUDZ's research data — too large for git and not ours to redistribute casually.
They live on the shared Google Drive. Pull them from there if you need to run the
pipeline locally; `.gitignore` keeps them out of commits.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install SimpleITK pynrrd numpy
python3 atlas_registration_pipeline.py <rat-number> [<rat-number> ...]
```
