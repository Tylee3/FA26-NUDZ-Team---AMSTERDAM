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
