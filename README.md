# FA26 NUDZ Team — Hypothalamus MRI Segmentation

IFSA Prague Challenge Project, Spring 2026. Partner org: NUDZ (Czech National Institute
of Mental Health, nano-neuromedicine lab). Goal: automatic hypothalamus segmentation in
rat brain MRI, benchmarked against Waxholm atlas registration, extended to human MRI,
assessed for mobile deployment.

## What's in this repo

- `atlas_registration_pipeline.py` — scriptable SimpleITK reimplementation of Charley
  Batte's `autoseg` 3D Slicer module (registers the Waxholm atlas to a subject T2w MRI,
  extracts the hypothalamus label). Matches the real module's recipe: rigid registration
  (Euler3D, GEOMETRY-based init, Mattes MI), single deterministic run, no GUI needed.
- `IoU_charley.py`, `nrrd_to_minc_andI0U.py`, `visualise_nrrd.py` — Charley's original
  IoU/Dice + visualization scripts.
- `auto_seg_out/results.csv`, `batch_bars.png`, `batch_overlay.png` — results from running
  the pipeline across all 10 pilot subjects: median IoU 0.60 vs. manual traces (range
  0.46-0.67, excluding rat 2 whose "manual" mask is itself atlas-derived), 8 of 9 at or
  above Charley's own reported Slicer result for rat 3 (IoU 0.49). Atlas registration is
  a solid baseline here — as long as it's run as RIGID registration, not affine. (An
  earlier version of this script used affine and found it unreliable, median IoU ~0.10 —
  affine's extra degrees of freedom let the optimizer wander on this small, thick-sliced
  data. See the script's docstring for the full story, including a real bug found in
  waxholm.py: its "Use Affine Registration" checkbox computes an affine refinement but
  never actually uses it when producing the output mask.)
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
- `generate_coco_annotations.py` — per the 2026-10-01 company meeting ("don't need to
  manually annotate, just automate the process"), this traces the existing masks (manual
  traces + the atlas pipeline's output) into COCO polygon annotations instead of
  hand-drawing in Roboflow. Traces every contour pixel exactly (no simplification), so
  the full boundary precision of the source mask survives the format conversion. Writes
  `coco_annotations/coco_manual_me.json`, `coco_manual_other.json`, `coco_atlas_auto.json`
  (gitignored — see below for why — regenerate locally once you have the Drive data).
  Covers only the 10 pilot subjects, since that's all we have real masks for.

## R-CNN baseline (`rcnn/`)

The first two-week task: run NUDZ's six R-CNN-family models on the data as it is (no
augmentation yet), score them the same way as the atlas baseline, and look at the results.

**Where the labels come from.** `WHS_SD_rat_atlas_v4.label` is the atlas's lookup table
(48 = "Hypothalamic region, unspecified"); the outline itself is in the atlas label
volume. Neither is drawn on our rats, so the training labels are the atlas outline carried
onto each rat by `atlas_registration_pipeline.py`. That's 53 thick-slice rats outside the
pilot 10, written to `atlas_labels/` (spot-checked visually: all in the right place, a
few sitting slightly high). The models never see the 9 hand-traced rats (1, 3-10) during
training; those are the test set, scored against the manual traces.

**The six models** (torchvision, all from the same COCO-pretrained ResNet-50, same
training settings — see `rcnn/train.py`):

| Model | What it changed | Output | Implementation |
|---|---|---|---|
| R-CNN (2014) | CNN run separately on each Selective Search region, then an SVM | box | ours (`rcnn_classic.py`) |
| Fast R-CNN (2015) | CNN once per image, regions cropped from its features | box | ours |
| Faster R-CNN (2015) | learned Region Proposal Network replaces Selective Search | box | torchvision |
| R-FCN (2016) | position-sensitive score maps make the per-region step nearly free | box | ours |
| Cascade R-CNN (2018) | three box heads, each stricter (IoU 0.5/0.6/0.7), refining the last | box | ours |
| Mask R-CNN (2017) | adds a pixel-mask branch | **mask** | torchvision |

Only Mask R-CNN outlines the structure; the other five draw a box. Every model gets a box
score; 3D Dice/IoU treats a box as a filled rectangle, so box models should be compared
with the "Atlas, as a box" row, not the atlas mask row.

**Known limit for R-CNN and Fast R-CNN:** Selective Search almost never proposes a box
around the hypothalamus here — only 16% of test slices have any candidate overlapping
the true box by >= 0.5 IoU, none by >= 0.7 (quality mode: 18%). It groups regions by
color/texture, and the hypothalamus has no strong boundary on grayscale MRI. These two
models can only choose among those candidates, so they start capped. That's the problem
Faster R-CNN's learned proposals were invented to solve.

**Run it** (on whichever machine; ~4-6 hours on an M1, mostly unattended):

```
cd ~/Downloads/hypo_segments
python3 -m rcnn.build_dataset
python3 -m rcnn.selective_search
caffeinate -i ./rcnn/run_all.sh
```

**Look at it:** `rcnn_out/comparison.md` (the table), `rcnn_out/comparison_grid.png`
(every model on every test rat, yellow = agreement, red = model only, green = missed),
`rcnn_out/<model>/overlay.png` (best/median/worst rat for one model), and in 3D Slicer:
View > Python Console, then `exec(open("<path>/rcnn/slicer_compare.py").read())` — edit
`RAT` and `SHOW` at the top of that file for a different rat or just your two models.

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

**Also gitignored on purpose:** `coco_manual_me.json` / `coco_manual_other.json` are a
lossless re-encoding of the proprietary manual traces (same mask shapes, different file
format) — per the 2026-10-01 meeting notes ("manual segment folder is proprietary"),
converting the format doesn't change that, so they stay local-only like the NRRDs they're
derived from. `coco_atlas_auto.json` is derived from our own registration output against
the public Waxholm atlas, not from proprietary data, but it's gitignored too for now,
consistent with "git file kept secret" — **this repo should be set to Private on GitHub**
if it isn't already.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install SimpleITK pynrrd numpy pillow matplotlib torch torchvision opencv-contrib-python-headless scikit-learn
python3 atlas_registration_pipeline.py <rat-number> [<rat-number> ...]
```
