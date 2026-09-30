# Project: Hypothalamus MRI Segmentation — NUDZ Challenge Project

**Company**: National Institute of Mental Health (NUDZ), Czech Republic — Nano-neuromedicine lab
**Program**: IFSA Prague Challenge Project, Spring 2026
**Goal**: Build and evaluate an automatic method for localizing/segmenting the hypothalamus in rat brain MRI, benchmark against atlas-based registration, extend to human MRI, and assess mobile deployment feasibility.

## Deliverables (from company brief)
1. Evaluate atlas-based registration using Dice coefficient and IoU
2. Develop/test segmentation pipelines (YOLO, U-Net, and/or R-CNN)
3. Compare results, improve segmentation performance
4. Translate best approach to human brain MRI, assess limitations
5. Evaluate mobile-application compatibility

## Local data — folder: `hypo_segments` (in Downloads)

This folder contains the files worked through so far. Contents:

### Raw scans
10 T2-weighted rat brain MRIs, `sub-001_ses-1_acq-RARE_T2w.nii` through `sub-010_ses-1_acq-RARE_T2w.nii` — a pilot subset of a larger 132-image dataset the team has separately (not all locally verified yet).

### Manual ground-truth segmentations (NRRD, 256×256×12, binary masks)
Confirmed for rats 1, 3, 5, 6, 7, 8, 10. Two gaps:
- Rat 2's file is explicitly labeled "from atlas" — it's an automated output, not a manual trace. Needs clarifying with NUDZ.
- Rats 4 and 9 are missing entirely — likely exist in the full folder but weren't part of what came through.

### Atlas assets
Waxholm Space rat brain atlas (`WAXHOLM_SPACE_ATLAS...V4_label.nii`, `..._S621.nii`, `...V1_01.nii`) and the label definitions (`WHS_SD_rat_atlas_v4.label`) — used to auto-generate atlas-based segmentations.

### Prior intern's tooling (Charley Batte, GitHub: `yokunerukosbelt/autoseg`)
- A 3D Slicer Python module that registers the Waxholm atlas to a subject MRI and auto-generates a hypothalamus segmentation (region label 48)
- `instructions.docx` — step-by-step for loading/using the module in 3D Slicer
- `IoU_charley.py`, `nrrd_to_minc_andI0U.py` — compute IoU/Dice between two NRRD masks
- `visualise_nrrd.py` — overlays manual vs. automated masks on a slice, reports 3D IoU/Dice, color-coded agreement map

### Reference material
A Scientific Data paper (Rodrigues et al., "High-resolution segmentations of the hypothalamus...") on a related human-hypothalamus synthetic-data segmentation approach — useful for pipeline/methodology ideas, not directly reusable since it's human ex vivo data.

## Where the team stands
- Team of 3 (roles drafted: CV/ML Engineer, Medical Imaging & Registration Specialist, Project Integration & Application Lead — individual assignments not finalized)
- Plan: use Roboflow for annotation/augmentation on the full 132-image set; train models locally
- Kickoff meeting with NUDZ has happened; IFSA WhatsApp communication policy also received and needs setup

## Next steps
1. **Run the atlas-registration baseline** on the 7 confirmed subjects (1, 3, 5, 6, 7, 8, 10) using the `autoseg` 3D Slicer module, then score against the manual masks with the IoU/Dice scripts — this gives a first real baseline number.
2. **Resolve the data gaps**: ask NUDZ whether rat 2 has a real manual trace, and get the missing rat 4 / rat 9 manual masks.
3. **Batch the pipeline**: currently the 3D Slicer workflow is manual/GUI-driven per subject — worth scripting it to loop over all subjects automatically rather than repeating it by hand.
4. **Scale to the full 132-image set**: set up Roboflow for annotation of a training/validation split, decide segmentation architecture (YOLO-seg vs. U-Net vs. R-CNN) to benchmark against the atlas baseline.
5. **Proposal writing**: Week 5 Challenge Project Proposal is due Sunday, Oct 11 — role justifications, tech stack, timeline, and risks sections can largely be drafted now; logistics/communication plan needs to be finalized with NUDZ.
