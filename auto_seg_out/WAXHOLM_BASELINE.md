# Waxholm atlas baseline

Registering the Waxholm atlas alone gives a median **Dice 0.75 / IoU 0.60** against our hand traces. Agreement is weakest at the front end of the structure: slice 3 scores a median Dice of 0.32.

**How it was measured:**
- **Atlas and label:** Waxholm Space rat atlas v4, label 48 ("Hypothalamic region, unspecified").
- **Registration:** rigid (rotation + translation only), the same recipe as Charley Batte's 3D Slicer module ([`atlas_registration_pipeline.py`](../atlas_registration_pipeline.py)).
- **Rats:** the 9 hand-traced rats (1, 3–10). Rat 2 is excluded because its "manual" file is itself atlas-derived.
- **Scoring:** 3D overlap with each labeler's traces, over the whole 12-slice volume.

## Per rat

| Rat | Dice (labeler 1) | IoU (labeler 1) | Dice (labeler 2) | IoU (labeler 2) |
|---|---|---|---|---|
| 1 | 0.76 | 0.62 | 0.81 | 0.68 |
| 3 | 0.70 | 0.54 | 0.70 | 0.54 |
| 4 | 0.71 | 0.55 | 0.68 | 0.51 |
| 5 | 0.79 | 0.65 | 0.79 | 0.65 |
| 6 | 0.81 | 0.67 | 0.82 | 0.69 |
| 7 | 0.79 | 0.65 | 0.79 | 0.66 |
| 8 | 0.75 | 0.60 | 0.71 | 0.55 |
| 9 | 0.63 | 0.46 | 0.60 | 0.43 |
| 10 | 0.69 | 0.53 | 0.79 | 0.65 |
| **Median** | **0.75** | **0.60** | **0.79** | **0.65** |

For reference, the two human labelers agree with each other at a median Dice of 0.83 / IoU 0.70 on the same 9 rats.

## Per slice (labeler 1)

| Slice | Rats with the structure | Median 2D Dice | Median pixels, manual | Median pixels, atlas |
|---|---|---|---|---|
| 3 | 8 | **0.32** | 73 | 58 |
| 4 | 9 | 0.75 | 499 | 528 |
| 5 | 9 | 0.74 | 570 | 807 |
| 6 | 9 | 0.82 | 601 | 700 |
| 7 | 9 | 0.75 | 523 | 530 |

Slices are numbered from 1 (the first of the 12). Two places the atlas fits poorly:
- **Slice 3**, the thin front end of the structure, where the atlas is placed badly.
- **Slice 5**, where the atlas outline is about 40% larger than the manual trace.

On slice 8, one rat has a manual trace there and the atlas has none.

## Next

Register the SIGMA rat atlas, whose reference brain is an in vivo T2 scan like these, with the same method, and compare it to this table.
