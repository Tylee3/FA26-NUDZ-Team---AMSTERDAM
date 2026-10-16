# Results

Median Dice and IoU of every method on the 9 hand-traced test rats (1, 3–10), measured over the
whole 3D volume. No method saw these rats during training. Labeler 1 is Tyler's traces;
labeler 2 is the second, independent set. Regenerate with `python3 make_results_table.py`.

| Method | Output | Dice (lab. 1) | IoU (lab. 1) | Dice (lab. 2) | IoU (lab. 2) | Slices found | False alarms | Notes |
|---|---|---|---|---|---|---|---|---|
| **Human vs. human** | outline | 0.82 | 0.70 | — | — | — | — | the realistic ceiling |
| **Waxholm atlas, rigid** (Charley's method) | outline | 0.75 | 0.60 | 0.79 | 0.65 | — | — | baseline; labels for training |
| Waxholm atlas, deformable | outline | 0.75 | 0.60 | 0.77 | 0.63 | — | — | best of 3 preset settings |
| SIGMA atlas | outline | 0.70 | 0.54 | 0.70 | 0.54 | — | — | best of 4 variants (waxholm_init) |
| R-CNN | box | 0.25 | 0.14 | 0.21 | 0.12 | 11/45 | 8/63 | scored as a filled box |
| Fast R-CNN | box | 0.00 | 0.00 | 0.00 | 0.00 | 5/45 | 0/63 | scored as a filled box |
| Faster R-CNN | box | 0.65 | 0.48 | 0.69 | 0.52 | 39/45 | 0/63 | scored as a filled box |
| R-FCN | box | 0.62 | 0.45 | 0.65 | 0.48 | 37/45 | 1/63 | scored as a filled box |
| Cascade R-CNN | box | 0.65 | 0.48 | 0.72 | 0.56 | 35/45 | 0/63 | scored as a filled box |
| Mask R-CNN | outline | 0.74 | 0.59 | 0.73 | 0.57 | 38/45 | 0/63 |  |
| Mask R-CNN + augmentation | outline | 0.72 | 0.56 | 0.73 | 0.58 | 44/45 | 5/63 |  |
| Mask R-CNN + neighbor slices | outline | 0.69 | 0.53 | 0.72 | 0.56 | 41/45 | 2/63 |  |
| Mask R-CNN + neighbor slices + augmentation | outline | 0.71 | 0.55 | 0.72 | 0.57 | 45/45 | 8/63 |  |
| Mesh R-CNN | 3D mesh | — | — | — | — | — | — | planned |

**How to read it:**
- Box-only models are scored as filled rectangles, which also cover tissue around the structure.
  Their fair reference is the Waxholm atlas scored the same way: Dice 0.64 / IoU 0.48.
- With 9 test rats, differences of a few hundredths are ties.
- All learned models were trained on Waxholm atlas labels for 53 other rats, so the atlas is their
  ceiling until training moves to hand traces.

Details: [Waxholm baseline](auto_seg_out/WAXHOLM_BASELINE.md) · six-model baseline and experiments:
`rcnn_out/comparison.md` · SIGMA: `sigma_registration.py` docstring.
