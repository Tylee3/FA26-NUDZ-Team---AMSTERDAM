# Round 9: the 5 high-resolution rats

Median 3D Dice / IoU. Expert 1 = Jaison (all 5 rats); expert 2 traced rats 50 and 60 only, so the expert 2 columns are 2 rats. No model trained on any of these rats or on this scan protocol. Fine-tuned R-CNN rows average their 3 fold models.

## 1 mm slices (like the training scans)

| Model | Dice vs expert 1 | IoU vs expert 1 | Dice vs expert 2 | IoU vs expert 2 | Slices found | False alarms |
|---|---|---|---|---|---|---|
| Expert 2 vs expert 1 (human vs human, rats 50, 60) | 0.900 | 0.821 | — | — | | |
| R-CNN 2.5D, fine-tuned on expert 1 | 0.606 | 0.435 | 0.658 | 0.492 | 18/26 | 2 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.580 | 0.409 | 0.659 | 0.495 | 18/26 | 2 |
| R-CNN 2D, fine-tuned on both experts | 0.646 | 0.477 | 0.639 | 0.471 | 16/26 | 1 |
| R-CNN 2D, fine-tuned on expert 1 | 0.634 | 0.464 | 0.664 | 0.497 | 16/26 | 1 |
| R-CNN 2D, fine-tuned on expert 2 | 0.569 | 0.398 | 0.545 | 0.375 | 16/26 | 1 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.480 | 0.316 | 0.482 | 0.319 | 21/26 | 13 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.439 | 0.281 | 0.490 | 0.330 | 18/26 | 11 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.520 | 0.351 | 0.317 | 0.217 | 20/26 | 15 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.516 | 0.347 | 0.533 | 0.364 | 16/26 | 10 |
| R-CNN 2.5D, atlas-trained | 0.370 | 0.227 | 0.459 | 0.298 | 17/26 | 2 |
| R-CNN 2D, atlas-trained | 0.428 | 0.272 | 0.512 | 0.345 | 18/26 | 2 |

Per rat, Dice vs expert 1:

| Model | Rat 50 | Rat 60 | Rat 75 | Rat 90 | Rat 110 |
|---|---|---|---|---|---|
| R-CNN 2.5D, fine-tuned on expert 1 | 0.65 | 0.61 | 0.65 | 0.50 | 0.52 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.64 | 0.58 | 0.68 | 0.51 | 0.52 |
| R-CNN 2D, fine-tuned on both experts | 0.60 | 0.61 | 0.73 | 0.65 | 0.66 |
| R-CNN 2D, fine-tuned on expert 1 | 0.62 | 0.63 | 0.71 | 0.65 | 0.63 |
| R-CNN 2D, fine-tuned on expert 2 | 0.47 | 0.54 | 0.75 | 0.57 | 0.58 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.48 | 0.54 | 0.60 | 0.15 | 0.06 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.44 | 0.60 | 0.00 | 0.44 | 0.23 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.63 | 0.06 | 0.67 | 0.52 | 0.06 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.58 | 0.52 | 0.66 | 0.39 | 0.25 |
| R-CNN 2.5D, atlas-trained | 0.41 | 0.44 | 0.37 | 0.29 | 0.29 |
| R-CNN 2D, atlas-trained | 0.44 | 0.53 | 0.43 | 0.39 | 0.24 |

## 0.2 mm slices (native)

| Model | Dice vs expert 1 | IoU vs expert 1 | Dice vs expert 2 | IoU vs expert 2 | Slices found | False alarms |
|---|---|---|---|---|---|---|
| Expert 2 vs expert 1 (human vs human, rats 50, 60) | 0.853 | 0.749 | — | — | | |
| R-CNN 2.5D, fine-tuned on expert 1 | 0.540 | 0.370 | 0.593 | 0.423 | 71/123 | 2 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.524 | 0.355 | 0.581 | 0.412 | 69/123 | 2 |
| R-CNN 2D, fine-tuned on both experts | 0.522 | 0.353 | 0.558 | 0.387 | 63/123 | 3 |
| R-CNN 2D, fine-tuned on expert 1 | 0.590 | 0.418 | 0.642 | 0.474 | 69/123 | 3 |
| R-CNN 2D, fine-tuned on expert 2 | 0.350 | 0.212 | 0.420 | 0.269 | 60/123 | 3 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.355 | 0.216 | 0.283 | 0.176 | 118/123 | 130 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.534 | 0.364 | 0.459 | 0.299 | 106/123 | 51 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.121 | 0.065 | 0.289 | 0.180 | 88/123 | 94 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.483 | 0.318 | 0.507 | 0.340 | 102/123 | 52 |
| R-CNN 2.5D, atlas-trained | 0.398 | 0.249 | 0.475 | 0.312 | 75/123 | 2 |
| R-CNN 2D, atlas-trained | 0.434 | 0.277 | 0.536 | 0.366 | 76/123 | 3 |

Per rat, Dice vs expert 1:

| Model | Rat 50 | Rat 60 | Rat 75 | Rat 90 | Rat 110 |
|---|---|---|---|---|---|
| R-CNN 2.5D, fine-tuned on expert 1 | 0.54 | 0.52 | 0.63 | 0.27 | 0.54 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.52 | 0.50 | 0.64 | 0.28 | 0.58 |
| R-CNN 2D, fine-tuned on both experts | 0.44 | 0.56 | 0.65 | 0.31 | 0.52 |
| R-CNN 2D, fine-tuned on expert 1 | 0.56 | 0.59 | 0.65 | 0.36 | 0.62 |
| R-CNN 2D, fine-tuned on expert 2 | 0.26 | 0.49 | 0.62 | 0.29 | 0.35 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.49 | 0.12 | 0.48 | 0.36 | 0.07 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.53 | 0.43 | 0.69 | 0.65 | 0.42 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.51 | 0.12 | 0.00 | 0.39 | 0.08 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.59 | 0.48 | 0.61 | 0.35 | 0.43 |
| R-CNN 2.5D, atlas-trained | 0.42 | 0.42 | 0.40 | 0.15 | 0.38 |
| R-CNN 2D, atlas-trained | 0.44 | 0.53 | 0.43 | 0.27 | 0.30 |

