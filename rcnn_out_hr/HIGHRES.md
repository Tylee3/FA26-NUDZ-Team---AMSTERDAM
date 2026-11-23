# Round 9: the 5 high-resolution rats

Median 3D IoU / Dice. Expert 1 = Jaison (all 5 rats). Expert 2 traced only the rats listed in the human row, so the expert 2, both-agree and STAPLE columns cover only those. Both-agree = the voxels both experts traced; STAPLE = SimpleITK's estimate of the true outline from the two traces (with only two experts it equals both-agree; kept for a third). No model trained on any of these rats or on this scan protocol. Fine-tuned R-CNN rows average their 3 fold models; rows fine-tuned on both experts learned from both, so their combined-reference scores are not held out.

Noise rule: a change counts only if it gains more than 0.01 against both experts; rerunning the same model moves scores by up to 0.007.

## 1 mm slices (like the training scans)

| Model | IoU / Dice vs expert 1 | IoU / Dice vs expert 2 | IoU / Dice vs both agree | IoU / Dice vs STAPLE | Slices found | False alarms |
|---|---|---|---|---|---|---|
| Expert 2 vs expert 1 (human vs human, rats 50, 60) | 0.821 / 0.900 | — | — | — | | |
| R-CNN 2.5D, fine-tuned on expert 1 | 0.435 / 0.606 | 0.492 / 0.658 | 0.500 / 0.665 | 0.500 / 0.665 | 18/26 | 2 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.409 / 0.580 | 0.495 / 0.659 | 0.505 / 0.668 | 0.505 / 0.668 | 18/26 | 2 |
| R-CNN 2D, fine-tuned on both experts | 0.477 / 0.646 | 0.471 / 0.639 | 0.478 / 0.646 | 0.478 / 0.646 | 16/26 | 1 |
| R-CNN 2D, fine-tuned on expert 1 | 0.464 / 0.634 | 0.497 / 0.664 | 0.503 / 0.669 | 0.503 / 0.669 | 16/26 | 1 |
| R-CNN 2D, fine-tuned on expert 2 | 0.398 / 0.569 | 0.375 / 0.545 | 0.383 / 0.553 | 0.383 / 0.553 | 16/26 | 1 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.316 / 0.480 | 0.319 / 0.482 | 0.313 / 0.475 | 0.313 / 0.475 | 21/26 | 13 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.281 / 0.439 | 0.330 / 0.490 | 0.325 / 0.484 | 0.325 / 0.484 | 18/26 | 11 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.351 / 0.520 | 0.217 / 0.317 | 0.213 / 0.313 | 0.213 / 0.313 | 20/26 | 15 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.347 / 0.516 | 0.364 / 0.533 | 0.359 / 0.528 | 0.359 / 0.528 | 16/26 | 10 |
| R-CNN 2.5D, atlas-trained | 0.227 / 0.370 | 0.298 / 0.459 | 0.298 / 0.459 | 0.298 / 0.459 | 17/26 | 2 |
| R-CNN 2D, atlas-trained | 0.272 / 0.428 | 0.345 / 0.512 | 0.346 / 0.514 | 0.346 / 0.514 | 18/26 | 2 |

Per rat, IoU / Dice vs expert 1:

| Model | Rat 50 | Rat 60 | Rat 75 | Rat 90 | Rat 110 |
|---|---|---|---|---|---|
| R-CNN 2.5D, fine-tuned on expert 1 | 0.48 / 0.65 | 0.43 / 0.61 | 0.49 / 0.65 | 0.33 / 0.50 | 0.36 / 0.52 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.47 / 0.64 | 0.41 / 0.58 | 0.51 / 0.68 | 0.34 / 0.51 | 0.35 / 0.52 |
| R-CNN 2D, fine-tuned on both experts | 0.43 / 0.60 | 0.44 / 0.61 | 0.57 / 0.73 | 0.48 / 0.65 | 0.49 / 0.66 |
| R-CNN 2D, fine-tuned on expert 1 | 0.45 / 0.62 | 0.46 / 0.63 | 0.55 / 0.71 | 0.48 / 0.65 | 0.46 / 0.63 |
| R-CNN 2D, fine-tuned on expert 2 | 0.30 / 0.47 | 0.37 / 0.54 | 0.60 / 0.75 | 0.40 / 0.57 | 0.40 / 0.58 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.32 / 0.48 | 0.37 / 0.54 | 0.43 / 0.60 | 0.08 / 0.15 | 0.03 / 0.06 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.28 / 0.44 | 0.43 / 0.60 | 0.00 / 0.00 | 0.28 / 0.44 | 0.13 / 0.23 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.46 / 0.63 | 0.03 / 0.06 | 0.50 / 0.67 | 0.35 / 0.52 | 0.03 / 0.06 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.41 / 0.58 | 0.35 / 0.52 | 0.49 / 0.66 | 0.24 / 0.39 | 0.15 / 0.25 |
| R-CNN 2.5D, atlas-trained | 0.25 / 0.41 | 0.28 / 0.44 | 0.23 / 0.37 | 0.17 / 0.29 | 0.17 / 0.29 |
| R-CNN 2D, atlas-trained | 0.28 / 0.44 | 0.36 / 0.53 | 0.27 / 0.43 | 0.24 / 0.39 | 0.13 / 0.24 |

## 0.2 mm slices (native)

| Model | IoU / Dice vs expert 1 | IoU / Dice vs expert 2 | IoU / Dice vs both agree | IoU / Dice vs STAPLE | Slices found | False alarms |
|---|---|---|---|---|---|---|
| Expert 2 vs expert 1 (human vs human, rats 50, 60) | 0.749 / 0.853 | — | — | — | | |
| R-CNN 2.5D, fine-tuned on expert 1 | 0.370 / 0.540 | 0.423 / 0.593 | 0.405 / 0.576 | 0.405 / 0.576 | 71/123 | 2 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.355 / 0.524 | 0.412 / 0.581 | 0.397 / 0.566 | 0.397 / 0.566 | 69/123 | 2 |
| R-CNN 2D, fine-tuned on both experts | 0.353 / 0.522 | 0.387 / 0.558 | 0.386 / 0.557 | 0.386 / 0.557 | 63/123 | 3 |
| R-CNN 2D, fine-tuned on expert 1 | 0.418 / 0.590 | 0.474 / 0.642 | 0.461 / 0.631 | 0.461 / 0.631 | 69/123 | 3 |
| R-CNN 2D, fine-tuned on expert 2 | 0.212 / 0.350 | 0.269 / 0.420 | 0.269 / 0.421 | 0.269 / 0.421 | 60/123 | 3 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.216 / 0.355 | 0.176 / 0.283 | 0.161 / 0.265 | 0.161 / 0.265 | 118/123 | 130 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.364 / 0.534 | 0.299 / 0.459 | 0.280 / 0.437 | 0.280 / 0.437 | 106/123 | 51 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.065 / 0.121 | 0.180 / 0.289 | 0.164 / 0.268 | 0.164 / 0.268 | 88/123 | 94 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.318 / 0.483 | 0.340 / 0.507 | 0.318 / 0.482 | 0.318 / 0.482 | 102/123 | 52 |
| R-CNN 2.5D, atlas-trained | 0.249 / 0.398 | 0.312 / 0.475 | 0.299 / 0.460 | 0.299 / 0.460 | 75/123 | 2 |
| R-CNN 2D, atlas-trained | 0.277 / 0.434 | 0.366 / 0.536 | 0.350 / 0.519 | 0.350 / 0.519 | 76/123 | 3 |

Per rat, IoU / Dice vs expert 1:

| Model | Rat 50 | Rat 60 | Rat 75 | Rat 90 | Rat 110 |
|---|---|---|---|---|---|
| R-CNN 2.5D, fine-tuned on expert 1 | 0.37 / 0.54 | 0.35 / 0.52 | 0.46 / 0.63 | 0.16 / 0.27 | 0.37 / 0.54 |
| R-CNN 2.5D, fine-tuned on expert 2 | 0.36 / 0.52 | 0.33 / 0.50 | 0.47 / 0.64 | 0.16 / 0.28 | 0.41 / 0.58 |
| R-CNN 2D, fine-tuned on both experts | 0.29 / 0.44 | 0.39 / 0.56 | 0.48 / 0.65 | 0.19 / 0.31 | 0.35 / 0.52 |
| R-CNN 2D, fine-tuned on expert 1 | 0.39 / 0.56 | 0.42 / 0.59 | 0.48 / 0.65 | 0.22 / 0.36 | 0.45 / 0.62 |
| R-CNN 2D, fine-tuned on expert 2 | 0.15 / 0.26 | 0.32 / 0.49 | 0.45 / 0.62 | 0.17 / 0.29 | 0.21 / 0.35 |
| MedSAM2 (no training), the atlas-trained detector's box, whole slice | 0.32 / 0.49 | 0.06 / 0.12 | 0.32 / 0.48 | 0.22 / 0.36 | 0.03 / 0.07 |
| MedSAM2 (no training), the atlas-trained detector's box, zoomed window | 0.36 / 0.53 | 0.27 / 0.43 | 0.53 / 0.69 | 0.48 / 0.65 | 0.27 / 0.42 |
| MedSAM2 (no training), a box around expert 1's trace, whole slice | 0.34 / 0.51 | 0.06 / 0.12 | 0.00 / 0.00 | 0.24 / 0.39 | 0.04 / 0.08 |
| MedSAM2 (no training), a box around expert 1's trace, zoomed window | 0.42 / 0.59 | 0.32 / 0.48 | 0.44 / 0.61 | 0.21 / 0.35 | 0.28 / 0.43 |
| R-CNN 2.5D, atlas-trained | 0.27 / 0.42 | 0.26 / 0.42 | 0.25 / 0.40 | 0.08 / 0.15 | 0.24 / 0.38 |
| R-CNN 2D, atlas-trained | 0.28 / 0.44 | 0.36 / 0.53 | 0.28 / 0.43 | 0.15 / 0.27 | 0.18 / 0.30 |

