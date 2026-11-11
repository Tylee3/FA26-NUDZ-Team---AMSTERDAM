| group | model | output | rats | slices_found | false_alarms | mean_box_iou | median_3d_iou_vs_me | median_3d_dice_vs_me | median_3d_iou_vs_other | median_3d_dice_vs_other |
|---|---|---|---|---|---|---|---|---|---|---|
| human | Labeler 2's traces (human vs. human) | mask | 14 | 67/69 | 1/99 | 0.661 | 0.701 | 0.824 | — | — |
| atlas | Atlas (Charley's method) | mask | 14 | 62/69 | 1/99 | 0.509 | 0.547 | 0.707 | 0.582 | 0.736 |
| atlas | Atlas, as a box | box | 14 | 62/69 | 1/99 | 0.509 | 0.471 | 0.641 | 0.52 | 0.684 |
| R-CNN family | Faster R-CNN | box | 14 | 64/69 | 3/99 | 0.527 | 0.484 | 0.653 | 0.501 | 0.667 |
| experiments | Mask R-CNN + zoom (box first) | mask | 14 | 64/69 | 3/99 | 0.532 | 0.545 | 0.706 | 0.539 | 0.7 |
| hand-trace preview | Mask R-CNN + zoom, fine-tuned on labeler 1's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.613 | 0.675 | 0.806 | 0.665 | 0.799 |
| hand-trace preview | Mask R-CNN + zoom, fine-tuned on labeler 2's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.618 | 0.663 | 0.798 | 0.681 | 0.81 |
| hand-trace preview | Mask R-CNN + zoom + warp, fine-tuned on labeler 1's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.612 | 0.666 | 0.8 | 0.67 | 0.802 |
| hand-trace preview | Mask R-CNN + zoom + warp, fine-tuned on labeler 2's hand traces (3-fold) | mask | 14 | 63/69 | 2/99 | 0.632 | 0.662 | 0.797 | 0.678 | 0.808 |
