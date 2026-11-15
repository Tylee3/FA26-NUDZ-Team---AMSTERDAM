| group | model | output | rats | slices_found | false_alarms | mean_box_iou | median_3d_iou_vs_me | median_3d_dice_vs_me | median_3d_iou_vs_other | median_3d_dice_vs_other |
|---|---|---|---|---|---|---|---|---|---|---|
| human | Labeler 2's traces (human vs. human) | mask | 14 | 68/69 | 0/99 | 0.675 | 0.701 | 0.824 | — | — |
| atlas | Atlas (Charley's method) | mask | 14 | 62/69 | 1/99 | 0.509 | 0.547 | 0.707 | 0.582 | 0.736 |
| atlas | Atlas, as a box | box | 14 | 62/69 | 1/99 | 0.509 | 0.471 | 0.641 | 0.52 | 0.684 |
| R-CNN family | Faster R-CNN | box | 14 | 64/69 | 3/99 | 0.527 | 0.484 | 0.653 | 0.501 | 0.667 |
| experiments | Average of the 3 fine-tuned models (labeler 1, labeler 2, both) + mirror pass (3-fold) | mask | 14 | 62/69 | 2/99 | 0.62 | 0.672 | 0.804 | 0.702 | 0.825 |
| experiments | Mask R-CNN + zoom, fine-tuned on both labelers' traces + mirror pass (3-fold) | mask | 14 | 60/69 | 2/99 | 0.613 | 0.665 | 0.798 | 0.689 | 0.816 |
| experiments | Mask R-CNN + zoom, fine-tuned on both experts' traces + end-slice rescue (3-fold) | mask | 14 | 67/69 | 18/99 | 0.651 | 0.578 | 0.733 | 0.614 | 0.761 |
| experiments | Mask R-CNN + zoom, fine-tuned on expert 1's traces + end-slice rescue (3-fold) | mask | 14 | 67/69 | 15/99 | 0.648 | 0.595 | 0.746 | 0.607 | 0.756 |
| experiments | Mask R-CNN + zoom, fine-tuned on expert 2's traces + end-slice rescue (3-fold) | mask | 14 | 67/69 | 13/99 | 0.642 | 0.635 | 0.777 | 0.665 | 0.799 |
| experiments | Mask R-CNN + zoom (box first) | mask | 14 | 64/69 | 3/99 | 0.532 | 0.545 | 0.706 | 0.543 | 0.704 |
| hand-trace preview | Mask R-CNN + zoom, fine-tuned on labeler 1's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.613 | 0.675 | 0.806 | 0.68 | 0.81 |
| hand-trace preview | Mask R-CNN + zoom, fine-tuned on labeler 2's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.618 | 0.663 | 0.798 | 0.686 | 0.814 |
| hand-trace preview | Mask R-CNN + zoom, fine-tuned on both labelers' hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.615 | 0.666 | 0.8 | 0.689 | 0.816 |
| hand-trace preview | Mask R-CNN + zoom + seed1, fine-tuned on labeler 1's hand traces (3-fold) | mask | 14 | 62/69 | 2/99 | 0.617 | 0.68 | 0.81 | 0.684 | 0.812 |
| hand-trace preview | Mask R-CNN + zoom + seed1, fine-tuned on labeler 2's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.625 | 0.668 | 0.801 | 0.708 | 0.829 |
| hand-trace preview | Mask R-CNN + zoom + seed1, fine-tuned on both labelers' hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.613 | 0.679 | 0.809 | 0.698 | 0.822 |
| hand-trace preview | Mask R-CNN + zoom + seed2, fine-tuned on labeler 1's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.605 | 0.675 | 0.806 | 0.682 | 0.811 |
| hand-trace preview | Mask R-CNN + zoom + seed2, fine-tuned on labeler 2's hand traces (3-fold) | mask | 14 | 62/69 | 2/99 | 0.629 | 0.662 | 0.797 | 0.696 | 0.821 |
| hand-trace preview | Mask R-CNN + zoom + seed2, fine-tuned on both labelers' hand traces (3-fold) | mask | 14 | 62/69 | 2/99 | 0.621 | 0.678 | 0.808 | 0.697 | 0.821 |
| hand-trace preview | Mask R-CNN + zoom + warp, fine-tuned on labeler 1's hand traces (3-fold) | mask | 14 | 61/69 | 2/99 | 0.612 | 0.666 | 0.8 | 0.694 | 0.82 |
| hand-trace preview | Mask R-CNN + zoom + warp, fine-tuned on labeler 2's hand traces (3-fold) | mask | 14 | 63/69 | 2/99 | 0.632 | 0.662 | 0.797 | 0.687 | 0.814 |
