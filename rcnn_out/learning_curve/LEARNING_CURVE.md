# Learning curve: zoomed Mask R-CNN

Median 3D score on the 9 hand-traced test rats, by number of atlas-labeled training rats.
Every run trains for the same 3180 steps; two random subsets per size.

| Training rats | Run | Dice (lab. 1) | IoU (lab. 1) | Dice (lab. 2) |
|---|---|---|---|---|
| 13 | zoom_n13_s0 | 0.709 | 0.549 | 0.710 |
| 13 | zoom_n13_s1 | 0.683 | 0.518 | 0.721 |
| 26 | zoom_n26_s0 | 0.752 | 0.603 | 0.787 |
| 26 | zoom_n26_s1 | 0.719 | 0.561 | 0.708 |
| 40 | zoom_n40_s0 | 0.756 | 0.608 | 0.758 |
| 40 | zoom_n40_s1 | 0.768 | 0.624 | 0.759 |
| 53 | zoom_mask_rcnn (existing) | 0.776 | 0.634 | 0.742 |
