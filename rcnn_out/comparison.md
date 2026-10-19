| model | output | rats | slices_found | false_alarms | mean_box_iou | median_3d_iou_vs_me | median_3d_dice_vs_me | median_3d_iou_vs_other |
|---|---|---|---|---|---|---|---|---|
| Atlas (Charley's method) | mask | 9 | 41/45 | 0/63 | 0.556 | 0.599 | 0.749 | 0.647 |
| Atlas, as a box | box | 9 | 41/45 | 0/63 | 0.556 | 0.475 | 0.644 | 0.529 |
| R-CNN | box | 9 | 11/45 | 8/63 | 0.096 | 0.142 | 0.249 | 0.119 |
| Fast R-CNN | box | 9 | 5/45 | 0/63 | 0.058 | 0.0 | 0.0 | 0.0 |
| Faster R-CNN | box | 9 | 39/45 | 0/63 | 0.53 | 0.476 | 0.645 | 0.521 |
| R-FCN | box | 9 | 37/45 | 1/63 | 0.493 | 0.446 | 0.617 | 0.48 |
| Cascade R-CNN | box | 9 | 35/45 | 0/63 | 0.491 | 0.482 | 0.651 | 0.56 |
| Mask R-CNN | mask | 9 | 38/45 | 0/63 | 0.527 | 0.586 | 0.739 | 0.572 |
| Mask R-CNN + augmentation | mask | 9 | 44/45 | 5/63 | 0.544 | 0.557 | 0.716 | 0.576 |
| Mask R-CNN + neighbor slices | mask | 9 | 41/45 | 2/63 | 0.534 | 0.526 | 0.69 | 0.563 |
| Mask R-CNN + neighbor slices + augmentation | mask | 9 | 45/45 | 8/63 | 0.55 | 0.547 | 0.707 | 0.566 |
| Mesh R-CNN | mask | 9 | 45/45 | 1/63 | 0.569 | 0.616 | 0.762 | 0.603 |
| 3D U-Net | mask | 9 | 43/45 | 1/63 | 0.523 | 0.567 | 0.724 | 0.567 |
| 3D U-Net + augmentation | mask | 9 | 39/45 | 1/63 | 0.551 | 0.594 | 0.745 | 0.619 |
| Mask R-CNN + zoom (box first) | mask | 9 | 39/45 | 0/63 | 0.586 | 0.634 | 0.776 | 0.59 |
| Mask R-CNN + augmentation + zoom (box first) | mask | 9 | 39/45 | 0/63 | 0.576 | 0.603 | 0.752 | 0.605 |
