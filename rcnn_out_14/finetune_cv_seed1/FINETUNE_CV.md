# Fine-tuning on hand traces: 3-fold (14 rats)

Median 3D Dice on the 14 hand-traced rats; every rat scored by a model that never saw it.
**Bold** = scored against the labeler the model did not learn from (the fair comparison with human vs. human).

| Model | Dice vs labeler 1 | Dice vs labeler 2 | Rats |
|---|---|---|---|
| Human vs. human (labeler 1 vs labeler 2) | 0.824 | 0.824 | 14 |
| atlas-trained (no fine-tune) | 0.706 | 0.700 | 14 |
| fine-tuned on labeler1 | 0.810 | **0.806** | 14 |
| fine-tuned on labeler2 | **0.801** | 0.822 | 14 |
| fine-tuned on both | 0.809 | 0.812 | 14 |

Fine-tuned on both = each training slice once with each labeler's trace, so the model learns both; scored against each, it is compared with human vs. human the way a third labeler would be.

Per rat (Dice vs labeler 1 / vs labeler 2):

| Rat | Human vs. human | Atlas-trained | Fine-tuned on labeler 1 | Fine-tuned on labeler 2 |
|---|---|---|---|---|
| 1 | 0.79 | 0.68 / 0.70 | 0.72 / 0.80 | 0.69 / 0.80 |
| 3 | 0.84 | 0.77 / 0.72 | 0.84 / 0.84 | 0.85 / 0.84 |
| 4 | 0.64 | 0.68 / 0.48 | 0.69 / 0.55 | 0.70 / 0.57 |
| 5 | 0.85 | 0.68 / 0.66 | 0.81 / 0.86 | 0.81 / 0.85 |
| 6 | 0.86 | 0.73 / 0.73 | 0.82 / 0.81 | 0.84 / 0.85 |
| 7 | 0.82 | 0.74 / 0.70 | 0.82 / 0.76 | 0.85 / 0.83 |
| 8 | 0.84 | 0.68 / 0.68 | 0.81 / 0.85 | 0.83 / 0.86 |
| 9 | 0.81 | 0.76 / 0.74 | 0.83 / 0.86 | 0.80 / 0.86 |
| 10 | 0.78 | 0.77 / 0.74 | 0.77 / 0.83 | 0.75 / 0.83 |
| 15 | 0.82 | 0.39 / 0.36 | 0.71 / 0.68 | 0.77 / 0.75 |
| 25 | 0.81 | 0.73 / 0.75 | 0.81 / 0.81 | 0.80 / 0.81 |
| 35 | 0.85 | 0.54 / 0.59 | 0.75 / 0.74 | 0.79 / 0.76 |
| 100 | 0.83 | 0.76 / 0.71 | 0.81 / 0.74 | 0.85 / 0.75 |
| 125 | 0.80 | 0.45 / 0.48 | 0.76 / 0.75 | 0.75 / 0.73 |
