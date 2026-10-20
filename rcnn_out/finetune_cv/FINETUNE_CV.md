# Fine-tuning on hand traces: 3-fold preview

Median 3D Dice on the 9 hand-traced rats; every rat scored by a model that never saw it.
**Bold** = scored against the labeler the model did not learn from (the fair comparison with human vs. human).

| Model | Dice vs labeler 1 | Dice vs labeler 2 | Rats |
|---|---|---|---|
| Human vs. human (labeler 1 vs labeler 2) | 0.825 | 0.825 | 9 |
| atlas-trained (no fine-tune) | 0.776 | 0.742 | 9 |
| fine-tuned on labeler1 | 0.812 | **0.809** | 9 |
| fine-tuned on labeler2 | **0.813** | 0.841 | 9 |

Per rat (Dice vs labeler 1 / vs labeler 2):

| Rat | Human vs. human | Atlas-trained | Fine-tuned on labeler 1 | Fine-tuned on labeler 2 |
|---|---|---|---|---|
| 1 | 0.79 | 0.68 / 0.74 | 0.75 / 0.78 | 0.67 / 0.78 |
| 3 | 0.84 | 0.79 / 0.77 | 0.85 / 0.82 | 0.82 / 0.85 |
| 4 | 0.64 | 0.73 / 0.62 | 0.75 / 0.59 | 0.75 / 0.62 |
| 5 | 0.85 | 0.74 / 0.73 | 0.78 / 0.81 | 0.81 / 0.86 |
| 6 | 0.86 | 0.78 / 0.79 | 0.84 / 0.83 | 0.84 / 0.84 |
| 7 | 0.82 | 0.81 / 0.73 | 0.84 / 0.75 | 0.86 / 0.81 |
| 8 | 0.84 | 0.73 / 0.73 | 0.81 / 0.81 | 0.80 / 0.82 |
| 9 | 0.81 | 0.80 / 0.78 | 0.85 / 0.84 | 0.84 / 0.86 |
| 10 | 0.78 | 0.80 / 0.81 | 0.80 / 0.74 | 0.80 / 0.84 |
