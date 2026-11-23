# Fine-tuning on hand traces: 3-fold (14 rats)

Median 3D Dice on the 14 hand-traced rats; every rat scored by a model that never saw it.
**Bold** = scored against the labeler the model did not learn from (the fair comparison with human vs. human).

| Model | Dice vs labeler 1 | Dice vs labeler 2 | Rats |
|---|---|---|---|
| Human vs. human (labeler 1 vs labeler 2) | 0.824 | 0.824 | 14 |
| atlas-trained (no fine-tune) | 0.734 | 0.729 | 14 |
| fine-tuned on labeler1 | 0.797 | **0.838** | 14 |
| fine-tuned on labeler2 | **0.787** | 0.826 | 14 |

Per rat (Dice vs labeler 1 / vs labeler 2):

| Rat | Human vs. human | Atlas-trained | Fine-tuned on labeler 1 | Fine-tuned on labeler 2 |
|---|---|---|---|---|
| 1 | 0.79 | 0.65 / 0.73 | 0.78 / 0.89 | 0.76 / 0.90 |
| 3 | 0.84 | 0.74 / 0.72 | 0.82 / 0.84 | 0.81 / 0.84 |
| 4 | 0.76 | 0.68 / 0.71 | 0.69 / 0.80 | 0.70 / 0.81 |
| 5 | 0.85 | 0.74 / 0.70 | 0.81 / 0.84 | 0.83 / 0.86 |
| 6 | 0.86 | 0.75 / 0.77 | 0.84 / 0.86 | 0.84 / 0.85 |
| 7 | 0.82 | 0.77 / 0.74 | 0.84 / 0.77 | 0.88 / 0.81 |
| 8 | 0.84 | 0.72 / 0.73 | 0.84 / 0.84 | 0.87 / 0.86 |
| 9 | 0.81 | 0.77 / 0.79 | 0.84 / 0.86 | 0.83 / 0.87 |
| 10 | 0.78 | 0.73 / 0.79 | 0.77 / 0.86 | 0.75 / 0.86 |
| 15 | 0.82 | 0.38 / 0.32 | 0.62 / 0.54 | 0.61 / 0.54 |
| 25 | 0.81 | 0.74 / 0.76 | 0.77 / 0.84 | 0.77 / 0.81 |
| 35 | 0.85 | 0.59 / 0.62 | 0.76 / 0.76 | 0.74 / 0.71 |
| 100 | 0.83 | 0.79 / 0.75 | 0.81 / 0.74 | 0.86 / 0.78 |
| 125 | 0.80 | 0.52 / 0.54 | 0.75 / 0.76 | 0.75 / 0.74 |
