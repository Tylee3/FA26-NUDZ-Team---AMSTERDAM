# Fine-tuning on hand traces: 3-fold preview

Median 3D Dice on the 9 hand-traced rats; every rat scored by a model that never saw it.
**Bold** = scored against the labeler the model did not learn from (the fair comparison with human vs. human).

| Model | Dice vs labeler 1 | Dice vs labeler 2 | Rats |
|---|---|---|---|
| Human vs. human (labeler 1 vs labeler 2) | 0.825 | 0.825 | 9 |
| atlas-trained (no fine-tune) | 0.739 | 0.748 | 9 |
| fine-tuned on labeler1 | 0.795 | **0.821** | 9 |
| fine-tuned on labeler2 | **0.806** | 0.830 | 9 |

Per rat (Dice vs labeler 1 / vs labeler 2):

| Rat | Human vs. human | Atlas-trained | Fine-tuned on labeler 1 | Fine-tuned on labeler 2 |
|---|---|---|---|---|
| 1 | 0.79 | 0.69 / 0.75 | 0.77 / 0.81 | 0.70 / 0.80 |
| 3 | 0.84 | 0.74 / 0.72 | 0.81 / 0.83 | 0.81 / 0.83 |
| 4 | 0.64 | 0.76 / 0.67 | 0.72 / 0.57 | 0.74 / 0.69 |
| 5 | 0.85 | 0.72 / 0.70 | 0.79 / 0.82 | 0.80 / 0.83 |
| 6 | 0.86 | 0.79 / 0.80 | 0.84 / 0.83 | 0.84 / 0.84 |
| 7 | 0.82 | 0.80 / 0.75 | 0.85 / 0.75 | 0.87 / 0.78 |
| 8 | 0.84 | 0.73 / 0.72 | 0.79 / 0.80 | 0.85 / 0.84 |
| 9 | 0.81 | 0.83 / 0.80 | 0.85 / 0.85 | 0.84 / 0.87 |
| 10 | 0.78 | 0.73 / 0.80 | 0.75 / 0.82 | 0.72 / 0.79 |
