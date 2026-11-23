# Fine-tuning on hand traces: 3-fold (14 rats)

Median 3D Dice on the 14 hand-traced rats; every rat scored by a model that never saw it.
**Bold** = scored against the labeler the model did not learn from (the fair comparison with human vs. human).

| Model | Dice vs labeler 1 | Dice vs labeler 2 | Rats |
|---|---|---|---|
| Human vs. human (labeler 1 vs labeler 2) | 0.824 | 0.824 | 14 |
| atlas-trained (no fine-tune) | 0.734 | 0.729 | 14 |
| fine-tuned on labeler1 | 0.791 | **0.839** | 14 |
| fine-tuned on labeler2 | **0.793** | 0.816 | 14 |

Per rat (Dice vs labeler 1 / vs labeler 2):

| Rat | Human vs. human | Atlas-trained | Fine-tuned on labeler 1 | Fine-tuned on labeler 2 |
|---|---|---|---|---|
| 1 | 0.79 | 0.65 / 0.73 | 0.79 / 0.90 | 0.66 / 0.79 |
| 3 | 0.84 | 0.74 / 0.72 | 0.82 / 0.84 | 0.82 / 0.85 |
| 4 | 0.76 | 0.68 / 0.71 | 0.70 / 0.81 | 0.70 / 0.81 |
| 5 | 0.85 | 0.74 / 0.70 | 0.81 / 0.85 | 0.82 / 0.86 |
| 6 | 0.86 | 0.75 / 0.77 | 0.84 / 0.85 | 0.83 / 0.84 |
| 7 | 0.82 | 0.77 / 0.74 | 0.89 / 0.77 | 0.87 / 0.80 |
| 8 | 0.84 | 0.72 / 0.73 | 0.85 / 0.84 | 0.87 / 0.86 |
| 9 | 0.81 | 0.77 / 0.79 | 0.84 / 0.86 | 0.84 / 0.88 |
| 10 | 0.78 | 0.73 / 0.79 | 0.77 / 0.85 | 0.77 / 0.85 |
| 15 | 0.82 | 0.38 / 0.32 | 0.68 / 0.64 | 0.70 / 0.66 |
| 25 | 0.81 | 0.74 / 0.76 | 0.77 / 0.85 | 0.74 / 0.82 |
| 35 | 0.85 | 0.59 / 0.62 | 0.72 / 0.72 | 0.73 / 0.72 |
| 100 | 0.83 | 0.79 / 0.75 | 0.79 / 0.73 | 0.85 / 0.78 |
| 125 | 0.80 | 0.52 / 0.54 | 0.68 / 0.68 | 0.69 / 0.67 |
