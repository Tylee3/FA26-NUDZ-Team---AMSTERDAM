# Round 8: end-slice rescue (no retraining)

Same fold models with and without the rule (bar 0.2, fixed in advance). Median 3D Dice over the 14 traced rats, each scored by a fold that never saw it. **Bold** = the fair score, against the expert the model did not learn from. Human vs. human is 0.824.

| Seed | Trained on | Slices found | False alarms | Dice vs expert 1 | Dice vs expert 2 | Mean Dice vs expert 1 | Mean Dice vs expert 2 |
|---|---|---|---|---|---|---|---|
| 0 | expert 1 | 61 → 67 of 69 | 2 → 15 | 0.806 → 0.746 | **0.799** → **0.737** | 0.779 → 0.741 | 0.774 → 0.737 |
| 0 | expert 2 | 61 → 67 of 69 | 2 → 13 | **0.798** → **0.777** | 0.811 → 0.774 | 0.787 → 0.750 | 0.788 → 0.755 |
| 0 | both experts | 61 → 67 of 69 | 2 → 18 | 0.800 → 0.733 | 0.803 → 0.747 | 0.781 → 0.736 | 0.778 → 0.733 |
| 1 | expert 1 | 62 → 67 of 69 | 2 → 13 | 0.810 → 0.742 | **0.807** → **0.754** | 0.783 → 0.743 | 0.778 → 0.738 |
| 1 | expert 2 | 61 → 67 of 69 | 2 → 16 | **0.800** → **0.770** | 0.822 → 0.759 | 0.791 → 0.746 | 0.792 → 0.750 |
| 1 | both experts | 61 → 67 of 69 | 2 → 17 | 0.809 → 0.750 | 0.812 → 0.737 | 0.783 → 0.738 | 0.781 → 0.738 |
| 2 | expert 1 | 61 → 66 of 69 | 2 → 15 | 0.806 → 0.737 | **0.805** → **0.747** | 0.777 → 0.738 | 0.772 → 0.732 |
| 2 | expert 2 | 62 → 67 of 69 | 2 → 13 | **0.797** → **0.774** | 0.815 → 0.771 | 0.791 → 0.757 | 0.788 → 0.759 |
| 2 | both experts | 62 → 67 of 69 | 2 → 16 | 0.808 → 0.746 | 0.816 → 0.742 | 0.785 → 0.743 | 0.782 → 0.743 |

Fair median change, averaged over the 3 seeds and both single-expert models: -0.041 (keep rule: more than +0.01; seed noise is about 0.007).

The rule mostly reaches the weakest rats, which a median barely registers, so the mean is shown too.

Does the exact bar matter? Fair median change averaged the same way, per bar:

| Bar | 0.1 | 0.2 | 0.3 |
|---|---|---|---|
| Change | -0.045 | -0.041 | -0.039 |

Per rat, seed 0, trained on expert 2, Dice vs expert 1 (fair), without → with:

| Rat | Without | With |
|---|---|---|
| 1 | 0.69 | 0.64 ← |
| 3 | 0.85 | 0.81 ← |
| 4 | 0.69 | 0.73 ← |
| 5 | 0.80 | 0.79 ← |
| 6 | 0.84 | 0.79 ← |
| 7 | 0.83 | 0.83 |
| 8 | 0.84 | 0.78 ← |
| 9 | 0.81 | 0.74 ← |
| 10 | 0.74 | 0.63 ← |
| 15 | 0.76 | 0.74 ← |
| 25 | 0.80 | 0.80 |
| 35 | 0.79 | 0.77 ← |
| 100 | 0.83 | 0.81 ← |
| 125 | 0.74 | 0.65 ← |
