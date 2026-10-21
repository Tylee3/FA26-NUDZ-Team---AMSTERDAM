# What each method does, and why it scores what it scores

A companion to [RESULTS.md](RESULTS.md) and the full table in
[rcnn_out/comparison.md](rcnn_out/comparison.md) (also as `comparison.csv`). Every score below is
the median over the same 9 hand-traced test rats (1, 3–10), measured on the whole 3D volume.

## How to read the scores

- **Dice** and **IoU** both measure how much a predicted outline overlaps the hand trace:
  1.0 is a perfect match, 0 is no overlap. Dice = 2 × overlap / (size of both), IoU =
  overlap / (everything either one covers). They rank methods the same way; Dice is always the
  higher number (Dice 0.82 ≈ IoU 0.70).
- **Labeler 1** is Tyler's traces, **labeler 2** the second, independent set. Scores are shown
  as *labeler 1 / labeler 2*.
- **Slices found**: out of the 45 slices that contain the hypothalamus, how many the method
  outlined at all. **False alarms**: outlines drawn on the 63 slices that don't contain it.
- **With 9 test rats, differences under about 0.02–0.03 are ties.** One rat moves a median.

## The one idea that explains most of the table

Every learned model was trained on **atlas outlines** (the Waxholm atlas carried onto 53 other
rats), not on hand traces. A model learns to copy its teacher, so **the atlas (Dice 0.75) is the
ceiling** for all of them, and the best ones land right on it. Two pieces of evidence:

- the [learning curve](rcnn_out/learning_curve/LEARNING_CURVE.md): more atlas-labeled rats help
  up to about 40, then the score flattens at the atlas line;
- the hand-trace preview at the bottom: fine-tuned on just 6 hand-traced rats, the best model
  jumps to 0.81, next to the 0.82 humans reach with each other.

The atlas's main error is consistent: its outline sits about one slice too far back. The front
slice (slice 3) has the structure in 8 of 9 hand traces but only 31 of 53 atlas labels, and the
back slice (slice 8) in 1 of 9 hand traces but 10 of 53 atlas labels. Models trained on it
inherit that offset.

---

## Reference: humans

### Human vs. human — 0.82 Dice / 0.70 IoU
Labeler 2's traces scored as if they were a prediction of labeler 1's. This is how much two
careful people agree, so it is the realistic target, not 1.0. Where they disagree: 71% of the
disagreeing pixels are the border on the middle slices (where exactly the edge goes), 29% are the
first and last slice. On only 2 slices did one person draw something and the other nothing.
Rat 4 is hard for everyone (humans agree at only 0.64).

## Atlas registration (no training)

### Waxholm atlas, rigid (Charley's method) — 0.75 / 0.79 Dice, IoU 0.60 / 0.65
Lines up a reference rat brain (the Waxholm atlas) with each scan by rotating and shifting it
only, then copies the atlas's hypothalamus outline (label 48) onto the scan. No stretching.
**Why 0.75:** it finds the structure reliably (41/45 slices, no false alarms), but every rat's
brain is a little different in size and shape, and a rigid fit can't bend to match; and the
atlas's outline sits about one slice back compared with how the traces are drawn.

### Waxholm atlas as a box — 0.64 / 0.69
The same atlas result, but drawn as a filled rectangle around the outline. Not a method of its
own: it's the fair yardstick for the box-only detectors below, which can only draw rectangles.

### Waxholm atlas, deformable (settings a, b, c) — 0.69, 0.74, 0.75 vs labeler 1
After the rigid fit, a B-spline step lets the atlas bend locally to match the scan. Three
settings were fixed in advance, from loose to stiff. **Why no gain:** the bending follows image
contrast, and the hypothalamus has weak edges on these 1 mm-thick T2 slices, so there is little
for the warp to lock onto. The loosest setting (a) bends the wrong way and gets worse. The error
that matters (where the label sits, one slice back) is in the atlas's definition, not its shape,
so bending can't fix it.

### SIGMA atlas (4 variants) — best 0.70 Dice (waxholm_init); others 0.68, 0.64, 0.63
A second public atlas (SIGMA, built from living Wistar rats), registered the same way. Variants:
plain, with a brain mask, starting from the Waxholm fit, and Waxholm start + refine.
**Why lower than Waxholm:** SIGMA draws the hypothalamus differently. Its label is smaller (18.8 mm³
vs 26.8), a narrower, taller column, and never reaches slice 3. NUDZ's traces follow the Waxholm
definition, so even a perfect SIGMA alignment would score lower. It also places the outline
partly behind where the traces are, and fails badly on rats 7 and 10.

## The six R-CNN family models (NUDZ's original list)

All six look at one 2D slice at a time and were trained on the 53 atlas-labeled rats. The first
five draw **boxes** only, so they are scored as filled rectangles against the outline. Their fair
reference is the atlas-as-a-box at 0.64.

### R-CNN — 0.25 / 0.21
The original 2014 design: an older algorithm (Selective Search) proposes about 2,000 candidate
regions per image, a CNN describes each one, and a simple classifier (an SVM) says yes or no.
**Why so low:** Selective Search looks for regions of uniform colour and texture, which works on
photos but not on MRI. Only about 16% of its candidates overlap the hypothalamus well enough to be
usable, so the model can only find what those candidates happen to cover (11/45 slices).

### Fast R-CNN — 0.00 / 0.00
The 2015 speed-up: the CNN runs once per image, and each candidate region is cut out of its
features. Same Selective Search candidates. **Why zero:** it inherits the same bad candidates and
almost never gets a box confident enough to count (5/45 slices, so the median rat gets nothing). It is the clearest proof that the candidate step was the bottleneck.

### Faster R-CNN — 0.65 / 0.69
The 2015 fix: the network learns its own candidates (a "region proposal network") instead of using
Selective Search. **Why 0.65:** it now finds the structure (39/45 slices, no false alarms), and
its box is about as good as the atlas's box (0.64). The rest of the gap to 0.75 is just that a
rectangle covers tissue around a rounded structure.

### R-FCN — 0.62 / 0.65
A 2016 variant that makes the box decision with "position-sensitive" maps (each part of the box
votes on whether its part of the object is there), which is faster. **Why a bit lower:** slightly
looser boxes and a couple fewer slices found (37/45). Within noise of Faster R-CNN.

### Cascade R-CNN — 0.65 / 0.72
Three box stages in a row, each trained to a stricter overlap standard (0.5, 0.6, 0.7), so the box
is tightened step by step. **Why about the same:** the boxes get tighter, but a box is a box, so it
hits the same rectangle ceiling as Faster R-CNN.

### Mask R-CNN — 0.74 / 0.73, IoU 0.59 / 0.57
Faster R-CNN plus a small extra branch that draws the actual outline inside each box. **Why 0.74:**
it's the first model that outputs an outline, and it lands right at the atlas it learned from
(0.75). It misses some thin end slices, where the structure is small (38/45).

## Experiments to push past the six

### Mask R-CNN + augmentation — 0.72 / 0.73
During training, each slice is randomly flipped, slightly rotated, shifted, zoomed and
brightened, so the model sees more variety. **Why not higher:** it gets better at finding thin end
slices (44/45), but it also learns the atlas's habit of drawing on slice 8, so it raises 5 false
alarms there, all on slice 8. Better at finding, slightly worse overall.

### Mask R-CNN + neighbor slices — 0.69 / 0.72
The model sees the slice below and above as extra input channels, so it knows where the structure
is tapering off. **Why lower (likely):** the 3D context teaches it more faithfully where the
*atlas* starts and stops, which is the one-slice-back offset, so it copies the atlas's mistake
more precisely.

### Mask R-CNN + neighbor slices + augmentation — 0.71 / 0.72
Both together: finds every slice (45/45) but has the most false alarms (8, on slice 8).

### Zoomed Mask R-CNN, box first (Karina's idea) — 0.78 / 0.74, IoU 0.63 / 0.59
Two steps: Faster R-CNN finds the box, then a window twice that size is cut out, enlarged to
256×256, and a Mask R-CNN trained only on such close-ups draws the outline. **Why the best single
model:** the hypothalamus is small (about 40 pixels across), and at the close-up the network sees
its edges in much more detail. It can only outline slices step 1 found (39/45).

### Zoomed Mask R-CNN + augmentation — 0.75 / 0.75
The same with the random alterations. **A tie with plain zoom:** a lower median against labeler 1,
but better on more individual rats (6 of 9, and 8 of 9 against labeler 2).

### 3D U-Net — 0.72 / 0.72; with augmentation 0.75 / 0.76
A 3D network that segments the whole 12-slice scan at once instead of slice by slice, so every
decision sees the slices around it. **Why augmentation helps it most:** 3D means one training
example per rat, only 53 in total, so the network is short on examples; the random alterations
make up for some of that. With them it reaches the atlas level.

### Mesh R-CNN — 0.76 / 0.75, IoU 0.62 / 0.60
From one slice (the one where the structure looks biggest), it predicts the whole 3D shape: first
a coarse block grid, then a triangle surface that's smoothed into shape. The surface is cut back
into slices to score it. **Why it does well:** because it predicts the whole shape at once, it
never misses an end slice (45/45, the only model to do so). Its outline is coarser (the grid is
32×32 in-plane), which costs a little on the edges. Also writes a 3D file per rat you can open in
3D Slicer.

### Mirror pass (on the zoomed models and the 3D U-Net) — +0.00 to +0.01
Each model also looks at the scan flipped left-right, and the two answers are averaged. **Why
almost no change:** the models were already nearly symmetric, so the mirror view mostly agrees
with the original. A tie: not worth the extra step.

## Combining models

### Vote of 5 — 0.79 / 0.80, IoU 0.66 / 0.67 (with mirror pass: 0.80 / 0.79)
A pixel counts as hypothalamus when at least 3 of 5 methods say so: the atlas, both zoomed models,
Mesh R-CNN, and the 3D U-Net with augmentation (fixed before scoring). **Why it beats every
member:** the five make different mistakes (the zoomed model has the best edges, Mesh R-CNN never
misses an end slice, the atlas comes from registration rather than learning), and a majority vote
cancels mistakes that only one or two make. **The best method that uses only atlas labels.**

## Diagnostics

### Learning curve — 13 rats 0.70, 26 rats 0.74, 40 rats 0.76, 53 rats 0.78 (vs labeler 1)
The zoomed model retrained on random subsets of the training rats (two subsets per size, same
training length). **What it shows:** more rats help up to about 40, then the score flattens at the
atlas level, and against labeler 2 it doesn't rise past 26 at all. More atlas-labeled scans won't
break the ceiling. Two random subsets of the same size can differ by up to 0.08, more than the
whole 40→53 step.

### Hand-trace preview — 0.81 against the labeler it did not learn from
The zoomed model, fine-tuned on the hand traces of 6 of the 9 rats and scored on the other 3,
rotated three times so every rat is scored once by a model that never saw it. Done once with each
labeler's traces.

| Fine-tuned on | vs labeler 1 | vs labeler 2 |
|---|---|---|
| labeler 1's traces | 0.81 | **0.81** |
| labeler 2's traces | **0.81** | 0.84 |

**Bold** is the fair comparison with human vs. human (0.82): the model is judged against a person
it never learned from, exactly as one labeler is judged against the other. **Why it jumps:** for the
first time the model learns from the right teacher. Even 6 traced rats move it 0.04–0.07 and
improve 7 of 9 rats. Against the labeler it trained on, it reaches 0.84, above human vs. human,
because it has picked up that person's style. It's still held-out rats, but that number isn't the
fair one. **Caveat:** 9 rats and 3 folds make this a preview. The real test comes with NUDZ's
new traces (around Oct 30).
