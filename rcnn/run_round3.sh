#!/bin/bash
# Round 3, before the new hand traces: (1) fine-tuning the zoomed Mask R-CNN on the hand
# traces we have, 3 folds x 2 labelers (about 70 min); (2) the vote of 5 and the mirror pass
# (a few minutes); then rescoring everything and rebuilding RESULTS.md.
# Finished fine-tune folds are skipped, so re-running after an interruption picks up.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

python3 -m rcnn.finetune_cv
python3 -m rcnn.ensemble
python3 -m rcnn.evaluate
python3 make_results_table.py
