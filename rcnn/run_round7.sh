#!/bin/bash
# Round 7, on all 14 traced rats (HYPO_TEST=14), starting from round 6's models:
#   1. fine-tune on both labelers' traces at once (finetune_cv/both, ~30 min)
#   2. mirror pass on it, and the average of the 3 fine-tuned models with mirror (~10 min)
#   3. score everything, so steps 1-2 can be read while step 4 runs
#   4. the noise check: the plain and both fine-tunes repeated with seeds 1 and 2 (~3 h),
#      so we know how big a difference has to be before it is more than chance
#   5. score again
# About 4 hours. Finished folds are skipped, so it can be stopped and restarted.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HYPO_TEST=14

python3 -m rcnn.finetune_cv --both
python3 -m rcnn.finetune_combine
python3 -m rcnn.evaluate
for seed in 1 2; do
  python3 -m rcnn.finetune_cv --seed "$seed"
  python3 -m rcnn.finetune_cv --seed "$seed" --both
done
python3 -m rcnn.evaluate
