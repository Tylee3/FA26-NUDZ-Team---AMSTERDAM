#!/bin/bash
# Trains and tests all six R-CNN-family models, then scores them against the atlas
# baseline. Models that already finished (predictions.pkl exists) are skipped, so
# re-running after an interruption picks up where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

for model in rcnn fast_rcnn faster_rcnn rfcn cascade_rcnn mask_rcnn; do
  if [ -f "rcnn_out/$model/predictions.pkl" ]; then
    echo "skipping $model (already done)"
    continue
  fi
  python3 -m rcnn.train --model "$model"
done

python3 -m rcnn.evaluate
