#!/bin/bash
# Improvement round 1 on Mask R-CNN, one change at a time against the existing baseline
# (rcnn_out/mask_rcnn): neighbor slices, augmentation, and both. About 1 hour per run.
# Finished runs are skipped, so re-running after an interruption picks up where it left off.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

run() {
  local dir="$1"; shift
  if [ -f "rcnn_out/$dir/predictions.pkl" ]; then
    echo "skipping $dir (already done)"
    return
  fi
  python3 -m rcnn.train --model mask_rcnn "$@"
}

run mask_rcnn_neighbors --neighbors
run mask_rcnn_augment --augment
run mask_rcnn_neighbors_augment --neighbors --augment

python3 -m rcnn.evaluate
