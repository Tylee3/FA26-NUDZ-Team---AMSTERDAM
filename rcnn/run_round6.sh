#!/bin/bash
# Round 6: NUDZ's 5 newly traced thick rats (15, 25, 35, 100, 125) join the test set (14 rats)
# and leave training (48 atlas-labeled rats). HYPO_TEST=14 sends data to rcnn_data_14/ and runs
# to rcnn_out_14/, so the 9-rat results stay untouched. Both stages of the zoomed model are
# retrained without the 5 rats (stage 1 Faster R-CNN, then the zoomed Mask R-CNN), then
# fine-tuned on the hand traces of 14 rats in 3 folds, and everything is scored on all 14.
# About 3 hours. Finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HYPO_TEST=14

python3 -m rcnn.build_dataset
for run in faster_rcnn zoom_mask_rcnn; do
  if [ -f "rcnn_out_14/$run/predictions.pkl" ]; then
    echo "skipping $run (already done)"
  elif [ "$run" = faster_rcnn ]; then
    python3 -m rcnn.train --model faster_rcnn
  else
    python3 -m rcnn.zoom
  fi
done
python3 -m rcnn.finetune_cv
python3 -m rcnn.evaluate
