#!/bin/bash
# Round 4: the three edge-focused mask heads (rcnn/edge_heads.py: edge loss, ASPP, gated edge
# branch) in a new Cascade Mask R-CNN (plain baseline + 3 variants) and in the zoomed Mask
# R-CNN (3 variants; its plain baseline already exists), then the variant of the zoomed model
# with the best median Dice vs labeler 1 goes through the hand-trace fine-tune preview.
# About 8-9 hours. Finished runs are skipped, so re-running after an interruption picks up.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

run() {
  local dir="$1"; shift
  if [ -f "rcnn_out/$dir/predictions.pkl" ]; then
    echo "skipping $dir (already done)"
    return
  fi
  python3 -m "$@"
}

run cascade_mask_rcnn rcnn.train --model cascade_mask_rcnn --head plain
run cascade_mask_rcnn_edge_loss rcnn.train --model cascade_mask_rcnn --head edge_loss
run cascade_mask_rcnn_aspp rcnn.train --model cascade_mask_rcnn --head aspp
run cascade_mask_rcnn_gated_edge rcnn.train --model cascade_mask_rcnn --head gated_edge
run zoom_mask_rcnn_edge_loss rcnn.zoom --head edge_loss
run zoom_mask_rcnn_aspp rcnn.zoom --head aspp
run zoom_mask_rcnn_gated_edge rcnn.zoom --head gated_edge

python3 -m rcnn.evaluate
python3 -m rcnn.finetune_cv --head best
python3 -m rcnn.evaluate
python3 make_results_table.py
