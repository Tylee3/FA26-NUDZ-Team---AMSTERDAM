#!/bin/bash
# Round 2: Karina's box-then-mask idea (zoomed Mask R-CNN) and a 3D U-Net (plain and with
# augmentation), then rescoring everything and rebuilding RESULTS.md. About 3.5 hours.
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
  python3 -m "$@"
}

run zoom_mask_rcnn rcnn.zoom
run unet3d rcnn.unet3d
run unet3d_augment rcnn.unet3d --augment
run zoom_mask_rcnn_augment rcnn.zoom --augment

python3 -m rcnn.evaluate
python3 make_results_table.py
