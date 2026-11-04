#!/bin/bash
# Round 5: 2.5D (Dr. Ankit's suggestion). The zoomed Mask R-CNN gets the slices below and
# above as input channels (rcnn.zoom --neighbors), trained on the atlas labels like the 2D
# zoomed model, then fine-tuned on the hand traces in the same 3 folds as the 2D preview.
# Compare: zoom_mask_rcnn (2D, 0.78) vs zoom_mask_rcnn_neighbors, and finetune_cv (2D, 0.81
# vs the other labeler) vs finetune_cv_neighbors. About 2-2.5 hours. Finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

if [ -f rcnn_out/zoom_mask_rcnn_neighbors/predictions.pkl ]; then
  echo "skipping zoom_mask_rcnn_neighbors (already done)"
else
  python3 -m rcnn.zoom --neighbors
fi
python3 -m rcnn.finetune_cv --neighbors
python3 -m rcnn.evaluate
python3 make_results_table.py
