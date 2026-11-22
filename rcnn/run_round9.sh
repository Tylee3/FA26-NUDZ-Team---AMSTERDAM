#!/bin/bash
# Round 9, overnight (~7.5 h), all on the 14 thick test rats (HYPO_TEST=14) plus the 5 new
# high-resolution rats. Each step is logged with its exit code in rcnn_out_14/round9_status.log;
# a failed step does not stop the ones after it. Finished folds are skipped on a restart.
#   1. MedSAM2 as released on the 14 rats: 2 prompts x 2 views (~15 min)
#   2. retrain the expert-2 and both-experts folds that learned from the duplicate rat 4
#      trace (folds 1 and 2, seeds 0-2, and warp), then redo mirror / average of 3 (~2.5 h).
#      The old fold models are moved, not deleted, to rcnn_out_14/_trained_on_duplicate_rat4/
#   3. 2.5D on 14 rats: atlas-trained base, then fine-tuned on expert 1 / expert 2 (~2 h)
#   4. every R-CNN model, 2D and 2.5D, on the high-resolution rats (~20 min)
#   5. MedSAM2 on the high-resolution rats (~20 min)
#   6. score everything
#   7. 2.5D seed repeats 1 and 2, for the noise check (~2 h), and score again
set -uo pipefail
cd "$(dirname "$0")/.."
export HYPO_TEST=14 PYTORCH_ENABLE_MPS_FALLBACK=1
PY=.venv/bin/python
MS=.venv-medsam2/bin/python
LOG=rcnn_out_14/round9_status.log
OLD=rcnn_out_14/_trained_on_duplicate_rat4

step() {
  local name=$1; shift
  echo "[$(date '+%a %H:%M')] start $name" | tee -a "$LOG"
  "$@"
  local rc=$?
  echo "[$(date '+%a %H:%M')] end   $name EXIT=$rc" | tee -a "$LOG"
}

move_duplicate_folds() {
  if [ -f "$OLD/MOVED" ]; then
    echo "duplicate-trace folds already moved to $OLD"
    return 0
  fi
  for run in finetune_cv finetune_cv_seed1 finetune_cv_seed2 finetune_cv_warp; do
    for member in labeler2 both; do
      for fold in fold1 fold2; do
        if [ -d "rcnn_out_14/$run/$member/$fold" ]; then
          mkdir -p "$OLD/$run/$member"
          mv "rcnn_out_14/$run/$member/$fold" "$OLD/$run/$member/$fold" || return 1
        fi
      done
    done
  done
  touch "$OLD/MOVED"
}

base_25d() {
  if [ -f rcnn_out_14/zoom_mask_rcnn_neighbors/predictions.pkl ]; then
    echo "zoom_mask_rcnn_neighbors already done"
  else
    $PY -m rcnn.zoom --neighbors
  fi
}

echo "=== round 9 started $(date) ===" >> "$LOG"
step "1 medsam2 14 rats" $MS -m rcnn.medsam2 --data thick
step "2a move duplicate folds" move_duplicate_folds
step "2b retrain expert 2 seed 0" $PY -m rcnn.finetune_cv
step "2c retrain both seed 0" $PY -m rcnn.finetune_cv --both
step "2d retrain expert 2 seed 1" $PY -m rcnn.finetune_cv --seed 1
step "2e retrain both seed 1" $PY -m rcnn.finetune_cv --seed 1 --both
step "2f retrain expert 2 seed 2" $PY -m rcnn.finetune_cv --seed 2
step "2g retrain both seed 2" $PY -m rcnn.finetune_cv --seed 2 --both
step "2h retrain warp expert 2" $PY -m rcnn.finetune_cv --warp
step "2i mirror + average of 3" $PY -m rcnn.finetune_combine
step "3a 2.5D atlas-trained" base_25d
step "3b 2.5D fine-tuned" $PY -m rcnn.finetune_cv --neighbors
step "4 R-CNN on high-res" $PY -m rcnn.highres
step "5 medsam2 high-res" $MS -m rcnn.medsam2 --data highres
step "6 score" $PY -m rcnn.evaluate
step "7a 2.5D seed 1" $PY -m rcnn.finetune_cv --neighbors --seed 1
step "7b 2.5D seed 2" $PY -m rcnn.finetune_cv --neighbors --seed 2
step "7c score again" $PY -m rcnn.evaluate
echo "=== round 9 finished $(date) ===" | tee -a "$LOG"
grep "EXIT=" "$LOG" | tail -18
