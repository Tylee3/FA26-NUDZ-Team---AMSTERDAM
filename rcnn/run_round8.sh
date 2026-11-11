#!/bin/bash
# Round 8, on all 14 traced rats (HYPO_TEST=14): the end-slice rescue (rcnn/rescue.py) on
# all 9 fine-tuned fold-model sets from rounds 6-7 (3 kinds x 3 seeds). No training, about
# 10-15 minutes. Then score everything, so the seed-0 rescue runs join comparison.md.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export HYPO_TEST=14

python3 -m rcnn.rescue
python3 -m rcnn.evaluate
