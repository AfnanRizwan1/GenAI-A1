#!/usr/bin/env bash
# Task 3 end to end: Optuna search -> final joint training. Needs the Task 2 outputs
# (classifier + three specialists, same TAG as run_t1_t2.sh).
#
#   TAG=_v2 TRIALS=20 TRIAL_EPOCHS=8 FINAL_EPOCHS=40 bash scripts/run_t3.sh
set -euo pipefail
TAG=${TAG:-}
TRIALS=${TRIALS:-20}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-8}
TRIAL_WARMUP=${TRIAL_WARMUP:-2}
FINAL_EPOCHS=${FINAL_EPOCHS:-40}
FINAL_WARMUP=${FINAL_WARMUP:-3}
export PYTHONUNBUFFERED=1
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=${DATA_ROOT:-data}
DR="--data-root $DATA_ROOT"
T3="t3$TAG"
CLS="outputs/t2_cls$TAG/best.pt"
SPECS="outputs/spec_salt$TAG/best.pt outputs/spec_blur$TAG/best.pt outputs/spec_occlusion$TAG/best.pt"
for f in $CLS $SPECS; do [ -f "$f" ] || { echo "missing $f - run scripts/run_t1_t2.sh first (TAG=$TAG)"; exit 1; }; done

step() { echo; echo "=== $* ($(date +%H:%M:%S)) ==="; }
skip() { [ -f "$1" ] && { echo "skip: $1 exists"; return 0; } || return 1; }

skip "outputs/$T3/best_params.json" || { step "T3 optuna"; python -m src.optuna_studies.moe_study --name "$T3" --cls "$CLS" --specs $SPECS --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS" --warmup-epochs "$TRIAL_WARMUP" $DR; }
skip "outputs/$T3/best.pt" || { step "T3 final"; python -m src.train.soft_moe --name "$T3" --cls "$CLS" --specs $SPECS --warmup-epochs "$FINAL_WARMUP" --epochs "$FINAL_EPOCHS" --params "outputs/$T3/best_params.json" $DR; }
step "done"
