#!/usr/bin/env bash
# Task 4 end to end: Optuna search (few epochs per trial) -> full retraining of the best configuration.
#
#   FS2K_ROOT=data/fs2k TRIALS=20 TRIAL_EPOCHS=15 FINAL_EPOCHS=150 bash scripts/run_t4.sh
set -euo pipefail
TAG=${TAG:-}
TRIALS=${TRIALS:-20}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-15}
FINAL_EPOCHS=${FINAL_EPOCHS:-150}
SAMPLE_EVERY=${SAMPLE_EVERY:-10}
FS2K_ROOT=${FS2K_ROOT:-data/fs2k}
export PYTHONUNBUFFERED=1
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU
T4="t4$TAG"

step() { echo; echo "=== $* ($(date +%H:%M:%S)) ==="; }
skip() { [ -f "$1" ] && { echo "skip: $1 exists"; return 0; } || return 1; }

skip "outputs/$T4/best_params.json" || { step "T4 optuna"; python -m src.optuna_studies.cgan_study --name "$T4" --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS" --data-root "$FS2K_ROOT"; }
skip "outputs/$T4/best.pt" || { step "T4 final"; python -m src.train.cgan --name "$T4" --epochs "$FINAL_EPOCHS" --sample-every "$SAMPLE_EVERY" --params "outputs/$T4/best_params.json" --data-root "$FS2K_ROOT"; }
step "done"
