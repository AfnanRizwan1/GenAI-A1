#!/usr/bin/env bash
# Tasks 1 and 2 end to end: Optuna searches -> final training. Resumable (Optuna studies use SQLite,
# finished stages are skipped when their output exists).
#
#   TRIALS=8 TRIAL_EPOCHS=5 FINAL_EPOCHS=40 bash scripts/run_t1_t2.sh
#   TAG=_v2 TRIALS=30 TRIAL_EPOCHS=10 FINAL_EPOCHS=150 bash scripts/run_t1_t2.sh   # separate output folders
#   GPU=1 ...                                                                       # pick a GPU
set -euo pipefail
TRIALS=${TRIALS:-8}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-5}
FINAL_EPOCHS=${FINAL_EPOCHS:-40}
CLS_FINAL_EPOCHS=${CLS_FINAL_EPOCHS:-30}
TAG=${TAG:-}
export PYTHONUNBUFFERED=1
[ -n "${GPU:-}" ] && export CUDA_VISIBLE_DEVICES=$GPU
DATA_ROOT=${DATA_ROOT:-data}
DR="--data-root $DATA_ROOT"
T1="t1$TAG"; CLS="t2_cls$TAG"; SHARED="spec_shared$TAG"

step() { echo; echo "=== $* ($(date +%H:%M:%S)) ==="; }
skip() { [ -f "$1" ] && { echo "skip: $1 exists"; return 0; } || return 1; }

# Task 1: universal autoencoder
skip "outputs/$T1/best_params.json" || { step "T1 optuna"; python -m src.optuna_studies.ae_study --name "$T1" --cond all --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS" $DR; }
skip "outputs/$T1/best.pt" || { step "T1 final"; python -m src.train.ae --name "$T1" --cond all --epochs "$FINAL_EPOCHS" --params "outputs/$T1/best_params.json" $DR; }

# Task 2: classifier
skip "outputs/$CLS/best_params.json" || { step "T2 classifier optuna"; python -m src.optuna_studies.cls_study --name "$CLS" --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS" $DR; }
skip "outputs/$CLS/best.pt" || { step "T2 classifier final"; python -m src.train.classifier --name "$CLS" --epochs "$CLS_FINAL_EPOCHS" --params "outputs/$CLS/best_params.json" $DR; }

# Task 2: specialists - one shared search for the architecture, three independent trainings
skip "outputs/$SHARED/best_params.json" || { step "T2 specialist shared optuna"; python -m src.optuna_studies.ae_study --name "$SHARED" --cond corrupted --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS" $DR; }
for c in salt blur occlusion; do
  skip "outputs/spec_$c$TAG/best.pt" || { step "T2 specialist $c"; python -m src.train.ae --name "spec_$c$TAG" --cond "$c" --epochs "$FINAL_EPOCHS" --params "outputs/$SHARED/best_params.json" $DR; }
done
step "done"
