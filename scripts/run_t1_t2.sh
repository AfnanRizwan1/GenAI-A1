#!/usr/bin/env bash
# Tasks 1 and 2 end to end: Optuna searches -> final training. Resumable (Optuna studies use SQLite,
# finished stages are skipped when their output exists).
#
#   TRIALS=8 TRIAL_EPOCHS=5 FINAL_EPOCHS=40 bash scripts/run_t1_t2.sh
set -euo pipefail
TRIALS=${TRIALS:-8}
TRIAL_EPOCHS=${TRIAL_EPOCHS:-5}
FINAL_EPOCHS=${FINAL_EPOCHS:-40}
CLS_FINAL_EPOCHS=${CLS_FINAL_EPOCHS:-30}
export PYTHONUNBUFFERED=1

step() { echo; echo "=== $* ($(date +%H:%M:%S)) ==="; }
skip() { [ -f "$1" ] && { echo "skip: $1 exists"; return 0; } || return 1; }

# Task 1: universal autoencoder
skip outputs/t1/best_params.json || { step "T1 optuna"; python -m src.optuna_studies.ae_study --name t1 --cond all --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS"; }
skip outputs/t1/best.pt || { step "T1 final"; python -m src.train.ae --name t1 --cond all --epochs "$FINAL_EPOCHS" --params outputs/t1/best_params.json; }

# Task 2: classifier
skip outputs/t2_cls/best_params.json || { step "T2 classifier optuna"; python -m src.optuna_studies.cls_study --name t2_cls --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS"; }
skip outputs/t2_cls/best.pt || { step "T2 classifier final"; python -m src.train.classifier --name t2_cls --epochs "$CLS_FINAL_EPOCHS" --params outputs/t2_cls/best_params.json; }

# Task 2: specialists - one shared search for the architecture, three independent trainings
skip outputs/spec_shared/best_params.json || { step "T2 specialist shared optuna"; python -m src.optuna_studies.ae_study --name spec_shared --cond corrupted --n-trials "$TRIALS" --epochs "$TRIAL_EPOCHS"; }
for c in salt blur occlusion; do
  skip "outputs/spec_$c/best.pt" || { step "T2 specialist $c"; python -m src.train.ae --name "spec_$c" --cond "$c" --epochs "$FINAL_EPOCHS" --params outputs/spec_shared/best_params.json; }
done
step "done"
