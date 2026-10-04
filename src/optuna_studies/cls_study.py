"""Optuna study for the corruption classifier (Task 2).

    python -m src.optuna_studies.cls_study --name t2_cls --n-trials 8 --epochs 5

Tunes learning rate, batch size, channel configuration, dropout and weight decay;
objective = validation macro-F1 (maximised).
"""
import argparse
import json
from pathlib import Path

import optuna
import torch

from src.common import tracking
from src.models.classifier import CHANNEL_CONFIGS
from src.train.classifier import load_data, train_cls


def make_objective(pets, val, epochs, device):
    def objective(trial):
        cfg = dict(
            lr=trial.suggest_float("lr", 3e-4, 5e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            channels=trial.suggest_categorical("channels", list(CHANNEL_CONFIGS)),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            weight_decay=trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True),
        )
        with tracking.run(f"trial-{trial.number}", params=cfg, tags={"stage": "optuna"}, nested=True):
            best, _, _, _ = train_cls(cfg, pets, val, epochs, device, trial=trial)
            tracking.log_metrics({"best_val_f1": best})
        return best
    return objective


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t2_cls")
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    pets, val = load_data(dev, a.data_root)
    study = optuna.create_study(
        study_name=a.name, direction="maximize", storage=f"sqlite:///{out / 'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2))
    tracking.init("optuna-classifier")
    done = len([t for t in study.trials if t.state.is_finished()])
    with tracking.run(f"study-{a.name}", params={"epochs_per_trial": a.epochs}, tags={"stage": "study"}):
        study.optimize(make_objective(pets, val, a.epochs, dev), n_trials=max(a.n_trials - done, 0), timeout=a.timeout)
    (out / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    states = [t.state.name for t in study.trials]
    print("trials:", {s: states.count(s) for s in set(states)})
    print("best value", study.best_value, "params", study.best_params)
