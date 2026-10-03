"""Optuna study for the autoencoders (Task 1 universal model; Task 2 shared specialist architecture).

    python -m src.optuna_studies.ae_study --name t1 --cond all --n-trials 8 --epochs 5
    python -m src.optuna_studies.ae_study --name spec_shared --cond corrupted --n-trials 8 --epochs 5

Search space: lr, batch size, bottleneck type (+ its size), base encoder channels, dropout, alpha (L1 vs SSIM weight).
Resumable: storage is a SQLite file in outputs/<name>/study.db.
"""
import argparse
import json
from pathlib import Path

import optuna
import torch

from src.common import tracking
from src.train.ae import COND_IDS, load_data, train_ae


BOTTLENECKS = ["linear", "conv"]


def make_objective(pets, val, cond, epochs, device):
    def objective(trial):
        cfg = dict(
            lr=trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [32, 64, 128]),
            base_ch=trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
            dropout=trial.suggest_float("dropout", 0.0, 0.3),
            alpha=trial.suggest_float("alpha", 0.5, 0.95),
        )
        # bottleneck type decides which size parameter is searched (conditional search space)
        cfg["bottleneck"] = trial.suggest_categorical("bottleneck", BOTTLENECKS)
        if cfg["bottleneck"] == "linear":
            cfg["latent_dim"] = trial.suggest_categorical("latent_dim", [128, 256, 512, 1024])
        else:
            cfg["latent_ch"] = trial.suggest_categorical("latent_ch", [4, 8, 16, 32])
        with tracking.run(f"trial-{trial.number}", params=cfg, tags={"stage": "optuna", "cond": cond}, nested=True):
            best, _, _, _ = train_ae(cfg, pets, val, cond, epochs, device, trial=trial)
            tracking.log_metrics({"best_val_objective": best})
        return best
    return objective


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cond", default="all", choices=list(COND_IDS))
    ap.add_argument("--n-trials", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=None, help="seconds")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    pets, val = load_data(a.cond, dev, a.data_root)
    study = optuna.create_study(
        study_name=a.name, direction="minimize", storage=f"sqlite:///{out / 'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=2))
    tracking.init("optuna-autoencoders")
    done = len([t for t in study.trials if t.state.is_finished()])
    with tracking.run(f"study-{a.name}", params={"cond": a.cond, "epochs_per_trial": a.epochs}, tags={"stage": "study"}):
        study.optimize(make_objective(pets, val, a.cond, a.epochs, dev), n_trials=max(a.n_trials - done, 0),
                       timeout=a.timeout)
    (out / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    states = [t.state.name for t in study.trials]
    print("trials:", {s: states.count(s) for s in set(states)})
    print("best value", study.best_value, "params", study.best_params)
