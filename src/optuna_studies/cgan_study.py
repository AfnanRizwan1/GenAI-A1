"""Optuna study for the face-to-sketch cGAN (Task 4).

    python -m src.optuna_studies.cgan_study --name t4 --n-trials 20 --epochs 15

Tunes generator and discriminator learning rates, batch size, base channels, dropout, style-embedding
dimension and the reconstruction weight lambda. Trials use few epochs; the best configuration is then
retrained for the full schedule (src.train.cgan). Objective: validation L1 + (1 - SSIM) of the generator.
"""
import argparse
import json
from pathlib import Path

import optuna
import torch

from src.common import tracking
from src.train.cgan import load_data, train_cgan


def make_objective(data, epochs, device):
    def objective(trial):
        cfg = dict(
            lr_g=trial.suggest_float("lr_g", 5e-5, 1e-3, log=True),
            lr_d=trial.suggest_float("lr_d", 5e-5, 1e-3, log=True),
            batch_size=trial.suggest_categorical("batch_size", [8, 16, 32]),
            base_ch=trial.suggest_categorical("base_ch", [16, 32, 48, 64]),
            dropout=trial.suggest_float("dropout", 0.0, 0.5),
            emb_dim=trial.suggest_categorical("emb_dim", [8, 16, 32, 64]),
            lam=trial.suggest_float("lam", 10.0, 200.0, log=True),
        )
        with tracking.run(f"trial-{trial.number}", params=cfg, tags={"stage": "optuna"}, nested=True):
            best, _, _, _ = train_cgan(cfg, data, epochs, device, trial=trial)
            tracking.log_metrics({"best_val_objective": best})
        return best
    return objective


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t4")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--data-root", default="data/fs2k")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    data = load_data(a.data_root)
    study = optuna.create_study(
        study_name=a.name, direction="minimize", storage=f"sqlite:///{out / 'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=4))
    tracking.init("optuna-cgan")
    done = len([t for t in study.trials if t.state.is_finished()])
    with tracking.run(f"study-{a.name}", params={"epochs_per_trial": a.epochs}, tags={"stage": "study"}):
        study.optimize(make_objective(data, a.epochs, dev), n_trials=max(a.n_trials - done, 0), timeout=a.timeout)
    (out / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    states = [t.state.name for t in study.trials]
    print("trials:", {s: states.count(s) for s in set(states)})
    print("best value", study.best_value, "params", study.best_params)
