"""Optuna study for the soft mixture-of-experts (Task 3).

    python -m src.optuna_studies.moe_study --name t3 --cls ... --specs ... --n-trials 20 --epochs 8 --warmup-epochs 2

Tunes the joint fine-tuning learning rate, temperature, classification weight, balance weight and the
L1/SSIM reconstruction weighting. Trials are pruned when the validation objective is poor or when a
single branch takes over the routing (mean weight > 0.9 after warm-up).
"""
import argparse
import json
from pathlib import Path

import optuna
import torch

from src.common import tracking
from src.train.classifier import load_data
from src.train.soft_moe import load_components, train_moe


def make_objective(pets, val, gate, experts, warmup, epochs, device):
    def objective(trial):
        cfg = dict(
            lr_joint=trial.suggest_float("lr_joint", 1e-5, 5e-4, log=True),
            temperature=trial.suggest_float("temperature", 0.5, 3.0),
            gamma=trial.suggest_float("gamma", 0.01, 1.0, log=True),
            delta=trial.suggest_float("delta", 1e-3, 1.0, log=True),
            a=trial.suggest_float("a", 0.5, 0.95),
        )
        with tracking.run(f"trial-{trial.number}", params=cfg, tags={"stage": "optuna"}, nested=True):
            best, _, _, _ = train_moe(cfg, pets, val, gate, experts, warmup, epochs, device, trial=trial)
            tracking.log_metrics({"best_val_objective": best})
        return best
    return objective


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t3")
    ap.add_argument("--cls", required=True)
    ap.add_argument("--specs", nargs=3, required=True)
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--warmup-epochs", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    pets, val = load_data(dev, a.data_root)
    gate, experts, _ = load_components(a.cls, a.specs, dev)
    study = optuna.create_study(
        study_name=a.name, direction="minimize", storage=f"sqlite:///{out / 'study.db'}", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=a.warmup_epochs + 1))
    tracking.init("optuna-soft-moe")
    done = len([t for t in study.trials if t.state.is_finished()])
    with tracking.run(f"study-{a.name}", params={"epochs_per_trial": a.epochs, "warmup": a.warmup_epochs}, tags={"stage": "study"}):
        study.optimize(make_objective(pets, val, gate, experts, a.warmup_epochs, a.epochs, dev),
                       n_trials=max(a.n_trials - done, 0), timeout=a.timeout)
    (out / "best_params.json").write_text(json.dumps(study.best_params, indent=2))
    study.trials_dataframe().to_csv(out / "trials.csv", index=False)
    states = [t.state.name for t in study.trials]
    print("trials:", {s: states.count(s) for s in set(states)})
    print("best value", study.best_value, "params", study.best_params)
