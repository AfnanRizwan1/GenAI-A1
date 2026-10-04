"""Copy the small result files into the repository and write the configuration files.

* ``configs/*.json``: the hyper-parameters selected by each Optuna study together with the search budget that was used.
* ``results/``: Optuna trial tables, training histories, evaluation summaries / tables / figures and the ONNX verification.
Large files (checkpoints, ONNX models, per-entry CSVs, the Optuna SQLite databases) are not copied.

    python scripts/collect_results.py --outputs outputs --models models --out .
"""
import argparse
import json
import shutil
from pathlib import Path

CONFIGS = {  # study folder -> (config file, task, search budget actually used)
    "t1_v2": ("task1_universal_autoencoder.json", "Task 1 universal denoising autoencoder", dict(trials=30, epochs_per_trial=10, final_epochs=150)),
    "t2_cls_v2": ("task2_classifier.json", "Task 2 corruption classifier", dict(trials=30, epochs_per_trial=10, final_epochs=60)),
    "spec_shared_v2": ("task2_specialists_shared.json", "Task 2 specialist autoencoders (shared architecture search; three independent trainings)",
                       dict(trials=30, epochs_per_trial=10, final_epochs=150)),
    "t3_v2": ("task3_soft_moe.json", "Task 3 soft mixture of experts", dict(trials=20, epochs_per_trial=8, warmup_epochs_in_trials=2, final_epochs=40, final_warmup_epochs=3)),
    "t4": ("task4_cgan.json", "Task 4 style-conditioned face-to-sketch cGAN", dict(trials=20, epochs_per_trial=15, final_epochs=150)),
}
RUN_FILES = ("trials.csv", "best_params.json", "history.json")
EVAL_SKIP = {"per_entry.csv", "per_image.csv"}


def collect(outputs: Path, models: Path, out: Path):
    copied = []
    (out / "configs").mkdir(parents=True, exist_ok=True)
    for folder, (fname, task, budget) in CONFIGS.items():
        bp = outputs / folder / "best_params.json"
        if not bp.exists():
            print("missing:", bp)
            continue
        cfg = {"task": task, "selected_by": "Optuna (TPE sampler, seed 42, median pruner)", "search_budget": budget,
               "selected_hyperparameters": json.loads(bp.read_text()), "seed": 42, "source_study": folder}
        (out / "configs" / fname).write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        copied.append(out / "configs" / fname)
    for folder in sorted(p.name for p in outputs.iterdir() if p.is_dir() and p.name not in ("mlflow",) and not p.name.startswith("eval")) if outputs.exists() else []:
        for f in RUN_FILES:
            src = outputs / folder / f
            if src.exists():
                dst = out / "results" / folder / f
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dst)
                copied.append(dst)
    for ev in sorted(p for p in outputs.glob("eval*") if p.is_dir()):
        for f in sorted(ev.iterdir()):
            if f.is_file() and f.name not in EVAL_SKIP and f.suffix in {".json", ".csv", ".png"}:
                dst = out / "results" / ev.name / f.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(f, dst)
                copied.append(dst)
    ov = models / "onnx_verification.json"
    if ov.exists():
        (out / "results").mkdir(parents=True, exist_ok=True)
        shutil.copy(ov, out / "results" / "onnx_verification.json")
        copied.append(out / "results" / "onnx_verification.json")
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default=".")
    a = ap.parse_args()
    files = collect(Path(a.outputs), Path(a.models), Path(a.out))
    size = sum(f.stat().st_size for f in files) / 1e6
    print(f"copied {len(files)} files ({size:.1f} MB) into {a.out}/configs and {a.out}/results")


if __name__ == "__main__":
    main()
