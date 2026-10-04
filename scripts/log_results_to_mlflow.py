"""Record the final checkpoints, evaluation results and ONNX verification in MLflow.

The training and Optuna runs already logged parameters, losses, metrics and sample images while they ran. This script adds, after
training, the pieces that only exist afterwards: the final **checkpoints**, the **test-set evaluation** (metrics, tables, example /
failure / routing figures) and the **ONNX verification**, so the whole experiment record lives in one MLflow store.

    python scripts/log_results_to_mlflow.py --outputs outputs --models models --mlflow-dir outputs/mlflow
    mlflow ui --backend-store-uri sqlite:///outputs/mlflow/mlflow.db
"""
import argparse
import json
import re
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.common import tracking  # noqa: E402

TRAINED = {  # output folder -> readable run name
    "t1_v2": "task1-universal-autoencoder", "t2_cls_v2": "task2-classifier", "spec_salt_v2": "task2-specialist-salt",
    "spec_blur_v2": "task2-specialist-blur", "spec_occlusion_v2": "task2-specialist-occlusion", "t3_v2": "task3-soft-moe",
    "t4": "task4-cgan-generator",
}
STUDIES = {"t1_v2": "task1", "t2_cls_v2": "task2-classifier", "spec_shared_v2": "task2-specialists-shared", "t3_v2": "task3", "t4": "task4"}
FIGURE_SUFFIXES = {".png", ".pdf"}
TABLE_SUFFIXES = {".csv", ".json"}


def flatten(d, prefix=""):
    """Numeric leaves of a nested dict as {name: float}. Lists (for example confusion matrices) are skipped."""
    out = {}
    for k, v in d.items():
        name = f"{prefix}{k}"
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            out[re.sub(r"[^0-9A-Za-z_\-\. /]", "_", name)[:240]] = float(v)
        elif isinstance(v, dict):
            out.update(flatten(v, name + "."))
    return out


def log_trained(outputs: Path):
    n = 0
    for folder, name in TRAINED.items():
        d = outputs / folder
        if not d.exists():
            print("skip (missing):", d)
            continue
        with mlflow.start_run(run_name=f"final-record-{name}"):
            mlflow.set_tags({"stage": "final-record", "task": name})
            bp = d / "best_params.json"
            if not bp.exists() and folder.startswith("spec_"):
                bp = outputs / "spec_shared_v2" / "best_params.json"  # specialists share the searched architecture
            if bp.exists():
                mlflow.log_params({f"best.{k}": v for k, v in json.loads(bp.read_text()).items()})
            hist = d / "history.json"
            if hist.exists():
                h = json.loads(hist.read_text())
                last = h[-1] if isinstance(h, list) and h else {}
                mlflow.log_metrics({f"final.{k}": float(v) for k, v in last.items() if isinstance(v, (int, float)) and not isinstance(v, bool)})
                mlflow.log_metric("epochs_trained", len(h))
                mlflow.log_artifact(str(hist))
            for f in ("best.pt", "trials.csv"):
                if (d / f).exists():
                    mlflow.log_artifact(str(d / f), "checkpoint" if f == "best.pt" else "optuna")
            samples = d / "samples"
            if samples.exists():
                mlflow.log_artifacts(str(samples), "generated_samples")
        n += 1
    return n


def log_studies(outputs: Path):
    n = 0
    for folder, name in STUDIES.items():
        csv = outputs / folder / "trials.csv"
        if not csv.exists():
            continue
        import pandas as pd
        df = pd.read_csv(csv)
        with mlflow.start_run(run_name=f"optuna-summary-{name}"):
            mlflow.set_tags({"stage": "final-record", "kind": "optuna-summary"})
            states = df["state"].value_counts().to_dict()
            mlflow.log_metrics({"trials_total": len(df), "trials_complete": states.get("COMPLETE", 0), "trials_pruned": states.get("PRUNED", 0)})
            done = df[df["state"] == "COMPLETE"]["value"]
            if len(done):
                mlflow.log_metrics({"best_value": float(done.max() if "cls" in folder else done.min())})
            mlflow.log_artifact(str(csv))
        n += 1
    return n


def log_eval(ev: Path, name: str):
    if not ev.exists():
        print("skip (missing):", ev)
        return 0
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({"stage": "final-record", "kind": "test-evaluation"})
        s = ev / "summary.json"
        if s.exists():
            mlflow.log_metrics(flatten(json.loads(s.read_text())))
            mlflow.log_artifact(str(s))
        for f in sorted(ev.iterdir()):
            if f.name == "per_entry.csv":  # large; the summary tables carry the information
                continue
            if f.suffix in FIGURE_SUFFIXES:
                mlflow.log_artifact(str(f), "figures")
            elif f.suffix in TABLE_SUFFIXES and f.name != "summary.json":
                mlflow.log_artifact(str(f), "tables")
    return 1


def log_onnx(models: Path):
    p = models / "onnx_verification.json"
    if not p.exists():
        print("skip (missing):", p)
        return 0
    d = json.loads(p.read_text())
    with mlflow.start_run(run_name="onnx-export-verification"):
        mlflow.set_tags({"stage": "final-record", "kind": "onnx"})
        for k, v in d.items():
            mlflow.log_metrics({f"{k}.max_abs_diff": v["max_abs_diff"], f"{k}.mean_abs_diff": v["mean_abs_diff"], f"{k}.size_mb": v["size_mb"]})
        mlflow.log_param("tolerance", next(iter(d.values()))["tolerance"])
        mlflow.log_artifact(str(p))
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--mlflow-dir", default="outputs/mlflow")
    a = ap.parse_args()
    tracking.TRACKING_DIR = a.mlflow_dir
    tracking.init("final-checkpoints-and-evaluation")
    outputs, models = Path(a.outputs), Path(a.models)
    runs = (log_trained(outputs) + log_studies(outputs) + log_eval(outputs / "eval_v2", "evaluation-tasks-1-3")
            + log_eval(outputs / "eval_t4", "evaluation-task-4") + log_onnx(models))
    print(f"logged {runs} runs to {a.mlflow_dir}")
    return runs


if __name__ == "__main__":
    main()
