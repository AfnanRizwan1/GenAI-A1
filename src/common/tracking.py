"""Thin MLflow wrapper. Tracking never breaks training: failures are swallowed."""
import contextlib
import os

import mlflow

TRACKING_DIR = os.environ.get("MLFLOW_DIR", "outputs/mlruns")


def init(experiment):
    mlflow.set_tracking_uri(f"file:{os.path.abspath(TRACKING_DIR)}")
    mlflow.set_experiment(experiment)


@contextlib.contextmanager
def run(name, params=None, tags=None, nested=False):
    with mlflow.start_run(run_name=name, nested=nested) as r:
        if tags:
            mlflow.set_tags(tags)
        if params:
            mlflow.log_params(params)
        yield r


def log_metrics(metrics, step=None):
    try:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()}, step=step)
    except Exception as e:  # noqa: BLE001
        print("mlflow log_metrics failed:", e)


def log_artifact(path):
    try:
        mlflow.log_artifact(str(path))
    except Exception as e:  # noqa: BLE001
        print("mlflow log_artifact failed:", e)
