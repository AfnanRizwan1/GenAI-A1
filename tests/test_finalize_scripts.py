import importlib.util
import json
import os
from pathlib import Path

import mlflow
import numpy as np
from mlflow.tracking import MlflowClient
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fake_outputs(root: Path):
    out = root / "outputs"
    for folder in ("t1_v2", "t2_cls_v2", "spec_shared_v2", "spec_salt_v2", "t3_v2", "t4"):
        d = out / folder
        d.mkdir(parents=True)
        (d / "best.pt").write_bytes(b"checkpoint-bytes")
        (d / "history.json").write_text(json.dumps([{"epoch": 0, "val_psnr": 20.0, "train_loss": 0.3}, {"epoch": 1, "val_psnr": 22.5, "train_loss": 0.2}]))
        if not folder.startswith("spec_salt"):
            (d / "best_params.json").write_text(json.dumps({"lr": 0.001, "batch_size": 32, "bottleneck": "conv"}))
            (d / "trials.csv").write_text("number,value,state\n0,0.5,COMPLETE\n1,0.7,PRUNED\n")
        (d / "study.db").write_bytes(b"sqlite")
    (out / "t4" / "samples").mkdir()
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(out / "t4" / "samples" / "epoch_000.png")
    ev = out / "eval_v2"
    ev.mkdir()
    (ev / "summary.json").write_text(json.dumps({"n_entries": 100, "overall": {"input_psnr": 15.5, "t1_psnr": 21.0},
                                                 "classifier": {"accuracy": 0.99, "per_class": {"clean": {"f1": 0.98}}, "confusion_normalized": [[1, 0], [0, 1]]}}))
    (ev / "by_condition.csv").write_text("type,level,t1_psnr\nclean,-,30\n")
    (ev / "per_entry.csv").write_text("big\n")
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(ev / "confusion_matrix.png")
    ev4 = out / "eval_t4"
    ev4.mkdir()
    (ev4 / "summary.json").write_text(json.dumps({"n_test_images": 10, "overall": {"gen_l1": 0.09}}))
    models = root / "models"
    models.mkdir()
    (models / "onnx_verification.json").write_text(json.dumps({"task1_universal": {"file": "universal_restoration.onnx", "size_mb": 40.0,
                                                                                  "max_abs_diff": 1e-6, "mean_abs_diff": 1e-8, "tolerance": 1e-4, "passed": True}}))
    return out, models


def test_collect_results_writes_configs_and_copies_only_small_files(tmp_path):
    cr = load("collect_results")
    outputs, models = fake_outputs(tmp_path)
    dest = tmp_path / "repo"
    files = cr.collect(outputs, models, dest)
    cfg = json.loads((dest / "configs" / "task1_universal_autoencoder.json").read_text())
    assert cfg["selected_hyperparameters"] == {"lr": 0.001, "batch_size": 32, "bottleneck": "conv"}
    assert cfg["search_budget"]["trials"] == 30 and cfg["seed"] == 42
    assert (dest / "configs" / "task4_cgan.json").exists() and (dest / "configs" / "task3_soft_moe.json").exists()
    names = {f.name for f in files}
    assert {"trials.csv", "history.json", "summary.json", "by_condition.csv", "confusion_matrix.png", "onnx_verification.json"} <= names
    assert not {"best.pt", "study.db", "per_entry.csv"} & names               # large / binary files stay out of the repository
    assert (dest / "results" / "eval_v2" / "summary.json").exists() and (dest / "results" / "t1_v2" / "trials.csv").exists()


def test_log_results_to_mlflow_records_checkpoints_evaluation_and_onnx(tmp_path, monkeypatch):
    lm = load("log_results_to_mlflow")
    outputs, models = fake_outputs(tmp_path)
    mdir = tmp_path / "mlflow"
    monkeypatch.setattr("sys.argv", ["x", "--outputs", str(outputs), "--models", str(models), "--mlflow-dir", str(mdir)])
    n = lm.main()
    assert n >= 10
    client = MlflowClient(tracking_uri=f"sqlite:///{os.path.abspath(mdir)}/mlflow.db")
    exp = client.get_experiment_by_name("final-checkpoints-and-evaluation")
    runs = {r.info.run_name: r for r in client.search_runs([exp.experiment_id])}

    t1 = runs["final-record-task1-universal-autoencoder"]
    assert t1.data.params["best.lr"] == "0.001" and t1.data.metrics["final.val_psnr"] == 22.5 and t1.data.metrics["epochs_trained"] == 2
    assert "best.pt" in [a.path.split("/")[-1] for a in client.list_artifacts(t1.info.run_id, "checkpoint")]
    salt = runs["final-record-task2-specialist-salt"]              # specialists take the shared search result as their parameters
    assert salt.data.params["best.bottleneck"] == "conv"
    gen = runs["final-record-task4-cgan-generator"]
    assert [a.path for a in client.list_artifacts(gen.info.run_id, "generated_samples")] == ["generated_samples/epoch_000.png"]

    ev = runs["evaluation-tasks-1-3"]
    assert ev.data.metrics["classifier.accuracy"] == 0.99 and ev.data.metrics["overall.t1_psnr"] == 21.0
    assert not any("confusion" in k for k in ev.data.metrics)       # lists are skipped, only numeric leaves become metrics
    figs = [a.path for a in client.list_artifacts(ev.info.run_id, "figures")]
    assert figs == ["figures/confusion_matrix.png"]
    tables = [a.path for a in client.list_artifacts(ev.info.run_id, "tables")]
    assert "tables/by_condition.csv" in tables and not any("per_entry" in t for t in tables)
    assert runs["evaluation-task-4"].data.metrics["overall.gen_l1"] == 0.09
    onnx = runs["onnx-export-verification"]
    assert onnx.data.metrics["task1_universal.max_abs_diff"] == 1e-6 and onnx.data.params["tolerance"] == "0.0001"
    summ = runs["optuna-summary-task1"]
    assert summ.data.metrics["trials_total"] == 2 and summ.data.metrics["trials_pruned"] == 1 and summ.data.metrics["best_value"] == 0.5


def test_flatten_keeps_numbers_only():
    lm = load("log_results_to_mlflow")
    f = lm.flatten({"a": 1, "b": {"c": 2.5, "d": [1, 2], "e": True, "f": "x"}})
    assert f == {"a": 1.0, "b.c": 2.5}
