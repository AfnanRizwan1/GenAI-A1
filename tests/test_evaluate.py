import json
import sys

import numpy as np
import torch

from src.eval import evaluate as ev
from src.models.autoencoder import ConvAE
from src.models.classifier import CorruptionClassifier
from tests.test_models_training import fake_images


def _save_ae(path, bottleneck):
    cfg = dict(base_ch=8, latent_dim=32, dropout=0.0, bottleneck=bottleneck, latent_ch=4)
    m = ConvAE(cfg["base_ch"], cfg["latent_dim"], 0.0, bottleneck=bottleneck, latent_ch=4)
    torch.save({"state_dict": m.state_dict(), "cfg": cfg}, path)


def test_full_evaluation_runs_and_is_consistent(tmp_path, monkeypatch):
    root = tmp_path / "data" / "cache"
    root.mkdir(parents=True)
    np.save(root / "pets_trainval_128.npy", fake_images(40))
    np.save(root / "pets_test_128.npy", fake_images(40, seed=3))
    _save_ae(tmp_path / "t1.pt", "conv")
    for n in ("salt", "blur", "occ"):
        _save_ae(tmp_path / f"{n}.pt", "linear")
    cls = CorruptionClassifier("small", 0.0)
    torch.save({"state_dict": cls.state_dict(), "cfg": {"channels": "small", "dropout": 0.0}}, tmp_path / "cls.pt")

    out = tmp_path / "eval"
    monkeypatch.chdir(__import__("pathlib").Path(__file__).resolve().parents[1])  # manifests/ live in the repo root
    monkeypatch.setattr(sys, "argv", [
        "evaluate", "--t1", str(tmp_path / "t1.pt"), "--cls", str(tmp_path / "cls.pt"),
        "--specs", str(tmp_path / "salt.pt"), str(tmp_path / "blur.pt"), str(tmp_path / "occ.pt"),
        "--out", str(out), "--data-root", str(tmp_path / "data"), "--bs", "64", "--max-entries", "200"])
    ev.main()

    for f in ("summary.json", "per_entry.csv", "by_condition.csv", "confusion_matrix.png", "examples_t1.png",
              "examples_t2_oracle.png", "examples_t2_pred.png", "failures_t1.png"):
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text())
    assert s["n_entries"] == 200 and set(s["methods"]) == {"input", "t1", "t2_oracle", "t2_pred"}
    # 9 corruption/severity cells + clean
    assert len(s["by_condition"]) == 10
    # identity baseline on clean images is perfect (capped), SSIM exactly 1
    assert s["by_condition"]["clean/-"]["input_psnr"] == ev.PSNR_CAP
    assert abs(s["by_condition"]["clean/-"]["input_ssim"] - 1.0) < 1e-4
    # oracle routing bypasses experts on clean images -> identical to the input there
    assert s["by_condition"]["clean/-"]["t2_oracle_psnr"] == ev.PSNR_CAP
    # classifier report is a proper 4x4 row-stochastic matrix
    cm = np.array(s["classifier"]["confusion_normalized"])
    assert cm.shape == (4, 4) and np.allclose(cm.sum(1), 1.0)
    assert 0.0 <= s["classifier"]["accuracy"] <= 1.0
    assert sum(v["support"] for v in s["classifier"]["per_class"].values()) == 200


def test_routing_uses_identity_for_clean_and_experts_otherwise():
    calls = []

    def expert(i):
        def f(x):
            calls.append(i)
            return torch.zeros_like(x)
        return f

    sysm = ev.Systems(specs=[expert(1), expert(2), expert(3)])
    x = torch.ones(4, 3, 8, 8)
    y = sysm.route(x, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(y[0], x[0]) and (y[1:] == 0).all() and sorted(calls) == [1, 2, 3]
