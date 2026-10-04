import optuna
import pytest
import torch
import torch.nn as nn

from src.models.autoencoder import ConvAE
from src.models.classifier import CorruptionClassifier
from src.models.loading import load_soft_moe
from src.models.soft_moe import SoftMoE, SoftMoEExport
from src.train.soft_moe import DEFAULTS, evaluate, moe_loss, train_moe
from tests.test_models_training import fake_images, fake_val


def tiny_experts():
    return [ConvAE(8, 16, 0.0, bottleneck="conv", latent_ch=4) for _ in range(3)]


class FixedGate(nn.Module):
    """Gate that ignores the image and returns fixed logits (for deterministic routing tests)."""

    def __init__(self, logits):
        super().__init__()
        self.register_buffer("l", torch.tensor(logits, dtype=torch.float32))
        self.dummy = nn.Parameter(torch.zeros(1))  # keeps the gate in the autograd graph / optimiser

    def forward(self, x):
        return self.l.expand(len(x), -1) + self.dummy * 0


def test_weights_sum_to_one_and_temperature_sharpens():
    x = torch.rand(3, 3, 128, 128)
    gate = CorruptionClassifier("small", 0.0)
    soft, sharp = SoftMoE(gate, tiny_experts(), 5.0).eval(), SoftMoE(gate, tiny_experts(), 0.1).eval()
    for m in (soft, sharp):
        y, w, logits = m(x)
        assert y.shape == x.shape and w.shape == (3, 4) and torch.allclose(w.sum(1), torch.ones(3), atol=1e-5)
    assert sharp(x)[1].max(1).values.mean() > soft(x)[1].max(1).values.mean()


def test_output_is_weighted_sum_of_branches():
    x = torch.rand(2, 3, 128, 128)
    experts = tiny_experts()
    m = SoftMoE(FixedGate([0.0, 1.0, -1.0, 2.0]), experts, 1.0).eval()
    y, w, _ = m(x)
    ref = w[:, 0].view(-1, 1, 1, 1) * x + sum(w[:, i + 1].view(-1, 1, 1, 1) * e.eval()(x) for i, e in enumerate(experts))
    assert torch.allclose(y, ref, atol=1e-5)
    # a near one-hot gate selects the identity branch (clean -> bypass)
    m2 = SoftMoE(FixedGate([30.0, 0.0, 0.0, 0.0]), experts, 1.0).eval()
    assert torch.allclose(m2(x)[0], x, atol=1e-4)


def test_balance_loss_zero_when_uniform_positive_when_collapsed():
    cfg = {**DEFAULTS}
    y = clean = torch.rand(8, 3, 128, 128)
    labels = torch.arange(8) % 4
    uni = torch.full((8, 4), 0.25)
    col = torch.tensor([[1.0, 0, 0, 0]]).repeat(8, 1)
    logits = torch.zeros(8, 4)
    _, p_uni = moe_loss(y, clean, logits, uni, labels, cfg)
    _, p_col = moe_loss(y, clean, logits, col, labels, cfg)
    assert p_uni["balance"] < 1e-9 and abs(p_col["balance"] - (0.75 ** 2 + 3 * 0.25 ** 2)) < 1e-6
    assert abs(p_uni["ce"] - torch.log(torch.tensor(4.0)).item()) < 1e-5


def test_warmup_freezes_experts_and_trains_gate():
    pets, val = {"train": fake_images(32)}, fake_val(4)
    gate, experts = CorruptionClassifier("small", 0.0), tiny_experts()
    init = {k: v.clone() for k, v in SoftMoE(gate, experts).state_dict().items()}
    # lr_joint=0: the joint phase cannot move any parameter, so any change in experts would come from warm-up
    best, state, hist, _ = train_moe(dict(lr_joint=0.0, batch_size=16), pets, val, gate, experts,
                                     warmup_epochs=2, epochs=3, device="cpu", log=False)
    assert [h["phase"] for h in hist] == [0, 0, 1]
    trainable = lambda k: "running" not in k and "num_batches" not in k
    for k, v in state.items():
        if trainable(k) and k.startswith("experts."):
            assert torch.equal(v, init[k]), f"expert parameter {k} changed during warm-up"
    assert any(not torch.equal(v, init[k]) for k, v in state.items() if k.startswith("gate.") and trainable(k))


def test_joint_training_reduces_loss():
    pets, val = {"train": fake_images(64)}, fake_val(4)
    best, state, hist, _ = train_moe(dict(batch_size=16, lr_joint=1e-3, lr_warm=2e-3), pets, val,
                                     CorruptionClassifier("small", 0.0), tiny_experts(),
                                     warmup_epochs=1, epochs=4, device="cpu", log=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert {"gate_acc", "max_mean_weight", "val_psnr", "w_clean_to_identity"} <= set(hist[-1])


def test_collapse_prunes_trial():
    class Stub:
        def report(self, *a): pass
        def should_prune(self): return False

    pets, val = {"train": fake_images(32)}, fake_val(4)
    with pytest.raises(optuna.TrialPruned):
        train_moe(dict(batch_size=16, lr_joint=0.0, lr_warm=0.0), pets, val, FixedGate([30.0, 0, 0, 0]), tiny_experts(),
                  warmup_epochs=1, epochs=2, device="cpu", trial=Stub(), log=False)


def test_checkpoint_roundtrip_and_export_wrapper(tmp_path):
    gate_cfg = {"channels": "small", "dropout": 0.0}
    exp_cfg = dict(base_ch=8, latent_dim=16, dropout=0.0, bottleneck="conv", latent_ch=4)
    m = SoftMoE(CorruptionClassifier("small", 0.0), tiny_experts(), 1.5).eval()
    torch.save({"state_dict": m.state_dict(), "cfg": {"temperature": 1.5}, "gate_cfg": gate_cfg,
                "expert_cfgs": [exp_cfg] * 3}, tmp_path / "moe.pt")
    m2 = load_soft_moe(tmp_path / "moe.pt")
    x = torch.rand(2, 3, 128, 128)
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-5)
    y, w = SoftMoEExport(m2)(x)
    assert y.shape == x.shape and w.shape == (2, 4)


def test_evaluation_with_moe_writes_routing_analysis(tmp_path, monkeypatch):
    import json, pathlib, sys
    import numpy as np
    import pandas as pd
    from src.eval import evaluate as ev

    cache = tmp_path / "data" / "cache"
    cache.mkdir(parents=True)
    np.save(cache / "pets_trainval_128.npy", fake_images(40))
    np.save(cache / "pets_test_128.npy", fake_images(40, seed=3))
    exp_cfg = dict(base_ch=8, latent_dim=16, dropout=0.0, bottleneck="conv", latent_ch=4)
    m = SoftMoE(CorruptionClassifier("small", 0.0), tiny_experts(), 1.0)
    torch.save({"state_dict": m.state_dict(), "cfg": {"temperature": 1.0},
                "gate_cfg": {"channels": "small", "dropout": 0.0}, "expert_cfgs": [exp_cfg] * 3}, tmp_path / "moe.pt")
    out = tmp_path / "eval"
    monkeypatch.chdir(pathlib.Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["evaluate", "--moe", str(tmp_path / "moe.pt"), "--out", str(out),
                                      "--data-root", str(tmp_path / "data"), "--bs", "64", "--max-entries", "200"])
    ev.main()
    for f in ("moe_routing_heatmap.png", "moe_examples_dominant.png", "moe_examples_distributed.png",
              "moe_weights_by_condition.csv", "summary.json"):
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text())
    assert set(s["methods"]) == {"input", "moe"} and len(s["moe"]["weights_by_condition"]) == 10
    for row in s["moe"]["weights_by_condition"].values():
        assert abs(sum(row.values()) - 1.0) < 1e-4
    assert set(s["moe"]["activity"]) == {"identity", "salt", "blur", "occlusion"}
    df = pd.read_csv(out / "per_entry.csv")
    assert np.allclose(df[["w_identity", "w_salt", "w_blur", "w_occlusion"]].sum(1), 1.0, atol=1e-4)
