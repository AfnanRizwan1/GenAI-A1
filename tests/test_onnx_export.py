import json
from argparse import Namespace

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.export import export_onnx as ex
from src.models.autoencoder import ConvAE
from src.models.cgan import UNetGenerator
from src.models.classifier import CorruptionClassifier
from src.models.soft_moe import SoftMoE


def _ae_ckpt(path, bottleneck, **kw):
    cfg = dict(base_ch=8, latent_dim=16, dropout=0.0, bottleneck=bottleneck, latent_ch=4)
    m = ConvAE(8, 16, 0.0, bottleneck=bottleneck, latent_ch=4)
    for mod in m.modules():                       # non-trivial batch-norm statistics, like a trained model
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.normal_(0, 0.1), mod.running_var.uniform_(0.5, 1.5)
    torch.save({"state_dict": m.state_dict(), "cfg": cfg}, path)


def test_export_all_models_match_pytorch(tmp_path):
    _ae_ckpt(tmp_path / "t1.pt", "conv")
    for n in ("salt", "blur", "occ"):
        _ae_ckpt(tmp_path / f"{n}.pt", "linear")
    torch.save({"state_dict": CorruptionClassifier("small", 0.0).state_dict(), "cfg": {"channels": "small", "dropout": 0.0}},
               tmp_path / "cls.pt")
    exp_cfg = dict(base_ch=8, latent_dim=16, dropout=0.0, bottleneck="conv", latent_ch=4)
    moe = SoftMoE(CorruptionClassifier("small", 0.0), [ConvAE(8, 16, 0.0, bottleneck="conv", latent_ch=4) for _ in range(3)], 1.5)
    torch.save({"state_dict": moe.state_dict(), "cfg": {"temperature": 1.5}, "gate_cfg": {"channels": "small", "dropout": 0.0},
                "expert_cfgs": [exp_cfg] * 3}, tmp_path / "moe.pt")
    torch.save({"state_dict": UNetGenerator(8, 8, 0.3).state_dict(), "cfg": {"base_ch": 8, "emb_dim": 8, "dropout": 0.3}}, tmp_path / "gen.pt")

    out = tmp_path / "models"
    report = ex.run(Namespace(t1=str(tmp_path / "t1.pt"), cls=str(tmp_path / "cls.pt"),
                              specs=[str(tmp_path / f"{n}.pt") for n in ("salt", "blur", "occ")],
                              moe=str(tmp_path / "moe.pt"), gen=str(tmp_path / "gen.pt"), out=str(out)))
    assert len(report) == 7 and all(r["passed"] and r["max_abs_diff"] < ex.TOLERANCE for r in report.values())
    assert json.loads((out / "onnx_verification.json").read_text()).keys() == report.keys()
    for f in out.glob("*.onnx"):
        onnx.checker.check_model(onnx.load(str(f)))                       # structurally valid ONNX
    # contracts the backend relies on
    s = ort.InferenceSession(str(out / "soft_moe.onnx"), providers=["CPUExecutionProvider"])
    assert [o.name for o in s.get_outputs()] == ["restored", "weights"]
    y, w = s.run(None, {"image": np.random.rand(3, 3, 128, 128).astype(np.float32)})      # unseen batch size 3
    assert y.shape == (3, 3, 128, 128) and np.allclose(w.sum(1), 1.0, atol=1e-5)
    s = ort.InferenceSession(str(out / "classifier.onnx"), providers=["CPUExecutionProvider"])
    p, = s.run(None, {"image": np.random.rand(2, 3, 128, 128).astype(np.float32)})
    assert p.shape == (2, 4) and np.allclose(p.sum(1), 1.0, atol=1e-5)
    s = ort.InferenceSession(str(out / "sketch_generator.onnx"), providers=["CPUExecutionProvider"])
    assert [i.name for i in s.get_inputs()] == ["photo", "style"]
    sk, = s.run(None, {"photo": np.random.rand(2, 3, 128, 128).astype(np.float32) * 2 - 1, "style": np.array([0, 2], dtype=np.int64)})
    assert sk.shape == (2, 3, 128, 128) and sk.min() >= -1 and sk.max() <= 1


def test_verifier_detects_a_mismatch(tmp_path):
    """If the PyTorch model no longer matches the exported file, verification must fail."""
    m = ConvAE(8, 16, 0.0, bottleneck="conv", latent_ch=4).eval()
    path = ex.export_one(m, ex.image_inputs()(1, 0), tmp_path / "m.onnx", ["image"], ["restored"])
    assert ex.verify_one(m, path, ex.image_inputs(), ["image"])["passed"]
    with torch.no_grad():
        m.decoder[-2].weight.add_(0.05)                                    # drift the PyTorch weights
    assert not ex.verify_one(m, path, ex.image_inputs(), ["image"])["passed"]
