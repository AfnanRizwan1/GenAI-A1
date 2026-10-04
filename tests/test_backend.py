import base64
import io
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
pytest.importorskip("multipart")

BACKEND = Path(__file__).resolve().parents[1] / "app" / "backend"
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient  # noqa: E402

from app import config, imaging  # noqa: E402
from app.main import create_app  # noqa: E402
from src.export import export_onnx as ex  # noqa: E402
from src.models.autoencoder import ConvAE  # noqa: E402
from src.models.cgan import UNetGenerator  # noqa: E402
from src.models.classifier import CorruptionClassifier  # noqa: E402
from src.models.soft_moe import SoftMoE  # noqa: E402
from tests.test_onnx_export import _ae_ckpt  # noqa: E402


def png_bytes(w=160, h=120, seed=0, fmt="PNG"):
    r = np.random.default_rng(seed)
    arr = (np.kron(r.random((6, 8, 3)), np.ones((h // 6 + 1, w // 8 + 1, 1)))[:h, :w] * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt)
    return buf.getvalue()


def decode_url(url):
    assert url.startswith("data:image/png;base64,")
    return Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))


@pytest.fixture(scope="module")
def models_dir(tmp_path_factory):
    """Real ONNX files exported from small untrained checkpoints (same architectures as the trained models)."""
    t = tmp_path_factory.mktemp("ckpt")
    _ae_ckpt(t / "t1.pt", "conv")
    for n in ("salt", "blur", "occ"):
        _ae_ckpt(t / f"{n}.pt", "linear")
    torch.save({"state_dict": CorruptionClassifier("small", 0.0).state_dict(), "cfg": {"channels": "small", "dropout": 0.0}}, t / "cls.pt")
    exp_cfg = dict(base_ch=8, latent_dim=16, dropout=0.0, bottleneck="conv", latent_ch=4)
    moe = SoftMoE(CorruptionClassifier("small", 0.0), [ConvAE(8, 16, 0.0, bottleneck="conv", latent_ch=4) for _ in range(3)], 1.0)
    torch.save({"state_dict": moe.state_dict(), "cfg": {"temperature": 1.0}, "gate_cfg": {"channels": "small", "dropout": 0.0},
                "expert_cfgs": [exp_cfg] * 3}, t / "moe.pt")
    torch.save({"state_dict": UNetGenerator(8, 8, 0.0).state_dict(), "cfg": {"base_ch": 8, "emb_dim": 8, "dropout": 0.0}}, t / "gen.pt")
    out = tmp_path_factory.mktemp("models")
    ex.run(Namespace(t1=str(t / "t1.pt"), cls=str(t / "cls.pt"), specs=[str(t / f"{n}.pt") for n in ("salt", "blur", "occ")],
                     moe=str(t / "moe.pt"), gen=str(t / "gen.pt"), out=str(out)))
    return out


@pytest.fixture()
def client(models_dir, tmp_path):
    samples = tmp_path / "samples"
    samples.mkdir()
    (samples / "cat_01.png").write_bytes(png_bytes(128, 128, seed=3))
    return TestClient(create_app(models_dir, samples))


def post(client, url, data=None, content=None, ctype="image/png", name="x.png"):
    files = {"file": (name, content, ctype)} if content is not None else None
    return client.post(url, data=data or {}, files=files)


# ---------------------------------------------------------------- health / meta / samples
def test_health_meta_samples(client):
    h = client.get("/api/health").json()
    assert h["status"] == "ok" and h["ready"] and len(h["models"]) == 7 and all(m["loaded"] for m in h["models"].values())
    m = client.get("/api/meta").json()
    assert m["image_size"] == 128 and m["styles"] == ["Style 1", "Style 2", "Style 3"] and len(m["classes"]) == 4
    s = client.get("/api/samples").json()
    assert [x["id"] for x in s] == ["cat_01"] and client.get(s[0]["url"]).status_code == 200
    assert client.get("/api/samples/..%2Fsecret").status_code == 404 and client.get("/api/samples/nope").status_code == 404


def test_missing_models_report_503(tmp_path):
    c = TestClient(create_app(tmp_path / "empty", tmp_path))
    h = c.get("/api/health").json()
    assert not h["ready"] and not any(m["loaded"] for m in h["models"].values())
    r = post(c, "/api/universal", content=png_bytes())
    assert r.status_code == 503 and "universal_restoration.onnx" in r.json()["detail"]


# ---------------------------------------------------------------- Task 1
def test_universal_with_corruption(client):
    r = post(client, "/api/universal", {"corruption": "salt", "severity": "medium", "seed": 7}, png_bytes())
    assert r.status_code == 200, r.text
    j = r.json()
    for k in ("original", "input", "restored"):
        assert decode_url(j[k]).size == (128, 128)
    assert j["settings"]["type"] == "salt" and j["settings"]["probability"] == 0.08 and j["settings"]["severity"] == "medium"
    assert j["inference_ms"] > 0 and j["total_ms"] >= j["inference_ms"]
    assert set(j["metrics"]) == {"input_psnr", "restored_psnr", "input_ssim", "restored_ssim"}
    assert j["metrics"]["input_psnr"] < 40  # the corruption really degraded the image


def test_universal_without_corruption_has_no_reference(client):
    j = post(client, "/api/universal", {"corruption": "none"}, png_bytes()).json()
    assert j["original"] is None and j["metrics"] is None and j["settings"]["type"] == "clean"


def test_same_seed_same_corruption_different_seed_differs(client):
    f = lambda seed: post(client, "/api/universal", {"corruption": "salt", "severity": "high", "seed": seed}, png_bytes()).json()["input"]
    assert f(5) == f(5) and f(5) != f(6)


def test_severity_presets_match_the_assignment(client):
    g = lambda c, s: post(client, "/api/universal", {"corruption": c, "severity": s}, png_bytes()).json()["settings"]
    assert g("blur", "high")["kernel_size"] == 7 and g("blur", "high")["sigma"] == 2.5
    assert g("blur", "low")["kernel_size"] == 3 and g("blur", "low")["sigma"] == 0.7
    assert g("salt", "low")["probability"] == 0.03 and g("salt", "high")["probability"] == 0.15
    occ = g("occlusion", "high")
    assert len(occ["rectangles"]) == 3 and abs(occ["area_covered"] - 0.35) < 0.03


def test_custom_parameters_and_validation(client):
    ok = post(client, "/api/universal", {"corruption": "salt", "severity": "custom", "salt_p": 0.2}, png_bytes()).json()
    assert ok["settings"]["probability"] == 0.2
    for bad in ({"corruption": "salt", "severity": "custom", "salt_p": 0.9}, {"corruption": "blur", "severity": "custom", "blur_kernel": 4},
                {"corruption": "fire"}, {"corruption": "salt", "severity": "extreme"}, {"corruption": "occlusion", "severity": "custom", "occ_rects": 9}):
        assert post(client, "/api/universal", bad, png_bytes()).status_code == 422, bad


def test_sample_id_input(client):
    j = post(client, "/api/universal", {"sample_id": "cat_01", "corruption": "occlusion", "severity": "low"}).json()
    assert j["settings"]["type"] == "occlusion" and j["metrics"] is not None
    assert post(client, "/api/universal", {"sample_id": "nope"}).status_code == 404


# ---------------------------------------------------------------- Task 2
def test_hard_routing_response_is_consistent(client):
    for corruption in ("salt", "blur", "occlusion", "none"):
        j = post(client, "/api/hard", {"corruption": corruption, "severity": "high", "seed": 1}, png_bytes()).json()
        p = j["probabilities"]
        assert list(p) == config.CLASS_NAMES and abs(sum(p.values()) - 1) < 1e-4
        pred = j["predicted_class"]
        assert pred == max(p, key=p.get)
        expected = {"clean": "none (identity bypass)", "salt-and-pepper": "salt-and-pepper specialist",
                    "gaussian blur": "blur specialist", "occlusion": "occlusion specialist"}[pred]
        assert j["selected_expert"] == expected
        assert j["classifier_ms"] > 0 and (j["expert_ms"] == 0) == (pred == "clean")
        if pred == "clean":   # identity bypass: the output IS the input
            assert j["restored"] == j["input"]


def test_hard_routing_dispatch_uses_the_right_expert(client, monkeypatch):
    """Force each classifier decision and check which expert model runs (clean runs none)."""
    store = client.app.router  # noqa: F841  (create_app keeps the store in a closure; patch via the module class)
    from app import models as mm
    calls = []
    real = mm.ModelStore._run

    def fake(self, name, feed):
        if name == "classifier":
            probs = np.zeros((1, 4), np.float32)
            probs[0, fake.target] = 1.0
            return [probs], 0.5
        calls.append(name)
        return real(self, name, feed)

    monkeypatch.setattr(mm.ModelStore, "_run", fake)
    for target, expert in ((0, None), (1, "specialist_salt"), (2, "specialist_blur"), (3, "specialist_occlusion")):
        fake.target = target
        calls.clear()
        r = post(client, "/api/hard", {"corruption": "salt"}, png_bytes())
        assert r.status_code == 200 and calls == ([expert] if expert else [])


# ---------------------------------------------------------------- Task 3
def test_soft_moe_weights(client):
    j = post(client, "/api/soft", {"corruption": "blur", "severity": "medium", "seed": 2}, png_bytes()).json()
    w = j["weights"]
    assert list(w) == config.BRANCH_NAMES and abs(sum(w.values()) - 1) < 1e-4 and all(v >= 0 for v in w.values())
    rank = [r["weight"] for r in j["contribution_ranking"]]
    assert rank == sorted(rank, reverse=True) and j["dominant_branch"] == j["contribution_ranking"][0]["branch"]
    assert decode_url(j["restored"]).size == (128, 128) and j["inference_ms"] > 0


# ---------------------------------------------------------------- Task 4
def test_sketch_styles(client):
    out = {}
    for style in (1, 2, 3):
        r = post(client, "/api/sketch", {"style": style}, png_bytes(), name="face.png")
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["style"] == f"Style {style}" and decode_url(j["sketch"]).size == (128, 128) and decode_url(j["photo"]).size == (128, 128)
        out[style] = j["sketch"]
    assert out[1] != out[2] != out[3]               # the style condition changes the generated sketch
    for bad in (0, 4):
        assert post(client, "/api/sketch", {"style": bad}, png_bytes()).status_code == 422


# ---------------------------------------------------------------- upload validation
def test_upload_validation(client, monkeypatch):
    url = "/api/universal"
    assert post(client, url, content=b"hello", ctype="text/plain", name="a.txt").status_code == 415
    assert post(client, url, content=b"not really a png", ctype="image/png").status_code == 400
    assert post(client, url, content=b"", ctype="image/png").status_code in (400, 422)
    assert post(client, url, content=png_bytes(8, 8)).status_code == 400           # smaller than 16x16
    assert client.post(url, data={}).status_code == 422                            # neither file nor sample
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 0.001)
    assert post(client, url, content=png_bytes(400, 300)).status_code == 413
    monkeypatch.setattr(config, "MAX_UPLOAD_MB", 10)
    monkeypatch.setattr(config, "MAX_PIXELS", 1000)
    assert post(client, url, content=png_bytes(160, 120)).status_code == 413
    monkeypatch.setattr(config, "MAX_PIXELS", 40_000_000)
    for fmt, ctype in (("JPEG", "image/jpeg"), ("PNG", "image/png"), ("BMP", "image/bmp")):
        assert post(client, url, content=png_bytes(300, 200, fmt=fmt), ctype=ctype).status_code == 200


# ---------------------------------------------------------------- metrics
def test_ssim_psnr_match_training_definition():
    from pytorch_msssim import ssim as ref_ssim
    r = np.random.default_rng(0)
    a = np.clip(np.kron(r.random((16, 16, 3)), np.ones((8, 8, 1))), 0, 1).astype(np.float32)
    b = np.clip(a + 0.1 * r.standard_normal(a.shape), 0, 1).astype(np.float32)
    t = lambda x: torch.from_numpy(x).permute(2, 0, 1)[None]
    assert abs(imaging.ssim(a, b) - float(ref_ssim(t(a), t(b), data_range=1.0))) < 1e-3
    assert imaging.ssim(a, a) > 0.9999 and imaging.psnr(a, a) == 99.0
    assert abs(imaging.psnr(a, b) - 10 * np.log10(1 / np.mean((a.astype(np.float64) - b) ** 2))) < 1e-9
