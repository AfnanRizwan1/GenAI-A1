import cv2
import numpy as np
import torch

import src.data.corruptions as C
from src.data.corruptions import apply_spec, corrupt_batch
from src.data.manifest import build_manifest


def test_manifest_deterministic():
    a = build_manifest(5, "test", 7)
    b = build_manifest(5, "test", 7)
    assert a == b and len(a) == 5 * 10


def test_test_manifest_levels():
    m = build_manifest(3, "test", 7)
    sp = {e["level"]: e["p"] for e in m if e["type"] == "salt"}
    assert sp == {"low": 0.03, "medium": 0.08, "high": 0.15}
    occ = [(len(e["rects"]), e["coverage"]) for e in m if e["type"] == "occlusion"][:3]
    for (n, cov), (tn, tc) in zip(occ, [(1, .10), (2, .20), (3, .35)]):
        assert n == tn and abs(cov - tc) < 0.03, (n, cov)


def test_apply_deterministic():
    img = np.random.default_rng(0).random((128, 128, 3)).astype(np.float32)
    for e in build_manifest(1, "val", 3):
        assert np.array_equal(apply_spec(img, e), apply_spec(img, e))


def test_gpu_corruption_matches_spec():
    x = torch.rand(256, 3, 128, 128)
    out, lab = corrupt_batch(x, torch.arange(256) % 4)
    assert torch.equal(out[lab == 0], x[lab == 0])
    assert (out[lab == 1] != x[lab == 1]).float().mean() > 0.01
    occ = ((out[lab == 3] == 0).all(1)).float().mean((1, 2))
    assert 0.05 < occ.mean() < 0.36 and occ.max() < 0.40
    d = lambda t: (t[..., 1:] - t[..., :-1]).abs().mean()
    assert d(out[lab == 2]) < d(x[lab == 2])


def test_gpu_blur_matches_cv2():
    x = torch.rand(1, 3, 128, 128)
    kern = C._gauss_kernels(torch.tensor([5]), torch.tensor([1.5])).repeat_interleave(3, 0)
    y = torch.nn.functional.pad(x, (3, 3, 0, 0), mode="reflect")
    y = torch.nn.functional.conv2d(y, kern[:, None, None, :], groups=3)
    y = torch.nn.functional.pad(y, (0, 0, 3, 3), mode="reflect")
    y = torch.nn.functional.conv2d(y, kern[:, None, :, None], groups=3)
    ref = cv2.GaussianBlur(x[0].permute(1, 2, 0).numpy(), (5, 5), 1.5)
    assert np.abs(y[0].permute(1, 2, 0).numpy() - ref).max() < 1e-4
