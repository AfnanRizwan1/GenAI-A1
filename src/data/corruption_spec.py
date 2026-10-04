"""Corruption specification and the deterministic numpy / cv2 implementation (Tasks 1-3).

No torch dependency, so the web backend can use exactly the corruptions the models were trained on
without installing PyTorch. The batched GPU version used while training lives in corruptions.py and
imports its constants from here.

Labels: 0 clean, 1 salt-and-pepper, 2 gaussian blur, 3 rectangular occlusion.
Images are float32 HxWx3 in [0, 1].
"""
import math

import cv2
import numpy as np

CLASSES = ["clean", "salt", "blur", "occlusion"]
LEVELS = ["low", "medium", "high"]

# training ranges from the assignment
SALT_RANGE = (0.02, 0.15)
BLUR_KERNELS = (3, 5, 7)
BLUR_SIGMA_RANGE = (0.5, 2.5)
OCC_COVERAGE_RANGE = (0.10, 0.35)
OCC_MAX_RECTS = 3

# fixed test severities
TEST_SALT = {"low": 0.03, "medium": 0.08, "high": 0.15}
TEST_BLUR = {"low": (3, 0.7), "medium": (5, 1.5), "high": (7, 2.5)}
TEST_OCC = {"low": (1, 0.10), "medium": (2, 0.20), "high": (3, 0.35)}


def _third(rng_range, level):
    """Sub-range of (lo, hi) for a severity level (equal thirds)."""
    lo, hi = rng_range
    step = (hi - lo) / 3
    i = LEVELS.index(level)
    return lo + i * step, lo + (i + 1) * step


# ---------------------------------------------------------------- numpy side
def sample_rects(rng, n, coverage, h=128, w=128):
    """n black rectangles that jointly cover ~`coverage` of the image.

    The total area is split randomly between the rectangles; each gets a random
    aspect ratio in [0.5, 2]. Placement is rejection-sampled to avoid overlap
    (falls back to overlap after 100 tries), so achieved coverage is close to
    the target.
    """
    total = coverage * h * w
    shares = rng.dirichlet(np.full(n, 4.0)) if n > 1 else np.array([1.0])
    rects = []
    for s in shares:
        area = s * total
        ar = math.exp(rng.uniform(math.log(0.5), math.log(2.0)))
        rw = int(np.clip(round(math.sqrt(area * ar)), 1, w))
        rh = int(np.clip(round(area / rw), 1, h))
        for _ in range(100):
            x = int(rng.integers(0, w - rw + 1))
            y = int(rng.integers(0, h - rh + 1))
            if all(x + rw <= ox or ox + ow <= x or y + rh <= oy or oy + oh <= y
                   for ox, oy, ow, oh in rects):
                break
        rects.append([x, y, rw, rh])
    return rects


def occlusion_mask(rects, h=128, w=128):
    m = np.zeros((h, w), dtype=bool)
    for x, y, rw, rh in rects:
        m[y:y + rh, x:x + rw] = True
    return m


def make_spec(kind, rng, seed, level=None, fixed=False, h=128, w=128):
    """Build one corruption spec (a plain dict, JSON-serialisable).

    level=None  -> sample over the full training range
    level=...   -> sample inside that severity third (validation manifest)
    fixed=True  -> use the exact test severity for `level` (test manifest)
    """
    spec = {"type": kind, "level": level, "seed": int(seed)}
    if kind == "clean":
        return spec
    if kind == "salt":
        if fixed:
            spec["p"] = TEST_SALT[level]
        else:
            lo, hi = _third(SALT_RANGE, level) if level else SALT_RANGE
            spec["p"] = float(rng.uniform(lo, hi))
    elif kind == "blur":
        if fixed:
            spec["k"], spec["sigma"] = TEST_BLUR[level]
        else:
            lo, hi = _third(BLUR_SIGMA_RANGE, level) if level else BLUR_SIGMA_RANGE
            spec["k"] = int(rng.choice(BLUR_KERNELS))
            spec["sigma"] = float(rng.uniform(lo, hi))
    elif kind == "occlusion":
        if fixed:
            n, cov = TEST_OCC[level]
        else:
            lo, hi = _third(OCC_COVERAGE_RANGE, level) if level else OCC_COVERAGE_RANGE
            n, cov = int(rng.integers(1, OCC_MAX_RECTS + 1)), float(rng.uniform(lo, hi))
        spec["rects"] = sample_rects(rng, n, cov, h, w)
        spec["coverage"] = float(occlusion_mask(spec["rects"], h, w).mean())
    else:
        raise ValueError(kind)
    return spec


def apply_spec(img, spec):
    """Apply a spec to one HxWx3 float32 image in [0, 1]. Deterministic."""
    kind = spec["type"]
    if kind == "clean":
        return img.copy()
    if kind == "salt":
        rng = np.random.default_rng(spec["seed"])
        h, w = img.shape[:2]
        hit = rng.random((h, w)) < spec["p"]
        white = rng.random((h, w)) < 0.5
        out = img.copy()
        out[hit] = white[hit][:, None].astype(np.float32)
        return out
    if kind == "blur":
        k = spec["k"]
        return cv2.GaussianBlur(img, (k, k), spec["sigma"])
    if kind == "occlusion":
        out = img.copy()
        out[occlusion_mask(spec["rects"], *img.shape[:2])] = 0.0
        return out
    raise ValueError(kind)
