"""Corruptions for Tasks 1-3.

Two implementations of the same specification:
  * numpy / cv2, parameter-driven and deterministic -> manifests, evaluation, web backend
  * torch, batched, runs on the GPU                  -> runtime corruption while training

Labels: 0 clean, 1 salt-and-pepper, 2 gaussian blur, 3 rectangular occlusion.
Images are float32 in [0, 1]; numpy images are HxWx3, torch images are Bx3xHxW.
"""
import math

import cv2
import numpy as np
import torch

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


# ---------------------------------------------------------------- torch side
def _gauss_kernels(k, sigma, size=7):
    """Per-sample 1-D gaussian taps padded to `size` (zeros outside the real kernel)."""
    t = torch.arange(size, device=k.device).float() - size // 2           # (size,)
    wgt = torch.exp(-(t[None] ** 2) / (2 * sigma[:, None] ** 2))           # (B,size)
    wgt = wgt * (t[None].abs() <= (k[:, None] - 1) / 2)
    return wgt / wgt.sum(1, keepdim=True)


def corrupt_batch(x, labels=None, generator=None):
    """Runtime corruption of a clean batch x (B,3,H,W) in [0,1] on its own device.

    labels: optional LongTensor (B,) of conditions; otherwise each sample picks
    one of the 4 conditions with equal probability. Returns (corrupted, labels).
    """
    b, _, h, w = x.shape
    dev = x.device
    rnd = lambda *s: torch.rand(*s, device=dev, generator=generator)
    if labels is None:
        labels = torch.randint(0, 4, (b,), device=dev, generator=generator)
    out = x.clone()

    # salt & pepper (pixel-level, same value on all channels)
    idx = (labels == 1).nonzero(as_tuple=True)[0]
    if len(idx):
        p = SALT_RANGE[0] + (SALT_RANGE[1] - SALT_RANGE[0]) * rnd(len(idx))
        hit = rnd(len(idx), 1, h, w) < p[:, None, None, None]
        white = (rnd(len(idx), 1, h, w) < 0.5).float().expand(-1, 3, -1, -1)
        out[idx] = torch.where(hit, white, x[idx])

    # gaussian blur (separable, reflect padding like cv2)
    idx = (labels == 2).nonzero(as_tuple=True)[0]
    if len(idx):
        n = len(idx)
        ks = torch.tensor(BLUR_KERNELS, device=dev)[torch.randint(0, 3, (n,), device=dev, generator=generator)]
        sg = BLUR_SIGMA_RANGE[0] + (BLUR_SIGMA_RANGE[1] - BLUR_SIGMA_RANGE[0]) * rnd(n)
        kern = _gauss_kernels(ks, sg).repeat_interleave(3, 0)               # (n*3,7)
        y = x[idx].reshape(1, n * 3, h, w)
        y = torch.nn.functional.pad(y, (3, 3, 0, 0), mode="reflect")
        y = torch.nn.functional.conv2d(y, kern[:, None, None, :], groups=n * 3)
        y = torch.nn.functional.pad(y, (0, 0, 3, 3), mode="reflect")
        y = torch.nn.functional.conv2d(y, kern[:, None, :, None], groups=n * 3)
        out[idx] = y.reshape(n, 3, h, w)

    # rectangular occlusion (1-3 black rectangles, 10-35 % total area)
    idx = (labels == 3).nonzero(as_tuple=True)[0]
    if len(idx):
        n = len(idx)
        nrect = torch.randint(1, OCC_MAX_RECTS + 1, (n,), device=dev, generator=generator)
        cov = OCC_COVERAGE_RANGE[0] + (OCC_COVERAGE_RANGE[1] - OCC_COVERAGE_RANGE[0]) * rnd(n)
        active = torch.arange(OCC_MAX_RECTS, device=dev)[None] < nrect[:, None]   # (n,3)
        share = (rnd(n, OCC_MAX_RECTS) + 0.5) * active
        share = share / share.sum(1, keepdim=True)
        area = (share * (cov * h * w)[:, None]).clamp(min=1)
        ar = torch.exp(math.log(0.5) + (math.log(2.0) - math.log(0.5)) * rnd(n, OCC_MAX_RECTS))
        rw = torch.sqrt(area * ar).round().clamp(1, w)
        rh = (area / rw).round().clamp(1, h)
        x0 = (rnd(n, OCC_MAX_RECTS) * (w - rw + 1)).floor()
        y0 = (rnd(n, OCC_MAX_RECTS) * (h - rh + 1)).floor()
        xs = torch.arange(w, device=dev).view(1, 1, 1, w)
        ys = torch.arange(h, device=dev).view(1, 1, h, 1)
        inside = ((xs >= x0[:, :, None, None]) & (xs < (x0 + rw)[:, :, None, None]) &
                  (ys >= y0[:, :, None, None]) & (ys < (y0 + rh)[:, :, None, None]) &
                  active[:, :, None, None])
        mask = inside.any(1, keepdim=True)                                       # (n,1,h,w)
        out[idx] = x[idx].masked_fill(mask, 0.0)
    return out, labels
