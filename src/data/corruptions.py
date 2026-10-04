"""Batched GPU corruption used while training (Tasks 1-3), plus re-exports of the numpy spec.

The specification (constants, deterministic numpy implementation used for manifests, evaluation and the
web backend) lives in corruption_spec.py; this module adds the torch implementation that applies the same
distributions on the GPU, one random condition per sample.

Labels: 0 clean, 1 salt-and-pepper, 2 gaussian blur, 3 rectangular occlusion.
Images are float32 in [0, 1]; torch images are Bx3xHxW.
"""
import math

import torch

from .corruption_spec import (  # noqa: F401  (re-exported for existing imports)
    BLUR_KERNELS, BLUR_SIGMA_RANGE, CLASSES, LEVELS, OCC_COVERAGE_RANGE, OCC_MAX_RECTS, SALT_RANGE, TEST_BLUR,
    TEST_OCC, TEST_SALT, _third, apply_spec, make_spec, occlusion_mask, sample_rects)

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
