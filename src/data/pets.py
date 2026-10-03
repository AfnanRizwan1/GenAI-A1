"""Oxford-IIIT Pet loading, caching and the fixed 80/20 split (seed 42).

All images are converted to RGB and resized to 128x128 once and cached as a uint8
array, so training only pays for the (GPU) corruption, not for image decoding.
"""
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

SIZE = 128
SEED = 42
DEFAULT_ROOT = Path("data")
SMOKE = int(os.environ.get("GENAI_SMOKE", "0"))


def _load_split(root, split):
    cache = Path(root) / "cache" / f"pets_{split}_{SIZE}.npy"
    if cache.exists():
        return np.load(cache)
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(root=str(root), split=split, target_types="category", download=True)
    imgs = np.stack([np.asarray(img.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)) for img, _ in ds])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, imgs)
    return imgs


def split_indices(n, val_frac=0.2, seed=SEED):
    perm = np.random.default_rng(seed).permutation(n)
    n_val = int(round(n * val_frac))
    return np.sort(perm[n_val:]), np.sort(perm[:n_val])


def load_pets(root=DEFAULT_ROOT):
    """Returns dict with uint8 arrays 'train', 'val' (from official trainval) and 'test'."""
    trainval = _load_split(root, "trainval")
    test = _load_split(root, "test")
    tr, va = split_indices(len(trainval))
    train = trainval[tr]
    if SMOKE:  # GENAI_SMOKE=n: tiny runs to smoke-test pipelines on CPU
        train = train[:SMOKE]
    return {"train": train, "val": trainval[va], "test": test}


def save_split_file(path="manifests/splits.json", n=3680):
    tr, va = split_indices(n)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"seed": SEED, "n_trainval": n, "train": tr.tolist(), "val": va.tolist()}))


def to_tensor(imgs_uint8, device="cpu"):
    """(N,H,W,3) uint8 -> (N,3,H,W) float32 in [0,1]."""
    return torch.from_numpy(imgs_uint8).to(device).permute(0, 3, 1, 2).float().div_(255)


class GpuBatcher:
    """Holds the whole clean split on the device and yields shuffled batches.

    No DataLoader / worker processes: the split is only a few hundred MB, and
    corruption is applied on the GPU by corruptions.corrupt_batch.
    """

    def __init__(self, imgs_uint8, batch_size, device="cuda", shuffle=True, drop_last=True, seed=SEED):
        self.x = torch.from_numpy(imgs_uint8).to(device).permute(0, 3, 1, 2).contiguous()  # uint8
        self.bs, self.shuffle, self.drop_last, self.device = min(batch_size, len(self.x)), shuffle, drop_last, device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        n = len(self.x)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        n = len(self.x)
        order = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        for i in range(len(self)):
            idx = order[i * self.bs:(i + 1) * self.bs].to(self.device)
            yield self.x[idx].float().div_(255)
