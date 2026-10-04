"""Deterministic validation / test corruption manifests (Tasks 1-3).

Each entry stores: image index, corruption type, severity level, seed, and the
parameters (salt probability, blur kernel+sigma, rectangle coordinates).
"""
import gzip
import json
import os
from pathlib import Path

import numpy as np
import torch

from .corruptions import CLASSES, LEVELS, apply_spec, make_spec

MANIFEST_DIR = Path("manifests")


def build_manifest(n_images, mode, base_seed):
    """mode 'val': clean + one corruption of each type per image (severity cycled, sampled in-range).
    mode 'test': clean + 3 corruptions x 3 fixed severities per image (10 entries)."""
    entries, counter = [], 0
    for i in range(n_images):
        for kind in CLASSES:
            if kind == "clean":
                combos = [None]
            elif mode == "test":
                combos = LEVELS
            else:
                combos = [LEVELS[(i + CLASSES.index(kind)) % 3]]
            for level in combos:
                seed = base_seed + counter
                counter += 1
                rng = np.random.default_rng(seed)
                spec = make_spec(kind, rng, seed, level=level, fixed=(mode == "test"))
                spec["idx"] = i
                spec["label"] = CLASSES.index(kind)
                entries.append(spec)
    return entries


def save_manifest(entries, name):
    MANIFEST_DIR.mkdir(exist_ok=True)
    with gzip.open(MANIFEST_DIR / f"{name}.json.gz", "wt") as f:
        json.dump(entries, f, separators=(",", ":"))


def load_manifest(name):
    with gzip.open(MANIFEST_DIR / f"{name}.json.gz", "rt") as f:
        return json.load(f)


def materialize(imgs_uint8, entries, device="cpu", labels_keep=None, batch_size=256):
    """Replay a manifest once and keep the result in memory (used for per-epoch validation).

    Returns dict(corr, clean, labels, levels) with tensors on `device`."""
    if labels_keep is not None:
        entries = [e for e in entries if e["label"] in labels_keep]
    smoke = int(os.environ.get("GENAI_SMOKE", "0"))
    if smoke:  # tiny runs to smoke-test pipelines on CPU
        entries = entries[:smoke]
    corr, clean, labels = [], [], []
    for c, t, l, _ in iter_manifest(imgs_uint8, entries, batch_size, device):
        corr.append(c), clean.append(t), labels.append(l)
    return {"corr": torch.cat(corr), "clean": torch.cat(clean), "labels": torch.cat(labels),
            "levels": [e["level"] for e in entries]}


def iter_manifest(imgs_uint8, entries, batch_size=128, device="cpu"):
    """Yield (corrupted, clean, labels, entries_in_batch) by replaying the manifest."""
    for s in range(0, len(entries), batch_size):
        chunk = entries[s:s + batch_size]
        clean = np.stack([imgs_uint8[e["idx"]] for e in chunk]).astype(np.float32) / 255
        corr = np.stack([apply_spec(c, e) for c, e in zip(clean, chunk)])
        to_t = lambda a: torch.from_numpy(a).permute(0, 3, 1, 2).contiguous().to(device)
        yield to_t(corr), to_t(clean), torch.tensor([e["label"] for e in chunk], device=device), chunk
