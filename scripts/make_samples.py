"""Create the sample images shown in the app's "pick a sample" row.

Picks N pictures from the Oxford-IIIT Pet *test* split (fixed seed), center-crops them to a square and saves
256x256 JPEGs. Only pet photos are used (no face photos are redistributed).

    python scripts/make_samples.py --data-root data --out app/backend/samples --n 8
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def make_samples(dataset, out, n=8, seed=0, size=256):
    """dataset: indexable of (PIL image, label). Returns the written paths."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    idx = np.random.default_rng(seed).choice(len(dataset), n, replace=False)
    paths = []
    for k, i in enumerate(sorted(int(j) for j in idx), start=1):
        img = dataset[i][0].convert("RGB")
        s = min(img.size)
        left, top = (img.width - s) // 2, (img.height - s) // 2
        img = img.crop((left, top, left + s, top + s)).resize((size, size), Image.LANCZOS)
        p = out / f"pet_{k:02d}.jpg"
        img.save(p, quality=92)
        paths.append(p)
    return paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--out", default="app/backend/samples")
    ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from torchvision.datasets import OxfordIIITPet
    ds = OxfordIIITPet(root=a.data_root, split="test", download=True)
    print("wrote", len(make_samples(ds, a.out, a.n)), "samples to", a.out)
