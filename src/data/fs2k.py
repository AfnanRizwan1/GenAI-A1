"""FS2K paired photo/sketch data helpers (Task 4).

Splits, paired augmentation and a GPU-resident batcher. Loading the raw dataset files from disk lives in
`load_fs2k` (added once the on-disk layout is known) and returns uint8 arrays.

Images are stored as uint8 (N,128,128,3); photo i and sketch i always share index i (pairing is never broken).
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

SEED = 42
VAL_FRAC = 0.15


def stratified_split(styles, val_frac=VAL_FRAC, seed=SEED):
    """Indices of a train/val split of the official training portion, stratified by sketch style."""
    styles = np.asarray(styles)
    rng = np.random.default_rng(seed)
    train, val = [], []
    for s in np.unique(styles):
        idx = rng.permutation(np.where(styles == s)[0])
        n_val = int(round(len(idx) * val_frac))
        val.extend(idx[:n_val]), train.extend(idx[n_val:])
    return np.sort(np.array(train)), np.sort(np.array(val))


def paired_augment(photo, sketch, generator=None, max_rot_deg=5.0, scale=(0.9, 1.0), max_shift=0.05, p_flip=0.5):
    """Random flip / small rotation / zoom / shift applied IDENTICALLY to photo and sketch.

    Both images are stacked on the channel axis and warped by one affine grid per sample, so the pixel-level
    correspondence is preserved by construction. Inputs (B,3,H,W) in [-1,1].
    """
    b, dev = photo.shape[0], photo.device
    rnd = lambda *s: torch.rand(*s, device=dev, generator=generator)
    ang = (rnd(b) * 2 - 1) * max_rot_deg * math.pi / 180
    sc = scale[0] + (scale[1] - scale[0]) * rnd(b)
    flip = torch.where(rnd(b) < p_flip, -1.0, 1.0)
    tx, ty = (rnd(b) * 2 - 1) * max_shift, (rnd(b) * 2 - 1) * max_shift
    cos, sin = torch.cos(ang) * sc, torch.sin(ang) * sc
    theta = torch.stack([torch.stack([cos * flip, -sin, tx], 1), torch.stack([sin * flip, cos, ty], 1)], 1)
    x = torch.cat([photo, sketch], 1)
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    return x[:, :3], x[:, 3:]


def to_tensor(imgs_uint8, device):
    """(N,H,W,3) uint8 -> (N,3,H,W) float in [-1,1]."""
    return torch.from_numpy(imgs_uint8).to(device).permute(0, 3, 1, 2).float().div_(127.5).sub_(1)


class PairBatcher:
    """Holds one split (photos, sketches, styles) on the device and yields shuffled (photo, sketch, style)."""

    def __init__(self, photos, sketches, styles, batch_size, device="cuda", shuffle=True, drop_last=True, seed=SEED):
        assert len(photos) == len(sketches) == len(styles)
        self.p = torch.from_numpy(photos).to(device).permute(0, 3, 1, 2).contiguous()   # uint8
        self.s = torch.from_numpy(sketches).to(device).permute(0, 3, 1, 2).contiguous()
        self.st = torch.from_numpy(np.asarray(styles)).long().to(device)
        self.bs, self.shuffle, self.drop_last, self.device = min(batch_size, len(self.p)), shuffle, drop_last, device
        self.gen = torch.Generator(device="cpu").manual_seed(seed)

    def __len__(self):
        n = len(self.p)
        return n // self.bs if self.drop_last else -(-n // self.bs)

    def __iter__(self):
        n = len(self.p)
        order = torch.randperm(n, generator=self.gen) if self.shuffle else torch.arange(n)
        for i in range(len(self)):
            idx = order[i * self.bs:(i + 1) * self.bs].to(self.device)
            yield (self.p[idx].float().div_(127.5).sub_(1), self.s[idx].float().div_(127.5).sub_(1), self.st[idx])


# ------------------------------------------------------------------ raw dataset loading
import json  # noqa: E402
import re  # noqa: E402
from pathlib import Path  # noqa: E402

from PIL import Image  # noqa: E402

SIZE = 128
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def _digits(stem):
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else stem


def _index_folder(folder):
    """trailing-number -> file path for every image in a folder (photo/sketch naming differs, the number does not)."""
    idx = {}
    for p in sorted(Path(folder).iterdir()):
        if p.suffix.lower() in IMG_EXT:
            idx[_digits(p.stem)] = p
    return idx


def _find_root(root):
    root = Path(root)
    for cand in (root, root / "FS2K"):
        if (cand / "anno_train.json").exists():
            return cand
    for p in root.rglob("anno_train.json"):  # extracted into some other sub-folder
        return p.parent
    raise FileNotFoundError(f"anno_train.json not found under {root}")


def _load_split(root, anno_file):
    anno = json.loads((root / anno_file).read_text())
    photos, sketches, styles, names = [], [], [], []
    cache = {}
    for a in anno:
        folder, stem = a["image_name"].split("/")           # e.g. photo1/image0110
        k = folder.replace("photo", "")
        if k not in cache:
            cache[k] = (_index_folder(root / "photo" / f"photo{k}"), _index_folder(root / "sketch" / f"sketch{k}"))
        pidx, sidx = cache[k]
        d = _digits(stem)
        if d not in pidx or d not in sidx:
            raise FileNotFoundError(f"no photo/sketch pair for {a['image_name']} (number {d})")
        photos.append(np.asarray(Image.open(pidx[d]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)))
        sketches.append(np.asarray(Image.open(sidx[d]).convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)))
        styles.append(int(a["style"]))
        names.append(a["image_name"])
    return {"photo": np.stack(photos), "sketch": np.stack(sketches), "style": np.array(styles), "name": np.array(names)}


def load_fs2k(root="data/fs2k"):
    """Official FS2K split: train (minus a 15% style-stratified validation set, seed 42) / val / official test.

    Returns {'train','val','test'} dicts with uint8 photo/sketch (N,128,128,3), int style, and image names.
    Cached as <root>/cache_128.npz after the first call.
    """
    base = _find_root(root)
    cache = base / f"cache_{SIZE}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        full = {s: {k: z[f"{s}_{k}"] for k in ("photo", "sketch", "style", "name")} for s in ("train", "test")}
    else:
        full = {"train": _load_split(base, "anno_train.json"), "test": _load_split(base, "anno_test.json")}
        np.savez_compressed(cache, **{f"{s}_{k}": v for s, d in full.items() for k, v in d.items()})
    tr, va = stratified_split(full["train"]["style"])
    pick = lambda d, idx: {k: v[idx] for k, v in d.items()}
    return {"train": pick(full["train"], tr), "val": pick(full["train"], va), "test": full["test"]}
