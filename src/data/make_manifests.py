"""Generate the fixed validation and test corruption manifests (run once, commit the output).

    python -m src.data.make_manifests
"""
import argparse

from .manifest import build_manifest, save_manifest
from .pets import save_split_file

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trainval", type=int, default=3680)   # official Oxford-IIIT Pet trainval size
    ap.add_argument("--n-test", type=int, default=3669)       # official test size
    a = ap.parse_args()
    save_split_file(n=a.n_trainval)
    n_val = int(round(a.n_trainval * 0.2))
    val = build_manifest(n_val, "val", base_seed=1_000_000)
    test = build_manifest(a.n_test, "test", base_seed=2_000_000)
    save_manifest(val, "val")
    save_manifest(test, "test")
    print(f"val: {n_val} images -> {len(val)} entries; test: {a.n_test} images -> {len(test)} entries")
