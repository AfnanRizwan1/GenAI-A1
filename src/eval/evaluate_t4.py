"""Test-set evaluation of the Task 4 face-to-sketch generator on the official FS2K test split.

Reports L1 / SSIM / PSNR (images in [0,1]) overall and per sketch style, a grayscale-photo baseline for context,
and a style-conditioning check: every test photo is generated with all three styles and compared with its
ground-truth sketch (rows: true style, columns: style fed to the generator; a generator that really uses the
style condition should do best on the diagonal). Also writes example grids (photo, the three styles, target) and
failure cases.

    python -m src.eval.evaluate_t4 --gen outputs/t4/best.pt --data-root data/fs2k --out outputs/eval_t4
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.common.metrics import l1_per_image, psnr_per_image, ssim_per_image  # noqa: E402
from src.data.fs2k import load_fs2k, to_tensor  # noqa: E402
from src.models.cgan import NUM_STYLES  # noqa: E402
from src.models.loading import load_generator  # noqa: E402

STYLE_NAMES = ["Style 1", "Style 2", "Style 3"]
unit = lambda x: (x + 1) / 2


def gray_baseline(photo):
    """The photograph converted to grayscale (3 channels): what you get without any generator."""
    g = (0.299 * photo[:, 0] + 0.587 * photo[:, 1] + 0.114 * photo[:, 2]).unsqueeze(1)
    return g.expand(-1, 3, -1, -1)


@torch.no_grad()
def run(G, test, device, bs=64):
    n = len(test["photo"])
    rows, gen_all = [], np.zeros((n, NUM_STYLES), dtype=np.float32)  # per-image L1 to target for each fed style
    keep = {}                                                        # generated images for the figures, filled lazily
    for i in range(0, n, bs):
        x = to_tensor(test["photo"][i:i + bs], device)
        y = to_tensor(test["sketch"][i:i + bs], device)
        st = torch.from_numpy(test["style"][i:i + bs]).long().to(device)
        outs = {s: G(x, torch.full_like(st, s)) for s in range(NUM_STYLES)}
        base = gray_baseline(unit(x))
        for s in range(NUM_STYLES):
            gen_all[i:i + len(x), s] = l1_per_image(unit(outs[s]), unit(y)).cpu().numpy()
        true_out = torch.stack([outs[int(k)][j] for j, k in enumerate(st)])      # generated with the true style
        m = {"gen": (l1_per_image(unit(true_out), unit(y)), ssim_per_image(unit(true_out), unit(y)), psnr_per_image(unit(true_out), unit(y))),
             "gray": (l1_per_image(base, unit(y)), ssim_per_image(base, unit(y)), psnr_per_image(base, unit(y)))}
        for j in range(len(x)):
            r = {"idx": i + j, "name": str(test["name"][i + j]), "style": int(st[j])}
            for k, (l1, ss, ps) in m.items():
                r[f"{k}_l1"], r[f"{k}_ssim"], r[f"{k}_psnr"] = float(l1[j]), float(ss[j]), float(ps[j])
            rows.append(r)
    return pd.DataFrame(rows), gen_all


@torch.no_grad()
def example_grid(path, G, test, idxs, device):
    """photo | generated Style 1 | Style 2 | Style 3 | ground-truth sketch (true style marked)."""
    x = to_tensor(test["photo"][idxs], device)
    y = to_tensor(test["sketch"][idxs], device)
    outs = [G(x, torch.full((len(idxs),), s, dtype=torch.long, device=device)) for s in range(NUM_STYLES)]
    fig, ax = plt.subplots(len(idxs), 5, figsize=(9.5, 1.9 * len(idxs)))
    ax = np.atleast_2d(ax)
    show = lambda t: unit(t).permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    for r, j in enumerate(idxs):
        cols = [(x[r], "photo")] + [(outs[s][r], f"generated {STYLE_NAMES[s]}") for s in range(NUM_STYLES)] + [(y[r], "target sketch")]
        for c, (im, name) in enumerate(cols):
            ax[r, c].imshow(show(im)), ax[r, c].set_xticks([]), ax[r, c].set_yticks([])
            if r == 0:
                ax[r, c].set_title(name, fontsize=8)
        ax[r, 4].set_xlabel(f"true: {STYLE_NAMES[int(test['style'][j])]}", fontsize=7)
    fig.tight_layout(), fig.savefig(path, dpi=140), plt.close(fig)


def plot_matrix(m, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    im = ax.imshow(m, cmap="viridis_r")
    ax.set_xticks(range(3), [f"fed {s}" for s in STYLE_NAMES], rotation=20), ax.set_yticks(range(3), [f"true {s}" for s in STYLE_NAMES])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.4f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, fraction=0.046, label="mean L1 to target sketch"), fig.tight_layout()
    fig.savefig(path, dpi=200), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--data-root", default="data/fs2k")
    ap.add_argument("--out", default="outputs/eval_t4")
    ap.add_argument("--max-images", type=int, default=0, help="debug: only the first N test images")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    G = load_generator(a.gen, dev)
    test = load_fs2k(a.data_root)["test"]
    if a.max_images:
        test = {k: v[:a.max_images] for k, v in test.items()}
    df, gen_all = run(G, test, dev)
    df.to_csv(out / "per_image.csv", index=False)

    cols = [f"{k}_{m}" for k in ("gen", "gray") for m in ("l1", "ssim", "psnr")]
    by_style = df.groupby("style")[cols].mean()
    by_style.index = [STYLE_NAMES[i] for i in by_style.index]
    overall = df[cols].mean()
    styles = df["style"].to_numpy()
    matrix = np.full((NUM_STYLES, NUM_STYLES), np.nan)
    for t in range(NUM_STYLES):
        if (styles == t).any():
            matrix[t] = gen_all[styles == t].mean(0)
    plot_matrix(np.nan_to_num(matrix), out / "style_conditioning_matrix.png")
    diff = [float(np.abs(gen_all[:, i] - gen_all[:, j]).mean()) for i in range(3) for j in range(i + 1, 3)]
    summary = {
        "n_test_images": len(df), "images_per_style": {STYLE_NAMES[s]: int((styles == s).sum()) for s in range(NUM_STYLES)},
        "overall": overall.to_dict(), "by_style": by_style.to_dict("index"),
        "style_conditioning": {"mean_l1_matrix_true_vs_fed": matrix.tolist(),
                               "diagonal_is_best_per_row": [bool(np.nanargmin(matrix[t]) == t) for t in range(NUM_STYLES) if not np.isnan(matrix[t]).any()]},
    }
    rng = np.random.default_rng(0)
    pick = [int(i) for i in rng.choice(len(df), min(8, len(df)), replace=False)]
    example_grid(out / "examples_test.png", G, test, pick, dev)
    worst = [int(i) for i in df.sort_values("gen_l1", ascending=False).index[:4]]
    example_grid(out / "failures_test.png", G, test, worst, dev)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    with pd.option_context("display.width", 200, "display.float_format", "{:.4f}".format):
        print("\n=== overall (generator vs grayscale-photo baseline) ===\n", overall.to_string())
        print("\n=== by style ===\n", by_style.to_string())
        print("\n=== mean L1 to the target sketch: rows = true style, columns = style fed to the generator ===\n", pd.DataFrame(matrix, index=STYLE_NAMES, columns=STYLE_NAMES).to_string())
    print("wrote", out)


if __name__ == "__main__":
    main()
