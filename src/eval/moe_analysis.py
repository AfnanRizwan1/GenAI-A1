"""Routing analysis for the soft mixture-of-experts (Task 3), called from src.eval.evaluate --moe.

Outputs: mean routing weights per true corruption x severity (CSV + heatmap), expert activity checks
(dead experts, experts dominating unrelated inputs), and example grids with the four weights for
inputs where one expert dominates and inputs where the weights are spread across several experts.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data.corruptions import CLASSES  # noqa: E402
from src.data.manifest import iter_manifest  # noqa: E402
from src.models.soft_moe import BRANCHES  # noqa: E402

WCOLS = [f"w_{b}" for b in BRANCHES]
DEAD_THRESHOLD = 0.05      # mean weight below this over all inputs -> expert is effectively inactive
DOMINANCE_THRESHOLD = 0.5  # mean weight above this on inputs of OTHER corruption types -> dominates unrelated inputs


def weight_table(df):
    order = [("clean", "-")] + [(t, l) for t in CLASSES[1:] for l in ("low", "medium", "high")]
    return df.groupby(["type", "level"])[WCOLS].mean().reindex(order)


def plot_heatmap(tab, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(tab.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4), BRANCHES, rotation=25)
    ax.set_yticks(range(len(tab)), [f"{t}/{l}" for t, l in tab.index], fontsize=8)
    ax.set_xlabel("branch"), ax.set_ylabel("true corruption / severity")
    for i in range(tab.shape[0]):
        for j in range(4):
            v = tab.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="white" if v < 0.6 else "black")
    fig.colorbar(im, fraction=0.046, label="mean routing weight"), fig.tight_layout()
    fig.savefig(path, dpi=200), plt.close(fig)


def activity_report(df):
    """Dead / dominant expert checks over the whole test manifest."""
    rep, mean_w = {}, df[WCOLS].mean()
    argmax = df[WCOLS].to_numpy().argmax(1)
    for j, b in enumerate(BRANCHES):
        own = df["type"] == CLASSES[j]            # inputs this branch is responsible for
        off = df[WCOLS[j]][~own].mean() if (~own).any() else float("nan")
        rep[b] = {"mean_weight_all": float(mean_w.iloc[j]),
                  "mean_weight_on_own_type": float(df[WCOLS[j]][own].mean()),
                  "mean_weight_on_other_types": float(off),
                  "fraction_argmax": float((argmax == j).mean()),
                  "inactive": bool(mean_w.iloc[j] < DEAD_THRESHOLD),
                  "dominates_unrelated_inputs": bool(off > DOMINANCE_THRESHOLD)}
    return rep


def entropy(w):
    return -(w * np.log(np.clip(w, 1e-12, 1))).sum(1)


@torch.no_grad()
def weights_grid(path, df, rows, imgs, entries, moe, device, title):
    sel = [entries[j] for j in rows]
    corr, clean, _, _ = next(iter_manifest(imgs, sel, len(sel), device))
    y, w, _ = moe(corr)
    show = lambda t: t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    fig, ax = plt.subplots(len(rows), 4, figsize=(8.8, 2.1 * len(rows)), gridspec_kw={"width_ratios": [1, 1, 1, 1.2]})
    ax = np.atleast_2d(ax)
    for i, j in enumerate(rows):
        r = df.loc[j]
        for k, (im, name) in enumerate([(show(clean[i]), "clean target"), (show(corr[i]), "input"), (show(y[i]), "soft MoE")]):
            ax[i, k].imshow(im), ax[i, k].set_xticks([]), ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(name, fontsize=9)
        ax[i, 0].set_ylabel(f"{r['type']}/{r['level']}", fontsize=7)
        ax[i, 2].set_xlabel(f"{r['moe_psnr']:.1f} dB", fontsize=7)
        ax[i, 3].barh(range(4), w[i].cpu().numpy(), color=["#888", "#d95f02", "#1b9e77", "#7570b3"])
        ax[i, 3].set_yticks(range(4), BRANCHES, fontsize=7), ax[i, 3].set_xlim(0, 1), ax[i, 3].invert_yaxis()
        if i == 0:
            ax[i, 3].set_title("routing weights", fontsize=9)
    fig.suptitle(title, fontsize=10), fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)


def analyze(df, imgs, entries, moe, out, device, k=4):
    tab = weight_table(df)
    tab.to_csv(out / "moe_weights_by_condition.csv")
    plot_heatmap(tab, out / "moe_routing_heatmap.png")
    rep = activity_report(df)
    d = df[df["type"] != "clean"]
    ent = entropy(d[WCOLS].to_numpy())
    order = np.argsort(ent)
    dominant = [int(d.index[i]) for i in order[:k]]            # lowest entropy: one expert dominates
    spread = [int(d.index[i]) for i in order[::-1][:k]]         # highest entropy: weights distributed
    weights_grid(out / "moe_examples_dominant.png", df, dominant, imgs, entries, moe, device, "One expert dominates")
    weights_grid(out / "moe_examples_distributed.png", df, spread, imgs, entries, moe, device, "Weights spread across experts")
    gate_acc = float((df[WCOLS].to_numpy().argmax(1) == df["label"].to_numpy()).mean())
    print("\n=== mean routing weights by true condition ===")
    print(tab.round(3).to_string())
    print("\nexpert activity:", {b: (v["inactive"], v["dominates_unrelated_inputs"]) for b, v in rep.items()})
    return {"weights_by_condition": {f"{t}/{l}": r.to_dict() for (t, l), r in tab.iterrows()},
            "activity": rep, "gate_argmax_accuracy": gate_acc,
            "mean_entropy_dominant_examples": float(entropy(df.loc[dominant, WCOLS].to_numpy()).mean()),
            "mean_entropy_distributed_examples": float(entropy(df.loc[spread, WCOLS].to_numpy()).mean())}
