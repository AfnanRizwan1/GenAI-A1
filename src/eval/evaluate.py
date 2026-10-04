"""Test-set evaluation for Tasks 1-2 on the fixed test corruption manifest.

Compares, per corruption type and severity level:
  input     no restoration (the corrupted image itself)  -- the baseline every model must beat
  t1        Task 1 universal autoencoder
  t2_oracle Task 2 hard routing, expert chosen from the manifest's true label
  t2_pred   Task 2 hard routing, expert chosen by the classifier (clean -> identity bypass)

Also writes classifier metrics (accuracy, macro P/R/F1, per-class, normalised confusion matrix),
12+ example grids with absolute error maps, and failure-case grids.

    python -m src.eval.evaluate --t1 outputs/t1_v2/best.pt --cls outputs/t2_cls_v2/best.pt \
        --specs outputs/spec_salt_v2/best.pt outputs/spec_blur_v2/best.pt outputs/spec_occlusion_v2/best.pt
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
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support  # noqa: E402

from src.common.metrics import l1_per_image, psnr_per_image, ssim_per_image  # noqa: E402
from src.data.corruptions import CLASSES, LEVELS  # noqa: E402
from src.data.manifest import iter_manifest, load_manifest  # noqa: E402
from src.data.pets import load_pets  # noqa: E402
from src.models.loading import load_ae, load_classifier  # noqa: E402

PSNR_CAP = 60.0  # identical images have infinite PSNR; cap so means stay finite and comparable
COND_ORDER = [("clean", "-")] + [(t, l) for t in CLASSES[1:] for l in LEVELS]


class Systems:
    """The restoration systems under test. Any of t1 / (cls + specs) may be missing."""

    def __init__(self, t1=None, cls=None, specs=None):
        self.t1, self.cls, self.specs = t1, cls, specs  # specs: [salt, blur, occlusion]

    def route(self, x, route_labels):
        y = x.clone()  # label 0 (clean): identity bypass, expert never called
        for c in (1, 2, 3):
            m = route_labels == c
            if m.any():
                y[m] = self.specs[c - 1](x[m])
        return y

    @torch.no_grad()
    def run(self, corr, labels):
        out, pred = {"input": corr}, None
        if self.t1 is not None:
            out["t1"] = self.t1(corr)
        if self.cls is not None and self.specs is not None:
            pred = self.cls(corr).argmax(1)
            out["t2_oracle"] = self.route(corr, labels)
            out["t2_pred"] = self.route(corr, pred)
        return out, pred


def per_entry_metrics(systems, imgs, entries, bs, device):
    rows = []
    for corr, clean, labels, chunk in iter_manifest(imgs, entries, bs, device):
        outs, pred = systems.run(corr, labels)
        m = {n: (psnr_per_image(y, clean).clamp(max=PSNR_CAP).cpu(), ssim_per_image(y, clean).cpu(),
                 l1_per_image(y, clean).cpu()) for n, y in outs.items()}
        for i, e in enumerate(chunk):
            r = {"idx": e["idx"], "type": e["type"], "level": e["level"] or "-", "label": e["label"]}
            if pred is not None:
                r["pred"] = int(pred[i])
            for n, (p, s, l) in m.items():
                r[f"{n}_psnr"], r[f"{n}_ssim"], r[f"{n}_l1"] = float(p[i]), float(s[i]), float(l[i])
            rows.append(r)
    return pd.DataFrame(rows)


def summary_tables(df, methods):
    cols = [f"{m}_{k}" for m in methods for k in ("psnr", "ssim", "l1")]
    by_cond = df.groupby(["type", "level"])[cols].mean().reindex(COND_ORDER)
    by_type = df.groupby("type")[cols].mean().reindex(CLASSES)
    return by_cond, by_type, df[cols].mean()


def classifier_report(df):
    y, p = df["label"].to_numpy(), df["pred"].to_numpy()
    pr, rc, f1, sup = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], zero_division=0)
    macro = precision_recall_fscore_support(y, p, labels=[0, 1, 2, 3], average="macro", zero_division=0)[:3]
    by_cond = (df.assign(ok=(df["label"] == df["pred"]))
               .groupby(["type", "level"])["ok"].mean().reindex(COND_ORDER))
    return {
        "accuracy": float((y == p).mean()),
        "macro_precision": float(macro[0]), "macro_recall": float(macro[1]), "macro_f1": float(macro[2]),
        "per_class": {c: {"precision": float(pr[i]), "recall": float(rc[i]), "f1": float(f1[i]), "support": int(sup[i])}
                      for i, c in enumerate(CLASSES)},
        "confusion_normalized": confusion_matrix(y, p, labels=[0, 1, 2, 3], normalize="true").tolist(),
        "accuracy_by_condition": {f"{t}/{l}": float(v) for (t, l), v in by_cond.items()},
    }


def plot_confusion(cm, path):
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES, rotation=30), ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("predicted"), ax.set_ylabel("true")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i][j]:.2f}", ha="center", va="center", color="white" if cm[i][j] > 0.5 else "black", fontsize=8)
    fig.colorbar(im, fraction=0.046), fig.tight_layout(), fig.savefig(path, dpi=200), plt.close(fig)


def pick_examples(df, n_extra=3, seed=0):
    """12 representative rows: every (corruption, severity) once, one clean, and extra high-severity cases."""
    rng, rows = np.random.default_rng(seed), []
    for t in CLASSES[1:]:
        for l in LEVELS:
            rows.append(int(rng.choice(df.index[(df["type"] == t) & (df["level"] == l)])))
    rows.append(int(rng.choice(df.index[df["type"] == "clean"])))
    hi = df.index[(df["level"] == "high")]
    rows += [int(r) for r in rng.choice(hi, n_extra - 1, replace=False)]
    return rows


def pick_failures(df, method="t1"):
    """Worst PSNR gain over the input, one per corruption type, then the worst remaining."""
    d = df[df["type"] != "clean"].assign(gain=lambda x: x[f"{method}_psnr"] - x["input_psnr"])
    rows = [int(d[d["type"] == t]["gain"].idxmin()) for t in CLASSES[1:]]
    rest = d.drop(index=rows)["gain"].sort_values()
    return rows + [int(rest.index[0])]


def pick_misroutes(df, k=4):
    d = df[df["pred"] != df["label"]].assign(loss=lambda x: x["t2_oracle_psnr"] - x["t2_pred_psnr"])
    return [int(i) for i in d.sort_values("loss", ascending=False).index[:k]]


@torch.no_grad()
def grid(path, df, rows, imgs, entries, system_fn, method_name, device, title_fn=None):
    """Columns: clean target | corrupted input | restored | absolute error map."""
    sel = [entries[j] for j in rows]
    corr, clean, labels, _ = next(iter_manifest(imgs, sel, len(sel), device))
    restored = system_fn(corr, labels)
    err = (restored - clean).abs().mean(1).cpu().numpy()
    fig, ax = plt.subplots(len(rows), 4, figsize=(8.4, 2.1 * len(rows)))
    ax = np.atleast_2d(ax)
    show = lambda t: t.permute(1, 2, 0).clamp(0, 1).cpu().numpy()
    for i, j in enumerate(rows):
        r = df.loc[j]
        for k, (im, name) in enumerate([(show(clean[i]), "clean target"), (show(corr[i]), "input"),
                                        (show(restored[i]), method_name), (err[i], "|error|")]):
            ax[i, k].imshow(im, cmap="inferno" if k == 3 else None, vmin=0 if k == 3 else None, vmax=0.5 if k == 3 else None)
            ax[i, k].set_xticks([]), ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(name, fontsize=9)
        ax[i, 0].set_ylabel(title_fn(r) if title_fn else f"{r['type']}/{r['level']}", fontsize=7)
        ax[i, 2].set_xlabel(f"{r[f'{method_name}_psnr']:.1f} dB", fontsize=7)
        ax[i, 1].set_xlabel(f"{r['input_psnr']:.1f} dB", fontsize=7)
    fig.tight_layout(), fig.savefig(path, dpi=150), plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1"), ap.add_argument("--cls"), ap.add_argument("--specs", nargs=3)
    ap.add_argument("--out", default="outputs/eval")
    ap.add_argument("--data-root", default="data")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--max-entries", type=int, default=0, help="debug: only the first N manifest entries")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    systems = Systems(load_ae(a.t1, dev) if a.t1 else None,
                      load_classifier(a.cls, dev) if a.cls else None,
                      [load_ae(p, dev) for p in a.specs] if a.specs else None)
    imgs = load_pets(a.data_root)["test"]
    entries = load_manifest("test")
    if a.max_entries:
        entries = entries[:a.max_entries]
    df = per_entry_metrics(systems, imgs, entries, a.bs, dev)
    df.to_csv(out / "per_entry.csv", index=False)

    methods = [m for m in ("input", "t1", "t2_oracle", "t2_pred") if f"{m}_psnr" in df]
    by_cond, by_type, overall = summary_tables(df, methods)
    by_cond.to_csv(out / "by_condition.csv"), by_type.to_csv(out / "by_type.csv")
    summary = {"n_entries": len(df), "methods": methods, "overall": overall.to_dict(),
               "by_type": {t: r.to_dict() for t, r in by_type.iterrows()},
               "by_condition": {f"{t}/{l}": r.to_dict() for (t, l), r in by_cond.iterrows()}}

    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.float_format", "{:.3f}".format):
        for m in methods:
            print(f"\n=== {m}: mean PSNR / SSIM by condition ===")
            print(by_cond[[f"{m}_psnr", f"{m}_ssim", f"{m}_l1"]])

    fn = {}
    if systems.t1 is not None:
        fn["t1"] = lambda c, l: systems.t1(c)
    if "pred" in df:
        rep = classifier_report(df)
        summary["classifier"] = rep
        plot_confusion(rep["confusion_normalized"], out / "confusion_matrix.png")
        print(f"\nclassifier: acc {rep['accuracy']:.4f} macro-F1 {rep['macro_f1']:.4f}")
        fn["t2_oracle"] = lambda c, l: systems.route(c, l)
        fn["t2_pred"] = lambda c, l: systems.route(c, systems.cls(c).argmax(1))
        mis = pick_misroutes(df)
        summary["misrouted_total"] = int((df["pred"] != df["label"]).sum())
        if mis:
            grid(out / "failures_t2_misrouting.png", df, mis, imgs, entries, fn["t2_pred"], "t2_pred", dev,
                 lambda r: f"true {CLASSES[int(r['label'])]} -> pred {CLASSES[int(r['pred'])]}\n{r['level']}")
    ex = pick_examples(df)
    for name, f in fn.items():
        grid(out / f"examples_{name}.png", df, ex, imgs, entries, f, name, dev)
    if "t1" in fn:
        grid(out / "failures_t1.png", df, pick_failures(df, "t1"), imgs, entries, fn["t1"], "t1", dev)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
