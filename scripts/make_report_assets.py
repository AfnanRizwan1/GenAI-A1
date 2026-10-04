"""Turn the experiment outputs into the LaTeX tables and figures used by report/main.tex.

Nothing is typed in by hand: every number in the report's tables comes from the JSON / CSV files written by the
evaluation, Optuna and training scripts.

    python scripts/make_report_assets.py --outputs outputs --models models --out report

Reads  : <outputs>/eval_v2/{summary.json,by_condition.csv,*.png}, <outputs>/eval_t4/{summary.json,*.png},
         <outputs>/<study>/{trials.csv,best_params.json,history.json}, <models>/onnx_verification.json
Writes : <out>/generated/*.tex  (tables, \\input by main.tex)   <out>/figures/*.png|pdf
Missing inputs are skipped with a warning, so the report still compiles (the tables then show "pending").
"""
import argparse
import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CONDS = [("clean", "-")] + [(t, l) for t in ("salt", "blur", "occlusion") for l in ("low", "medium", "high")]
COND_LABEL = {"clean": "Clean", "salt": "Salt-and-pepper", "blur": "Gaussian blur", "occlusion": "Occlusion"}
METHOD_LABEL = {"input": "No restoration", "t1": "Universal AE", "t2_oracle": "Hard (oracle)", "t2_pred": "Hard (predicted)", "moe": "Soft MoE"}
STUDIES = {"t1_v2": "Task 1 autoencoder", "t2_cls_v2": "Task 2 classifier", "spec_shared_v2": "Task 2 specialists (shared)",
           "t3_v2": "Task 3 soft MoE", "t4": "Task 4 cGAN"}


def esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def warn(msg):
    print("warning:", msg)


def write_tex(out, name, text):
    (out / "generated").mkdir(parents=True, exist_ok=True)
    (out / "generated" / name).write_text(text, encoding="utf-8", newline="\n")


def tabular(header, rows, spec=None, bold_best=None):
    """booktabs tabular; bold_best: list of (col index) -> 'max' | 'min' applied on numeric strings of each row."""
    spec = spec or "l" + "r" * (len(header) - 1)
    lines = [rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    for r in rows:
        lines.append(" & ".join(r) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def best_marks(values, mode):
    """Indices of the best numeric entries (ties included)."""
    v = np.array([np.nan if x is None else x for x in values], dtype=float)
    if np.all(np.isnan(v)):
        return set()
    best = np.nanmax(v) if mode == "max" else np.nanmin(v)
    return {i for i, x in enumerate(v) if not np.isnan(x) and abs(x - best) < 1e-9}


# --------------------------------------------------------------------------- Tasks 1-3 evaluation tables
def restoration_tables(ev, out):
    csv = ev / "by_condition.csv"
    if not csv.exists():
        return warn(f"{csv} missing")
    df = pd.read_csv(csv, index_col=[0, 1])
    methods = [m for m in METHOD_LABEL if f"{m}_psnr" in df.columns]
    for metric, fmt, name in (("psnr", "{:.2f}", "psnr"), ("ssim", "{:.3f}", "ssim"), ("l1", "{:.4f}", "l1")):
        rows = []
        for t, l in CONDS:
            if (t, str(l)) not in df.index:
                continue
            vals = [df.loc[(t, str(l)), f"{m}_{metric}"] for m in methods]
            best = best_marks(vals, "min" if metric == "l1" else "max")
            cells = [(r"\textbf{" + fmt.format(v) + "}") if i in best else fmt.format(v) for i, v in enumerate(vals)]
            rows.append([COND_LABEL[t], "--" if l == "-" else str(l)] + cells)
        header = ["Corruption", "Level"] + [esc(METHOD_LABEL[m]) for m in methods]
        write_tex(out, f"restoration_{name}.tex", tabular(header, rows, "ll" + "r" * len(methods)))
    # per corruption type (mean over severities)
    t = pd.read_csv(ev / "by_type.csv", index_col=0) if (ev / "by_type.csv").exists() else None
    if t is not None:
        rows = [[COND_LABEL[i]] + [f"{t.loc[i, f'{m}_psnr']:.2f} / {t.loc[i, f'{m}_ssim']:.3f}" for m in methods] for i in t.index if i in COND_LABEL]
        write_tex(out, "restoration_by_type.tex", tabular(["Corruption"] + [esc(METHOD_LABEL[m]) for m in methods], rows, "l" + "r" * len(methods)))


def classifier_tables(ev, out, figs):
    p = ev / "summary.json"
    if not p.exists():
        return warn(f"{p} missing")
    s = json.loads(p.read_text())
    c = s.get("classifier")
    if not c:
        return warn("no classifier block in summary.json")
    rows = [[COND_LABEL.get(k, k).replace("salt", "Salt-and-pepper"), f"{v['precision']:.3f}", f"{v['recall']:.3f}", f"{v['f1']:.3f}", str(v["support"])]
            for k, v in c["per_class"].items()]
    rows.append([r"\textbf{Macro average}", f"{c['macro_precision']:.3f}", f"{c['macro_recall']:.3f}", f"{c['macro_f1']:.3f}", str(sum(v["support"] for v in c["per_class"].values()))])
    write_tex(out, "classifier_metrics.tex", tabular(["Class", "Precision", "Recall", "F1", "Support"], rows))
    write_tex(out, "classifier_summary.tex", f"{c['accuracy'] * 100:.2f}\\%")
    acc = c["accuracy_by_condition"]
    rows = [[COND_LABEL[t], "--" if l == "-" else l, f"{acc[f'{t}/{l}'] * 100:.1f}"] for t, l in CONDS if f"{t}/{l}" in acc]
    write_tex(out, "classifier_by_condition.tex", tabular(["Corruption", "Level", "Accuracy (\\%)"], rows, "llr"))
    write_tex(out, "misrouted.tex", str(s.get("misrouted_total", "n/a")))
    for f in ("confusion_matrix.png", "examples_t1.png", "failures_t1.png", "examples_t2_pred.png", "failures_t2_misrouting.png",
              "examples_moe.png", "moe_routing_heatmap.png", "moe_examples_dominant.png", "moe_examples_distributed.png"):
        if (ev / f).exists():
            shutil.copy(ev / f, figs / f)


def moe_tables(ev, out):
    p = ev / "summary.json"
    if not p.exists():
        return
    m = json.loads(p.read_text()).get("moe")
    if not m:
        return warn("no moe block in summary.json")
    cols = ["w_identity", "w_salt", "w_blur", "w_occlusion"]
    rows = []
    for t, l in CONDS:
        r = m["weights_by_condition"].get(f"{t}/{l}")
        if r:
            top = int(np.argmax([r[c] for c in cols]))
            rows.append([COND_LABEL[t], "--" if l == "-" else l] + [(r"\textbf{" + f"{r[c]:.3f}" + "}") if i == top else f"{r[c]:.3f}" for i, c in enumerate(cols)])
    write_tex(out, "moe_weights.tex", tabular(["True corruption", "Level", "Identity", "Salt", "Blur", "Occl."], rows, "ll" + "r" * 4))
    act = m["activity"]
    rows = [[esc(b), f"{v['mean_weight_all']:.3f}", f"{v['mean_weight_on_own_type']:.3f}", f"{v['mean_weight_on_other_types']:.3f}", f"{v['fraction_argmax'] * 100:.1f}",
             "yes" if v["inactive"] else "no", "yes" if v["dominates_unrelated_inputs"] else "no"] for b, v in act.items()]
    write_tex(out, "moe_activity.tex", tabular(["Branch", "Mean w", "Own", "Other", "Argmax (\\%)", "Inactive", "Dominates"], rows, "l" + "r" * 6))
    write_tex(out, "moe_gate_acc.tex", f"{m['gate_argmax_accuracy'] * 100:.2f}\\%")


# --------------------------------------------------------------------------- Task 4
def t4_tables(ev, out, figs):
    p = ev / "summary.json"
    if not p.exists():
        return warn(f"{p} missing")
    s = json.loads(p.read_text())
    rows = []
    for name, v in s["by_style"].items():
        rows.append([name, str(s["images_per_style"][name])] + [f"{v[k]:.4f}" if "l1" in k or "ssim" in k else f"{v[k]:.2f}" for k in ("gen_l1", "gen_ssim", "gen_psnr", "gray_l1", "gray_ssim", "gray_psnr")])
    o = s["overall"]
    rows.append([r"\textbf{All}", str(s["n_test_images"])] + [f"{o[k]:.4f}" if "l1" in k or "ssim" in k else f"{o[k]:.2f}" for k in ("gen_l1", "gen_ssim", "gen_psnr", "gray_l1", "gray_ssim", "gray_psnr")])
    write_tex(out, "t4_metrics.tex", tabular(["Style", "N", "L1", "SSIM", "PSNR", "L1$_{g}$", "SSIM$_{g}$", "PSNR$_{g}$"], rows, "lr" + "r" * 6))
    m = np.array(s["style_conditioning"]["mean_l1_matrix_true_vs_fed"], dtype=float)
    rows = [[f"True Style {i + 1}"] + [(r"\textbf{" + f"{m[i, j]:.4f}" + "}") if j == int(np.nanargmin(m[i])) else f"{m[i, j]:.4f}" for j in range(3)] for i in range(3) if not np.isnan(m[i]).any()]
    write_tex(out, "t4_conditioning.tex", tabular(["", "Fed Style 1", "Fed Style 2", "Fed Style 3"], rows, "lrrr"))
    for f in ("examples_test.png", "failures_test.png", "style_conditioning_matrix.png"):
        if (ev / f).exists():
            shutil.copy(ev / f, figs / f"t4_{f}")


# --------------------------------------------------------------------------- Optuna and training curves
def optuna_tables(outputs, out):
    rows, detail = [], []
    for key, label in STUDIES.items():
        csv = outputs / key / "trials.csv"
        if not csv.exists():
            warn(f"{csv} missing")
            continue
        df = pd.read_csv(csv)
        state = df["state"].value_counts().to_dict()
        best = df[df["state"] == "COMPLETE"].sort_values("value", ascending=("cls" not in key))
        bv = df[df["state"] == "COMPLETE"]["value"]
        bval = (bv.max() if "cls" in key else bv.min()) if len(bv) else float("nan")
        rows.append([esc(label), str(len(df)), str(state.get("COMPLETE", 0)), str(state.get("PRUNED", 0)), f"{bval:.4f}"])
        bp = outputs / key / "best_params.json"
        if bp.exists():
            params = json.loads(bp.read_text())
            detail.append([esc(label), ", ".join(f"{esc(k)}={v:.4g}" if isinstance(v, float) else f"{esc(k)}={esc(v)}" for k, v in params.items())])
    write_tex(out, "optuna_summary.tex", tabular(["Study", "Trials", "Complete", "Pruned", "Best value"], rows, "lrrrr"))
    items = "\n".join(rf"\item \textbf{{{n}}}: {d}" for n, d in detail) or r"\item No Optuna results found yet."
    write_tex(out, "optuna_best.tex", items + "\n")  # never empty: an empty itemize is a LaTeX error


def curves(outputs, figs):
    def hist(name):
        p = outputs / name / "history.json"
        return pd.DataFrame(json.loads(p.read_text())) if p.exists() else None

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    for name, label in (("t1_v2", "Universal AE"), ("spec_salt_v2", "Salt specialist"), ("spec_blur_v2", "Blur specialist"), ("spec_occlusion_v2", "Occl. specialist")):
        h = hist(name)
        if h is not None:
            ax[0].plot(h["epoch"], h["val_psnr"], label=label)
            ax[1].plot(h["epoch"], h["val_ssim"], label=label)
    ax[0].set(xlabel="epoch", ylabel="validation PSNR (dB)"), ax[1].set(xlabel="epoch", ylabel="validation SSIM")
    h = hist("t2_cls_v2")
    if h is not None:
        ax[2].plot(h["epoch"], h["val_f1"], label="macro-F1"), ax[2].plot(h["epoch"], h["val_acc"], label="accuracy")
        ax[2].set(xlabel="epoch", ylabel="validation score", title="Classifier")
    for a in ax:
        a.grid(alpha=0.3), a.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(figs / "curves_t1_t2.pdf"), plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(10.5, 2.8))
    h = hist("t3_v2")
    if h is not None:
        ax[0].plot(h["epoch"], h["val_psnr"], label="val PSNR"), ax[0].set(xlabel="epoch", ylabel="validation PSNR (dB)")
        ax[1].plot(h["epoch"], h["gate_acc"], label="gate accuracy"), ax[1].plot(h["epoch"], h["max_mean_weight"], label="max mean weight")
        ax[1].set(xlabel="epoch", ylabel="routing")
        w = next((i for i, p in enumerate(h["phase"]) if p == 1), None)
        if w is not None:
            for a in ax[:2]:
                a.axvline(w - 0.5, color="gray", ls="--", lw=0.8)
    h4 = hist("t4")
    if h4 is not None:
        for k, lab in (("d_real", "D real"), ("d_fake", "D fake"), ("g_adv", "G adversarial")):
            ax[2].plot(h4["epoch"], h4[k], label=lab)
        ax[2].plot(h4["epoch"], h4["g_rec"], label="G reconstruction (L1)"), ax[2].set(xlabel="epoch", ylabel="training loss", title="Task 4")
    for a in ax:
        a.grid(alpha=0.3), a.legend(fontsize=7)
    fig.tight_layout(), fig.savefig(figs / "curves_t3_t4.pdf"), plt.close(fig)
    if h4 is not None:
        fig, ax = plt.subplots(figsize=(3.4, 2.6))
        ax.plot(h4["epoch"], h4["val_l1"], label="val L1"), ax.plot(h4["epoch"], h4["val_ssim"], label="val SSIM")
        ax.set(xlabel="epoch"), ax.grid(alpha=0.3), ax.legend(fontsize=7)
        fig.tight_layout(), fig.savefig(figs / "t4_validation.pdf"), plt.close(fig)


def onnx_table(models, out):
    p = models / "onnx_verification.json"
    if not p.exists():
        return warn(f"{p} missing")
    d = json.loads(p.read_text())
    rows = [[esc(k.replace("_", " ")), esc(v["file"]), f"{v['size_mb']:.1f}", f"{v['max_abs_diff']:.1e}", f"{v['mean_abs_diff']:.1e}"] for k, v in d.items()]
    write_tex(out, "onnx.tex", tabular(["Model", "File", "MB", "Max $|\\Delta|$", "Mean $|\\Delta|$"], rows, "llrrr"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outputs", default="outputs")
    ap.add_argument("--models", default="models")
    ap.add_argument("--out", default="report")
    a = ap.parse_args()
    outputs, models, out = Path(a.outputs), Path(a.models), Path(a.out)
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    (out / "generated").mkdir(parents=True, exist_ok=True)
    restoration_tables(outputs / "eval_v2", out)
    classifier_tables(outputs / "eval_v2", out, figs)
    moe_tables(outputs / "eval_v2", out)
    t4_tables(outputs / "eval_t4", out, figs)
    optuna_tables(outputs, out)
    curves(outputs, figs)
    onnx_table(models, out)
    print("generated:", ", ".join(sorted(p.name for p in (out / "generated").glob("*.tex"))))


if __name__ == "__main__":
    main()
