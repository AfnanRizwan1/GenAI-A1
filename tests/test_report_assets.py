import importlib.util
import json
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location("make_report_assets", Path(__file__).resolve().parents[1] / "scripts" / "make_report_assets.py")
mra = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mra)

CONDS = mra.CONDS


def write_fake_eval(ev: Path):
    ev.mkdir(parents=True)
    rows = []
    for i, (t, l) in enumerate(CONDS):
        rows.append({"type": t, "level": l, "input_psnr": 20.0 - i, "t1_psnr": 18.0 + i * 0.5, "moe_psnr": 19.0,
                     "input_ssim": 0.5, "t1_ssim": 0.6, "moe_ssim": 0.55, "input_l1": 0.1, "t1_l1": 0.08, "moe_l1": 0.09})
    df = pd.DataFrame(rows).set_index(["type", "level"])
    df.to_csv(ev / "by_condition.csv")
    cls = {"accuracy": 0.975, "macro_precision": 0.97, "macro_recall": 0.96, "macro_f1": 0.965,
           "per_class": {c: {"precision": 0.9, "recall": 0.9, "f1": 0.9, "support": 100} for c in ("clean", "salt", "blur", "occlusion")},
           "confusion_normalized": [[1, 0, 0, 0]] * 4,
           "accuracy_by_condition": {f"{t}/{l}": 0.9 for t, l in CONDS}}
    moe = {"weights_by_condition": {f"{t}/{l}": {"w_identity": 0.1, "w_salt": 0.2, "w_blur": 0.6, "w_occlusion": 0.1} for t, l in CONDS},
           "activity": {b: {"mean_weight_all": 0.25, "mean_weight_on_own_type": 0.5, "mean_weight_on_other_types": 0.1, "fraction_argmax": 0.25,
                            "inactive": False, "dominates_unrelated_inputs": False} for b in ("identity", "salt", "blur", "occlusion")},
           "gate_argmax_accuracy": 0.9}
    (ev / "summary.json").write_text(json.dumps({"classifier": cls, "misrouted_total": 42, "moe": moe}))


def test_tables_are_generated_from_the_outputs_with_best_marked(tmp_path):
    outputs, out = tmp_path / "outputs", tmp_path / "report"
    write_fake_eval(outputs / "eval_v2")
    mra.restoration_tables(outputs / "eval_v2", out)
    mra.classifier_tables(outputs / "eval_v2", out, tmp_path)
    mra.moe_tables(outputs / "eval_v2", out)
    g = out / "generated"
    psnr = (g / "restoration_psnr.tex").read_text()
    assert r"\toprule" in psnr and "Universal AE" in psnr and "Soft MoE" in psnr
    clean_row = next(r for r in psnr.splitlines() if r.startswith("Clean"))
    assert r"\textbf{20.00}" in clean_row                       # no-restoration is the best on the first row (20.0 vs 18.0 / 19.0)
    low_psnr = next(r for r in psnr.splitlines() if r.startswith("Occlusion") and "high" in r)
    assert r"\textbf{" in low_psnr
    l1 = (g / "restoration_l1.tex").read_text()
    assert r"\textbf{0.0800}" in l1                              # for L1 the smallest value is the best
    assert (g / "classifier_summary.tex").read_text() == r"97.50\%"
    assert (g / "misrouted.tex").read_text() == "42"
    assert r"\textbf{0.600}" in (g / "moe_weights.tex").read_text()   # dominant branch of each row in bold
    assert "Macro average" in (g / "classifier_metrics.tex").read_text()


def test_missing_inputs_are_skipped_without_crashing(tmp_path, capsys):
    out = tmp_path / "report"
    mra.restoration_tables(tmp_path / "nope", out)
    mra.t4_tables(tmp_path / "nope", out, tmp_path)
    mra.onnx_table(tmp_path / "nope", out)
    mra.optuna_tables(tmp_path / "nope", out)
    assert "missing" in capsys.readouterr().out
    assert (out / "generated" / "optuna_summary.tex").exists()           # empty table, report still compiles


def test_latex_escaping():
    assert mra.esc("a_b & 50%") == r"a\_b \& 50\%"


def test_optuna_list_is_never_empty_so_latex_compiles(tmp_path):
    out = tmp_path / "report"
    mra.optuna_tables(tmp_path / "nothing", out)
    text = (out / "generated" / "optuna_best.tex").read_text()
    assert r"\item" in text
