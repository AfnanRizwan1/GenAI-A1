"""Export the trained models to ONNX and verify them against PyTorch.

Models written to --out (default models/):
    universal_restoration.onnx   Task 1  image -> restored image
    classifier.onnx              Task 2  image -> class probabilities [clean, salt, blur, occlusion]
    specialist_salt.onnx / specialist_blur.onnx / specialist_occlusion.onnx   Task 2  image -> restored image
    soft_moe.onnx                Task 3  image -> (restored image, routing weights [identity, salt, blur, occlusion])
    sketch_generator.onnx        Task 4  (photo in [-1,1], style index int64) -> sketch in [-1,1]

Images are NCHW float32 128x128 with a dynamic batch axis; Tasks 1-3 take [0,1], Task 4 takes [-1,1].
Verification runs every ONNX model with ONNX Runtime on several inputs (including two batch sizes) and
records the max / mean absolute difference to the PyTorch output in <out>/onnx_verification.json.

    python -m src.export.export_onnx --t1 outputs/t1_v2/best.pt --cls outputs/t2_cls_v2/best.pt \
        --specs outputs/spec_salt_v2/best.pt outputs/spec_blur_v2/best.pt outputs/spec_occlusion_v2/best.pt \
        --moe outputs/t3_v2/best.pt --gen outputs/t4/best.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.loading import load_ae, load_classifier, load_generator, load_soft_moe
from src.models.soft_moe import SoftMoEExport

TOLERANCE = 1e-4   # max absolute difference allowed between PyTorch and ONNX Runtime outputs (fp32)
OPSET = 17


class ClassifierProbs(nn.Module):
    """Classifier + softmax, so the exported model returns probabilities."""

    def __init__(self, cls):
        super().__init__()
        self.cls = cls

    def forward(self, x):
        return torch.softmax(self.cls(x), dim=1)


def export_one(model, example_inputs, path, input_names, output_names):
    model.eval()
    axes = {n: {0: "batch"} for n in input_names + output_names}
    kwargs = dict(input_names=input_names, output_names=output_names, dynamic_axes=axes,
                  opset_version=OPSET, do_constant_folding=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:   # newer torch defaults to the dynamo exporter; the classic TorchScript exporter is the stable choice here
        torch.onnx.export(model, example_inputs, str(path), dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, example_inputs, str(path), **kwargs)
    return path


def _to_tuple(o):
    return o if isinstance(o, (tuple, list)) else (o,)


@torch.no_grad()
def verify_one(model, onnx_path, make_inputs, input_names, batches=(1, 4), seeds=(0, 1)):
    """Compare PyTorch and ONNX Runtime outputs; returns worst-case max/mean abs difference."""
    import onnxruntime as ort
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst_max, worst_mean = 0.0, 0.0
    for b in batches:
        for s in seeds:
            inputs = make_inputs(b, s)
            ref = _to_tuple(model(*inputs))
            feed = {n: t.numpy() for n, t in zip(input_names, inputs)}
            got = sess.run(None, feed)
            assert len(ref) == len(got), "output count differs"
            for r, g in zip(ref, got):
                d = np.abs(r.numpy() - g)
                worst_max, worst_mean = max(worst_max, float(d.max())), max(worst_mean, float(d.mean()))
    return {"max_abs_diff": worst_max, "mean_abs_diff": worst_mean, "tolerance": TOLERANCE,
            "passed": worst_max < TOLERANCE, "batch_sizes_tested": list(batches)}


def image_inputs(lo=0.0, hi=1.0):
    """Smooth-plus-noise test images in [lo, hi] (closer to real images than pure noise)."""
    def make(b, seed):
        g = torch.Generator().manual_seed(seed)
        x = torch.rand(b, 3, 16, 16, generator=g)
        x = torch.nn.functional.interpolate(x, size=128, mode="bilinear", align_corners=False)
        x = (0.8 * x + 0.2 * torch.rand(b, 3, 128, 128, generator=g)).clamp(0, 1)
        return (lo + (hi - lo) * x,)
    return make


def sketch_inputs(b, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(b, 3, 128, 128, generator=g) * 2 - 1
    return x, torch.randint(0, 3, (b,), generator=g)


def run(args):
    out = Path(args.out)
    jobs = []   # (name, filename, model, input_names, output_names, make_inputs)
    if args.t1:
        jobs.append(("task1_universal", "universal_restoration.onnx", load_ae(args.t1), ["image"], ["restored"], image_inputs()))
    if args.cls:
        jobs.append(("task2_classifier", "classifier.onnx", ClassifierProbs(load_classifier(args.cls)).eval(),
                     ["image"], ["probabilities"], image_inputs()))
    if args.specs:
        for name, p in zip(("salt", "blur", "occlusion"), args.specs):
            jobs.append((f"task2_specialist_{name}", f"specialist_{name}.onnx", load_ae(p), ["image"], ["restored"], image_inputs()))
    if args.moe:
        jobs.append(("task3_soft_moe", "soft_moe.onnx", SoftMoEExport(load_soft_moe(args.moe)).eval(),
                     ["image"], ["restored", "weights"], image_inputs()))
    if args.gen:
        jobs.append(("task4_sketch_generator", "sketch_generator.onnx", load_generator(args.gen),
                     ["photo", "style"], ["sketch"], sketch_inputs))
    report = {}
    for name, fname, model, in_names, out_names, make in jobs:
        example = make(1, 0)
        path = export_one(model, example, out / fname, in_names, out_names)
        res = verify_one(model, path, make, in_names)
        res["file"], res["size_mb"] = fname, round(path.stat().st_size / 1e6, 2)
        report[name] = res
        print(f"{name:26s} {fname:28s} {res['size_mb']:7.2f} MB  max|diff| {res['max_abs_diff']:.2e}  "
              f"mean|diff| {res['mean_abs_diff']:.2e}  {'OK' if res['passed'] else 'FAIL'}")
    (out / "onnx_verification.json").write_text(json.dumps(report, indent=2))
    failed = [n for n, r in report.items() if not r["passed"]]
    if failed:
        raise SystemExit(f"ONNX output differs from PyTorch beyond {TOLERANCE}: {failed}")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1"), ap.add_argument("--cls"), ap.add_argument("--specs", nargs=3)
    ap.add_argument("--moe"), ap.add_argument("--gen")
    ap.add_argument("--out", default="models")
    run(ap.parse_args())
