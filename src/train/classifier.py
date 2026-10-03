"""Train the 4-class corruption classifier (Task 2).

Training batches are exactly balanced (B/4 samples of every class); labels come
from the runtime corruption pipeline. Loss: multiclass cross-entropy.

    python -m src.train.classifier --name t2_cls --epochs 30 --params outputs/t2_cls/best_params.json
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from src.common import tracking
from src.data.corruptions import corrupt_batch
from src.data.manifest import load_manifest, materialize
from src.data.pets import GpuBatcher, load_pets
from src.models.classifier import CorruptionClassifier

DEFAULTS = dict(lr=1e-3, batch_size=64, channels="medium", dropout=0.3, weight_decay=1e-4)


def balanced_labels(b, device):
    """Exactly balanced, shuffled labels for a batch (b is rounded down to a multiple of 4 by the caller)."""
    return torch.randperm(b, device=device) % 4


@torch.no_grad()
def evaluate(model, val, bs=512):
    model.eval()
    logits = torch.cat([model(val["corr"][i:i + bs]) for i in range(0, len(val["corr"]), bs)])
    y, pred = val["labels"].cpu().numpy(), logits.argmax(1).cpu().numpy()
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"val_loss": F.cross_entropy(logits, val["labels"]).item(), "val_acc": float((y == pred).mean()),
            "val_precision": p, "val_recall": r, "val_f1": f}


def train_cls(cfg, pets, val, epochs=20, device="cuda", trial=None, log=True, seed=42):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed)
    bs = cfg["batch_size"] - cfg["batch_size"] % 4
    model = CorruptionClassifier(cfg["channels"], cfg["dropout"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batcher = GpuBatcher(pets["train"], bs, device, seed=seed)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=epochs * len(batcher))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    best, best_state, history = -1.0, None, []
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for clean in batcher:
            corr, labels = corrupt_batch(clean, balanced_labels(len(clean), device))
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(corr).float(), labels)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt), scaler.update(), sched.step()
            run += loss.item()
        m = {"train_loss": run / len(batcher), **evaluate(model, val), "epoch_time": time.time() - t0}
        history.append({"epoch": ep, **m})
        if log:
            tracking.log_metrics(m, step=ep)
        print(f"ep {ep:3d} loss {m['train_loss']:.4f} val_acc {m['val_acc']:.4f} val_f1 {m['val_f1']:.4f} ({m['epoch_time']:.1f}s)", flush=True)
        if m["val_f1"] > best:
            best = m["val_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            import optuna
            trial.report(m["val_f1"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, best_state, history, cfg


def load_data(device, root="data"):
    pets = load_pets(root)
    return pets, materialize(pets["val"], load_manifest("val"), device)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t2_cls")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--params", default=None)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(a.params)) if a.params else {}
    pets, val = load_data(dev, a.data_root)
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    tracking.init("task2-classifier")
    with tracking.run(f"final-{a.name}", params={**DEFAULTS, **cfg, "epochs": a.epochs}, tags={"stage": "final"}):
        best, state, hist, full = train_cls(cfg, pets, val, a.epochs, dev)
        torch.save({"state_dict": state, "cfg": full}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist))
        tracking.log_artifact(out / "history.json")
        tracking.log_metrics({"best_val_f1": best})
    print("saved", out / "best.pt", "best val macro-F1", best)
