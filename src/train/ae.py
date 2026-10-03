"""Train the denoising autoencoder (Task 1 universal model and Task 2 specialists).

`cond` selects which corruption conditions the model sees during training:
    all (clean+salt+blur+occ), corrupted (salt+blur+occ), salt, blur, occlusion

Final training:
    python -m src.train.ae --name t1 --cond all --epochs 40 --params outputs/t1/best_params.json
"""
import argparse
import json
import time
from pathlib import Path

import torch

from src.common import tracking
from src.common.losses import restoration_loss
from src.common.metrics import l1_per_image, psnr_per_image, ssim_per_image
from src.data.corruptions import CLASSES, corrupt_batch
from src.data.manifest import load_manifest, materialize
from src.data.pets import GpuBatcher, load_pets
from src.models.autoencoder import ConvAE

COND_IDS = {"all": [0, 1, 2, 3], "corrupted": [1, 2, 3], "salt": [1], "blur": [2], "occlusion": [3]}
DEFAULTS = dict(lr=1e-3, batch_size=64, base_ch=32, latent_dim=512, dropout=0.1, alpha=0.8, weight_decay=0.0,
                bottleneck="linear", latent_ch=16)


def load_data(cond, device, root="data"):
    pets = load_pets(root)
    val = materialize(pets["val"], load_manifest("val"), device, labels_keep=COND_IDS[cond])
    return pets, val


@torch.no_grad()
def evaluate(model, val, bs=256):
    model.eval()
    l1, ss, ps = [], [], []
    for i in range(0, len(val["corr"]), bs):
        pred = model(val["corr"][i:i + bs])
        t = val["clean"][i:i + bs]
        l1.append(l1_per_image(pred, t)), ss.append(ssim_per_image(pred, t)), ps.append(psnr_per_image(pred, t))
    l1, ss, ps = torch.cat(l1), torch.cat(ss), torch.cat(ps)
    out = {"val_l1": l1.mean().item(), "val_ssim": ss.mean().item(), "val_psnr": ps.mean().item()}
    out["val_objective"] = out["val_l1"] + (1 - out["val_ssim"])  # lower is better
    for c in sorted(set(val["labels"].tolist())):
        m = val["labels"] == c
        out[f"val_psnr_{CLASSES[c]}"] = ps[m].mean().item()
        out[f"val_ssim_{CLASSES[c]}"] = ss[m].mean().item()
    return out


def train_ae(cfg, pets, val, cond="all", epochs=20, device="cuda", trial=None, log=True, seed=42):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed)
    model = ConvAE(cfg["base_ch"], cfg["latent_dim"], cfg["dropout"], bottleneck=cfg["bottleneck"],
                   latent_ch=cfg["latent_ch"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    batcher = GpuBatcher(pets["train"], cfg["batch_size"], device, seed=seed)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg["lr"], total_steps=epochs * len(batcher))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ids = torch.tensor(COND_IDS[cond], device=device)
    best, best_state, history = float("inf"), None, []
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for clean in batcher:
            labels = ids[torch.randint(0, len(ids), (len(clean),), device=device)]
            corr, _ = corrupt_batch(clean, labels)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                pred = model(corr)
            loss = restoration_loss(pred, clean, cfg["alpha"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt), scaler.update(), sched.step()
            run += loss.item()
        metrics = {"train_loss": run / len(batcher), **evaluate(model, val), "epoch_time": time.time() - t0}
        history.append({"epoch": ep, **metrics})
        if log:
            tracking.log_metrics(metrics, step=ep)
        print(f"ep {ep:3d} loss {metrics['train_loss']:.4f} val_obj {metrics['val_objective']:.4f} "
              f"psnr {metrics['val_psnr']:.2f} ssim {metrics['val_ssim']:.4f} ({metrics['epoch_time']:.1f}s)", flush=True)
        if metrics["val_objective"] < best:
            best = metrics["val_objective"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            import optuna
            trial.report(metrics["val_objective"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, best_state, history, cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--cond", default="all", choices=list(COND_IDS))
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--params", default=None, help="json with the Optuna best params")
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(a.params)) if a.params else {}
    pets, val = load_data(a.cond, dev, a.data_root)
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    tracking.init("task1-2-autoencoders")
    with tracking.run(f"final-{a.name}", params={**DEFAULTS, **cfg, "cond": a.cond, "epochs": a.epochs}, tags={"stage": "final"}):
        best, state, hist, full_cfg = train_ae(cfg, pets, val, a.cond, a.epochs, dev)
        torch.save({"state_dict": state, "cfg": full_cfg, "cond": a.cond}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist))
        tracking.log_artifact(out / "history.json")
        tracking.log_metrics({"best_val_objective": best})
    print("saved", out / "best.pt", "best val objective", best)
