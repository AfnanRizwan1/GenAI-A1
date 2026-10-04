"""Joint training of the soft mixture-of-experts (Task 3).

Stage 1 (warm-up): experts frozen (and in eval mode), only the gate is trained.
Stage 2 (joint):   everything unfrozen, fine-tuned with a smaller learning rate.

Loss = a*L1 + (1-a)*(1-SSIM) + gamma*CE(gate logits / T, true corruption) + delta*L_balance
with L_balance = sum_i (mean_batch(w_i) - 1/K)^2 over a class-balanced batch (K = 4 branches).

    python -m src.train.soft_moe --name t3 --cls outputs/t2_cls_v2/best.pt \
        --specs outputs/spec_salt_v2/best.pt outputs/spec_blur_v2/best.pt outputs/spec_occlusion_v2/best.pt \
        --warmup-epochs 3 --epochs 40 --params outputs/t3/best_params.json
"""
import argparse
import copy
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from src.common import tracking
from src.common.losses import ssim_fn
from src.common.metrics import l1_per_image, psnr_per_image, ssim_per_image
from src.data.corruptions import CLASSES, corrupt_batch
from src.data.pets import GpuBatcher
from src.models.loading import load_ae, load_classifier
from src.models.soft_moe import BRANCHES, SoftMoE
from src.train.classifier import balanced_labels, load_data

DEFAULTS = dict(temperature=1.0, lr_warm=1e-3, lr_joint=1e-4, gamma=0.1, delta=0.01, a=0.8,
                weight_decay=1e-4, batch_size=64)
K = len(BRANCHES)


def moe_loss(y, clean, logits, w, labels, cfg):
    l1 = F.l1_loss(y.float(), clean)
    ssim_term = 1 - ssim_fn(y, clean)
    ce = F.cross_entropy(logits.float() / cfg["temperature"], labels)
    balance = ((w.float().mean(0) - 1.0 / K) ** 2).sum()
    total = cfg["a"] * l1 + (1 - cfg["a"]) * ssim_term + cfg["gamma"] * ce + cfg["delta"] * balance
    return total, {"l1": l1.item(), "ssim_term": ssim_term.item(), "ce": ce.item(), "balance": balance.item()}


@torch.no_grad()
def evaluate(model, val, bs=256):
    model.eval()
    ys, ws = [], []
    l1, ss, ps = [], [], []
    for i in range(0, len(val["corr"]), bs):
        y, w, _ = model(val["corr"][i:i + bs])
        t = val["clean"][i:i + bs]
        l1.append(l1_per_image(y, t)), ss.append(ssim_per_image(y, t)), ps.append(psnr_per_image(y, t))
        ws.append(w.float())
    l1, ss, ps, w = torch.cat(l1), torch.cat(ss), torch.cat(ps), torch.cat(ws)
    out = {"val_l1": l1.mean().item(), "val_ssim": ss.mean().item(), "val_psnr": ps.mean().item(),
           "gate_acc": (w.argmax(1) == val["labels"]).float().mean().item(),
           "max_mean_weight": w.mean(0).max().item()}  # routing-collapse indicator
    out["val_objective"] = out["val_l1"] + (1 - out["val_ssim"])
    for c in range(K):
        m = val["labels"] == c
        if m.any():
            for j, b in enumerate(BRANCHES):
                out[f"w_{CLASSES[c]}_to_{b}"] = w[m][:, j].mean().item()
    return out


def train_moe(cfg, pets, val, gate, experts, warmup_epochs=3, epochs=30, device="cuda", trial=None, log=True, seed=42):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed)
    model = SoftMoE(copy.deepcopy(gate), [copy.deepcopy(e) for e in experts], cfg["temperature"]).to(device)
    bs = cfg["batch_size"] - cfg["batch_size"] % 4
    batcher = GpuBatcher(pets["train"], bs, device, seed=seed)
    use_amp = device == "cuda"
    best, best_state, history = float("inf"), None, []
    joint_epochs = max(epochs - warmup_epochs, 1)

    def make_phase(warm):
        model.set_experts_trainable(not warm)
        params = list(model.gate.parameters()) if warm else list(model.parameters())
        lr = cfg["lr_warm"] if warm else cfg["lr_joint"]
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg["weight_decay"])
        n_ep = warmup_epochs if warm else joint_epochs
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_ep * len(batcher), 1))
        return opt, sched, torch.amp.GradScaler(enabled=use_amp)

    for ep in range(warmup_epochs + joint_epochs):
        warm = ep < warmup_epochs
        if ep == 0 or ep == warmup_epochs:
            opt, sched, scaler = make_phase(warm)
        model.train()
        if warm:
            model.experts.eval()  # frozen experts keep their batch-norm statistics
        t0, run = time.time(), 0.0
        for clean in batcher:
            labels = balanced_labels(len(clean), device)
            corr, _ = corrupt_batch(clean, labels)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                y, w, logits = model(corr)
            loss, parts = moe_loss(y, clean, logits, w, labels, cfg)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt), scaler.update(), sched.step()
            run += loss.item()
        m = {"train_loss": run / len(batcher), "phase": 0 if warm else 1, **evaluate(model, val),
             "epoch_time": time.time() - t0}
        history.append({"epoch": ep, **m})
        if log:
            tracking.log_metrics(m, step=ep)
        print(f"ep {ep:3d} {'warm' if warm else 'joint'} loss {m['train_loss']:.4f} val_obj {m['val_objective']:.4f} "
              f"psnr {m['val_psnr']:.2f} ssim {m['val_ssim']:.4f} gate_acc {m['gate_acc']:.3f} "
              f"max_w {m['max_mean_weight']:.2f} ({m['epoch_time']:.1f}s)", flush=True)
        if m["val_objective"] < best:
            best = m["val_objective"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if trial is not None:
            import optuna
            if not warm and m["max_mean_weight"] > 0.9:  # one branch swallowed the routing: collapse
                raise optuna.TrialPruned()
            trial.report(m["val_objective"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, best_state, history, cfg


def load_components(cls_path, spec_paths, device):
    gate = load_classifier(cls_path, device)
    experts = [load_ae(p, device) for p in spec_paths]
    cfgs = (torch.load(cls_path, map_location="cpu", weights_only=False)["cfg"],
            [torch.load(p, map_location="cpu", weights_only=False)["cfg"] for p in spec_paths])
    return gate, experts, cfgs


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t3")
    ap.add_argument("--cls", required=True)
    ap.add_argument("--specs", nargs=3, required=True)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--params", default=None)
    ap.add_argument("--data-root", default="data")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(a.params)) if a.params else {}
    pets, val = load_data(dev, a.data_root)
    gate, experts, (gate_cfg, expert_cfgs) = load_components(a.cls, a.specs, dev)
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    tracking.init("task3-soft-moe")
    with tracking.run(f"final-{a.name}", params={**DEFAULTS, **cfg, "warmup_epochs": a.warmup_epochs, "epochs": a.epochs},
                      tags={"stage": "final"}):
        best, state, hist, full = train_moe(cfg, pets, val, gate, experts, a.warmup_epochs, a.epochs, dev)
        torch.save({"state_dict": state, "cfg": full, "gate_cfg": gate_cfg, "expert_cfgs": expert_cfgs}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist))
        tracking.log_artifact(out / "history.json")
        tracking.log_metrics({"best_val_objective": best})
    print("saved", out / "best.pt", "best val objective", best)
