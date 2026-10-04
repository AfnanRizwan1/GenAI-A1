"""Train the style-conditioned face-to-sketch cGAN (Task 4).

G loss = BCE(D(x, s, G(x,s)), 1) + lambda * L1(G(x,s), y);  D loss = 0.5 * (BCE real + BCE fake).
D-real, D-fake, G-adversarial, G-reconstruction and validation metrics are logged separately every epoch,
and the same fixed validation photos are rendered every `sample_every` epochs.

    python -m src.train.cgan --name t4 --epochs 150 --params outputs/t4/best_params.json
"""
import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from src.common import tracking  # noqa: E402
from src.common.metrics import l1_per_image, psnr_per_image, ssim_per_image  # noqa: E402
from src.data.fs2k import PairBatcher, paired_augment, to_tensor  # noqa: E402
from src.models.cgan import PatchDiscriminator, UNetGenerator  # noqa: E402

DEFAULTS = dict(lr_g=2e-4, lr_d=2e-4, batch_size=16, base_ch=32, dropout=0.3, emb_dim=16, lam=100.0)
bce = F.binary_cross_entropy_with_logits


def _unit(x):
    return (x + 1) / 2  # [-1,1] -> [0,1] for metrics


@torch.no_grad()
def evaluate(G, val, device, bs=64):
    """val: dict(photo, sketch, style) uint8/ints; metrics on [0,1] images with the ground-truth style."""
    G.eval()
    l1, ss, ps = [], [], []
    for i in range(0, len(val["photo"]), bs):
        x = to_tensor(val["photo"][i:i + bs], device)
        y = to_tensor(val["sketch"][i:i + bs], device)
        st = torch.from_numpy(val["style"][i:i + bs]).long().to(device)
        g = G(x, st)
        l1.append(l1_per_image(_unit(g), _unit(y))), ss.append(ssim_per_image(_unit(g), _unit(y)))
        ps.append(psnr_per_image(_unit(g), _unit(y)))
    l1, ss, ps = torch.cat(l1), torch.cat(ss), torch.cat(ps)
    out = {"val_l1": l1.mean().item(), "val_ssim": ss.mean().item(), "val_psnr": ps.mean().item()}
    out["val_objective"] = out["val_l1"] + (1 - out["val_ssim"])
    return out


@torch.no_grad()
def save_samples(G, val, device, path, n=8):
    """Fixed validation photos (always the first n) -> photo | generated | target, one row each."""
    G.eval()
    x = to_tensor(val["photo"][:n], device)
    y = to_tensor(val["sketch"][:n], device)
    st = torch.from_numpy(val["style"][:n]).long().to(device)
    g = G(x, st)
    fig, ax = plt.subplots(len(x), 3, figsize=(5.4, 1.8 * len(x)))
    ax = np.atleast_2d(ax)
    for i in range(len(x)):
        for k, (im, name) in enumerate([(x[i], "photo"), (g[i], "generated"), (y[i], "target")]):
            ax[i, k].imshow(_unit(im).permute(1, 2, 0).clamp(0, 1).cpu().numpy())
            ax[i, k].set_xticks([]), ax[i, k].set_yticks([])
            if i == 0:
                ax[i, k].set_title(name, fontsize=9)
        ax[i, 0].set_ylabel(f"style {int(st[i]) + 1}", fontsize=8)
    fig.tight_layout(), fig.savefig(path, dpi=120), plt.close(fig)


def train_cgan(cfg, data, epochs=20, device="cuda", trial=None, log=True, seed=42, out_dir=None, sample_every=10):
    cfg = {**DEFAULTS, **cfg}
    torch.manual_seed(seed)
    G = UNetGenerator(cfg["base_ch"], cfg["emb_dim"], cfg["dropout"]).to(device)
    D = PatchDiscriminator(cfg["base_ch"], cfg["emb_dim"]).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg["lr_g"], betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg["lr_d"], betas=(0.5, 0.999))
    tr = data["train"]
    batcher = PairBatcher(tr["photo"], tr["sketch"], tr["style"], cfg["batch_size"], device, seed=seed)
    gen = torch.Generator(device=device).manual_seed(seed)
    best, best_state, history = float("inf"), None, []
    for ep in range(epochs):
        G.train(), D.train()
        t0, acc = time.time(), np.zeros(4)
        for photo, sketch, style in batcher:
            photo, sketch = paired_augment(photo, sketch, gen)  # identical warp for both images of a pair
            fake = G(photo, style)
            # discriminator
            d_real = bce(D(photo, sketch, style), torch.ones_like(D(photo, sketch, style)))
            out_f = D(photo, fake.detach(), style)
            d_fake = bce(out_f, torch.zeros_like(out_f))
            opt_d.zero_grad(set_to_none=True)
            (0.5 * (d_real + d_fake)).backward()
            opt_d.step()
            # generator
            out_g = D(photo, fake, style)
            g_adv = bce(out_g, torch.ones_like(out_g))
            g_rec = F.l1_loss(fake, sketch)
            opt_g.zero_grad(set_to_none=True)
            (g_adv + cfg["lam"] * g_rec).backward()
            opt_g.step()
            acc += np.array([d_real.item(), d_fake.item(), g_adv.item(), g_rec.item()])
        acc /= len(batcher)
        m = {"d_real": acc[0], "d_fake": acc[1], "g_adv": acc[2], "g_rec": acc[3], **evaluate(G, data["val"], device),
             "epoch_time": time.time() - t0}
        history.append({"epoch": ep, **m})
        if log:
            tracking.log_metrics(m, step=ep)
        print(f"ep {ep:3d} D_real {m['d_real']:.3f} D_fake {m['d_fake']:.3f} G_adv {m['g_adv']:.3f} G_rec {m['g_rec']:.4f} "
              f"| val L1 {m['val_l1']:.4f} SSIM {m['val_ssim']:.4f} PSNR {m['val_psnr']:.2f} ({m['epoch_time']:.1f}s)", flush=True)
        if out_dir is not None and (ep % sample_every == 0 or ep == epochs - 1):
            p = Path(out_dir) / "samples" / f"epoch_{ep:03d}.png"
            p.parent.mkdir(parents=True, exist_ok=True)
            save_samples(G, data["val"], device, p)
            if log:
                tracking.log_artifact(p)
        if m["val_objective"] < best:
            best = m["val_objective"]
            best_state = {k: v.detach().cpu().clone() for k, v in G.state_dict().items()}
        if trial is not None:
            import optuna
            trial.report(m["val_objective"], ep)
            if trial.should_prune():
                raise optuna.TrialPruned()
    return best, best_state, history, cfg


def load_data(root, device=None):
    from src.data.fs2k import load_fs2k
    return load_fs2k(root)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="t4")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--params", default=None)
    ap.add_argument("--sample-every", type=int, default=10)
    ap.add_argument("--data-root", default="data/fs2k")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = json.load(open(a.params)) if a.params else {}
    data = load_data(a.data_root)
    out = Path("outputs") / a.name
    out.mkdir(parents=True, exist_ok=True)
    tracking.init("task4-cgan")
    with tracking.run(f"final-{a.name}", params={**DEFAULTS, **cfg, "epochs": a.epochs}, tags={"stage": "final"}):
        best, state, hist, full = train_cgan(cfg, data, a.epochs, dev, out_dir=out, sample_every=a.sample_every)
        torch.save({"state_dict": state, "cfg": full}, out / "best.pt")
        (out / "history.json").write_text(json.dumps(hist))
        tracking.log_artifact(out / "history.json")
        tracking.log_metrics({"best_val_objective": best})
    print("saved", out / "best.pt", "best val objective", best)
