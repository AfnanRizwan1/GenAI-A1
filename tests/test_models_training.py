"""Model shape/behaviour and tiny end-to-end training checks (CPU, synthetic structured images)."""
import numpy as np
import optuna
import pytest
import torch

from src.common.losses import restoration_loss
from src.data.corruptions import corrupt_batch
from src.data.pets import GpuBatcher, split_indices
from src.models.autoencoder import ConvAE
from src.models.classifier import CHANNEL_CONFIGS, CorruptionClassifier
from src.train.ae import COND_IDS, evaluate as ae_eval, train_ae
from src.train.classifier import balanced_labels, evaluate as cls_eval, train_cls


def fake_images(n, seed=0):
    """Low-dimensional structured images (linear colour gradients) so tiny models can learn them."""
    r = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:128, 0:128] / 127.0
    out = []
    for _ in range(n):
        a, b, c = r.random(3), r.random(3), r.random(3)
        out.append(np.clip(np.stack([a[i] * xx + b[i] * yy + 0.5 * c[i] for i in range(3)], -1), 0, 1))
    return (np.stack(out) * 255).astype(np.uint8)


def fake_val(n_img=16, seed=1, labels=(0, 1, 2, 3)):
    imgs = torch.from_numpy(fake_images(n_img, seed)).permute(0, 3, 1, 2).float() / 255
    corr_l, clean_l, lab_l = [], [], []
    for c in labels:
        corr, _ = corrupt_batch(imgs, torch.full((n_img,), c))
        corr_l.append(corr), clean_l.append(imgs), lab_l.append(torch.full((n_img,), c))
    return {"corr": torch.cat(corr_l), "clean": torch.cat(clean_l), "labels": torch.cat(lab_l), "levels": [None] * n_img * len(labels)}


def test_split_80_20_seed42_is_stable_and_disjoint():
    tr, va = split_indices(3680)
    assert len(tr) == 2944 and len(va) == 736 and not set(tr) & set(va)
    tr2, va2 = split_indices(3680)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)


def test_gpu_batcher_covers_epoch_once():
    imgs = fake_images(40)
    b = GpuBatcher(imgs, 8, "cpu", drop_last=True)
    seen = [x for x in b]
    assert len(seen) == 5 and all(x.shape == (8, 3, 128, 128) for x in seen)
    assert seen[0].min() >= 0 and seen[0].max() <= 1


def test_ae_shapes_and_bottleneck():
    m = ConvAE(base_ch=16, latent_dim=64, dropout=0.1)
    x = torch.rand(2, 3, 128, 128)
    assert m.encode(x).shape == (2, 64)
    y = m(x)
    assert y.shape == x.shape and y.min() >= 0 and y.max() <= 1
    # no skip connections: the output must depend only on the latent code
    m.eval()
    z = m.encode(x)
    assert torch.allclose(m.decode(z), m(x), atol=1e-6)


@pytest.mark.parametrize("cfg", list(CHANNEL_CONFIGS))
def test_classifier_shapes(cfg):
    m = CorruptionClassifier(cfg, 0.3)
    assert m(torch.rand(3, 3, 128, 128)).shape == (3, 4)


def test_balanced_labels_exact():
    for b in (8, 64, 128):
        counts = torch.bincount(balanced_labels(b, "cpu"), minlength=4)
        assert counts.tolist() == [b // 4] * 4


def test_restoration_loss_perfect_is_zero_and_bounded():
    x = torch.rand(2, 3, 128, 128)
    assert restoration_loss(x, x, 0.8).item() < 1e-4
    assert restoration_loss(x, 1 - x, 0.8).item() > 0.1


def test_ae_trains_and_beats_untrained():
    pets = {"train": fake_images(64)}
    val = fake_val(8, labels=(1,))
    best, state, hist, cfg = train_ae(dict(base_ch=16, latent_dim=64, batch_size=16, lr=3e-3), pets, val, "salt", 6, "cpu", log=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert best < hist[0]["val_objective"]
    m = ConvAE(16, 64, cfg["dropout"])
    m.load_state_dict(state)
    # restored PSNR must clearly improve over the first epoch (learning signal, not an absolute target)
    assert ae_eval(m, val)["val_psnr"] > hist[0]["val_psnr"] + 1.0


def test_classifier_learns_corruptions():
    pets = {"train": fake_images(128)}
    val = fake_val(16)
    best, state, hist, _ = train_cls(dict(channels="small", batch_size=32, lr=3e-3, dropout=0.1), pets, val, 14, "cpu", log=False)
    assert best > 0.6, f"macro-F1 {best} should be well above chance (0.25) on easy synthetic data"
    m = CorruptionClassifier("small", 0.1)
    m.load_state_dict(state)
    assert cls_eval(m, val)["val_acc"] > 0.55


def test_optuna_objective_runs_and_prunes_api():
    from src.optuna_studies.cls_study import make_objective
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    import src.common.tracking as T
    pets = {"train": fake_images(130)}  # >= the largest batch size in the search space (128)
    val = fake_val(4)
    study = optuna.create_study(direction="maximize")
    # mlflow runs need an initialised tracking uri; use a throwaway dir
    import os, tempfile
    T.TRACKING_DIR = tempfile.mkdtemp()
    T.init("pytest")
    study.optimize(make_objective(pets, val, 2, "cpu"), n_trials=2)
    assert len(study.trials) == 2 and all(t.state.name == "COMPLETE" for t in study.trials)
    assert set(study.best_params) == {"lr", "batch_size", "channels", "dropout", "weight_decay"}


def test_gpu_batcher_clamps_batch_size_to_dataset():
    b = GpuBatcher(fake_images(10), 128, "cpu")
    assert b.bs == 10 and len(b) == 1 and next(iter(b)).shape[0] == 10


@pytest.mark.parametrize("latent_ch", [4, 16])
def test_conv_bottleneck_is_compressed_and_has_no_skip(latent_ch):
    m = ConvAE(base_ch=16, dropout=0.1, bottleneck="conv", latent_ch=latent_ch).eval()
    x = torch.rand(2, 3, 128, 128)
    z = m.encode(x)
    assert z.shape == (2, latent_ch, 8, 8) and m.latent_size == latent_ch * 64
    assert m.latent_size < 3 * 128 * 128 / 10          # at least 10x compression
    y = m(x)
    assert y.shape == x.shape and torch.allclose(m.decode(z), y, atol=1e-6)  # output depends only on the code


def test_linear_checkpoint_keys_unchanged_by_conv_option():
    """Checkpoints from the linear-bottleneck run must still load."""
    old = ConvAE(base_ch=16, latent_dim=64, bottleneck="linear")
    assert any(k.startswith("to_latent.1.") for k in old.state_dict())
    ConvAE(base_ch=16, latent_dim=64).load_state_dict(old.state_dict())


def test_ae_trains_with_conv_bottleneck():
    pets = {"train": fake_images(64)}
    val = fake_val(8, labels=(1,))
    best, state, hist, cfg = train_ae(dict(base_ch=16, batch_size=16, lr=3e-3, bottleneck="conv", latent_ch=8),
                                      pets, val, "salt", 6, "cpu", log=False)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"] and best < hist[0]["val_objective"]


def test_ae_study_objective_samples_conditional_space():
    from src.optuna_studies.ae_study import make_objective
    import src.common.tracking as T, tempfile
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    T.TRACKING_DIR = tempfile.mkdtemp()
    T.init("pytest-ae")
    pets, val = {"train": fake_images(40)}, fake_val(4, labels=(1,))
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(make_objective(pets, val, "salt", 1, "cpu"), n_trials=4)
    for t in study.trials:
        assert t.state.name == "COMPLETE"
        assert ("latent_dim" in t.params) == (t.params["bottleneck"] == "linear")
        assert ("latent_ch" in t.params) == (t.params["bottleneck"] == "conv")
