import numpy as np
from pathlib import Path
from PIL import Image
import optuna
import torch

from src.data.fs2k import PairBatcher, paired_augment, stratified_split
from src.models.cgan import PatchDiscriminator, UNetGenerator
from src.train.cgan import evaluate, train_cgan


def fake_pairs(n, seed=0, styles=3):
    """Smooth photos; sketch = inverted photo with a style-dependent offset (an easy, learnable mapping)."""
    r = np.random.default_rng(seed)
    p = np.kron(r.random((n, 8, 8, 3)), np.ones((1, 16, 16, 1)))
    st = np.arange(n) % styles
    s = np.clip(1 - p + 0.1 * st[:, None, None, None], 0, 1)
    return (p * 255).astype(np.uint8), (s * 255).astype(np.uint8), st


def fake_data(n_train=24, n_val=6):
    tp, ts, tst = fake_pairs(n_train)
    vp, vs, vst = fake_pairs(n_val, seed=5)
    return {"train": {"photo": tp, "sketch": ts, "style": tst}, "val": {"photo": vp, "sketch": vs, "style": vst}}


def test_shapes_and_ranges():
    g, d = UNetGenerator(8, 8, 0.3).eval(), PatchDiscriminator(8, 8).eval()
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    s = torch.tensor([0, 2])
    y = g(x, s)
    assert y.shape == x.shape and y.min() >= -1 and y.max() <= 1
    assert d(x, y, s).shape[0] == 2 and d(x, y, s).shape[1] == 1


def test_style_conditions_generator_and_discriminator():
    g, d = UNetGenerator(8, 8, 0.0).eval(), PatchDiscriminator(8, 8).eval()
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    a, b = torch.tensor([0, 0]), torch.tensor([1, 1])
    assert (g(x, a) - g(x, b)).abs().max() > 1e-6
    assert (d(x, x, a) - d(x, x, b)).abs().max() > 1e-6


def test_style_embedding_is_trained_by_both_losses():
    """The embedding must receive gradients in G (input path + FiLM blocks) and in D, i.e. it is part of the model."""
    g, d = UNetGenerator(8, 8, 0.0), PatchDiscriminator(8, 8)
    for blk in g.ups:  # FiLM starts at zero; make it active so every block passes gradient to the embedding
        torch.nn.init.normal_(blk.film.weight, std=0.1)
    x = torch.rand(2, 3, 128, 128) * 2 - 1
    s = torch.tensor([0, 2])
    fake = g(x, s)
    d(x, fake, s).mean().backward()
    assert g.emb.weight.grad is not None and g.emb.weight.grad[[0, 2]].abs().sum() > 0
    assert g.emb.weight.grad[1].abs().sum() == 0            # unused style gets no gradient
    assert d.emb.weight.grad is not None and d.emb.weight.grad[[0, 2]].abs().sum() > 0
    assert all(blk.film.weight.grad is not None and blk.film.weight.grad.abs().sum() > 0 for blk in g.ups)


def test_stratified_split_proportions_disjoint_deterministic():
    styles = np.array([0] * 100 + [1] * 60 + [2] * 40)
    tr, va = stratified_split(styles)
    assert not set(tr) & set(va) and len(tr) + len(va) == 200
    for s, n in ((0, 100), (1, 60), (2, 40)):
        assert (styles[va] == s).sum() == round(n * 0.15)
    tr2, va2 = stratified_split(styles)
    assert np.array_equal(tr, tr2) and np.array_equal(va, va2)


def test_paired_augment_applies_identical_warp_to_both_images():
    gen = torch.Generator().manual_seed(0)
    p = torch.rand(8, 3, 128, 128) * 2 - 1
    s = -p                                       # any pointwise-linear relation must survive the warp
    pa, sa = paired_augment(p, s, gen)
    assert pa.shape == p.shape and torch.allclose(sa, -pa, atol=1e-5)
    assert (pa - p).abs().max() > 0.01          # the augmentation really changed the images
    # pure flip: exactly the mirrored image
    pf, sf = paired_augment(p, s, gen, max_rot_deg=0, scale=(1, 1), max_shift=0, p_flip=1.0)
    assert torch.allclose(pf, p.flip(-1), atol=1e-4) and torch.allclose(sf, s.flip(-1), atol=1e-4)


def test_pair_batcher_keeps_pairing():
    n = 20
    photos = np.stack([np.full((128, 128, 3), i, np.uint8) for i in range(n)])
    sketches = np.stack([np.full((128, 128, 3), 255 - i, np.uint8) for i in range(n)])
    b = PairBatcher(photos, sketches, np.arange(n) % 3, 8, "cpu")
    for p, s, st in b:
        pv = ((p[:, 0, 0, 0] + 1) * 127.5).round().long()
        sv = ((s[:, 0, 0, 0] + 1) * 127.5).round().long()
        assert torch.equal(pv + sv, torch.full_like(pv, 255)) and torch.equal(st, pv % 3)


def test_training_logs_all_losses_and_saves_samples(tmp_path):
    data = fake_data()
    cfg = dict(base_ch=8, emb_dim=8, batch_size=8, lam=100.0, dropout=0.0)
    best, state, hist, _ = train_cgan(cfg, data, epochs=6, device="cpu", log=False, out_dir=tmp_path, sample_every=3)
    assert len(hist) == 6
    for k in ("d_real", "d_fake", "g_adv", "g_rec", "val_l1", "val_ssim", "val_psnr"):
        assert k in hist[-1] and np.isfinite(hist[-1][k]), k
    assert hist[-1]["g_rec"] < hist[0]["g_rec"]                       # the generator learns the easy mapping
    assert (tmp_path / "samples" / "epoch_000.png").exists() and (tmp_path / "samples" / "epoch_005.png").exists()
    g = UNetGenerator(8, 8, 0.0)
    g.load_state_dict(state)                                           # best checkpoint is loadable
    assert evaluate(g, data["val"], "cpu")["val_psnr"] > 0


def test_optuna_objective_runs():
    from src.optuna_studies.cgan_study import make_objective
    import src.common.tracking as T, tempfile
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    T.TRACKING_DIR = tempfile.mkdtemp()
    T.init("pytest-cgan")
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=1))
    study.optimize(make_objective(fake_data(16, 4), 1, "cpu"), n_trials=2)
    assert all(t.state.name == "COMPLETE" for t in study.trials)
    assert {"lr_g", "lr_d", "batch_size", "base_ch", "dropout", "emb_dim", "lam"} <= set(study.best_params)


def _write_fake_fs2k(root, n_train=30, n_test=9):
    """On-disk layout like FS2K but with different photo/sketch file names and extensions."""
    import json
    from PIL import Image
    r = np.random.default_rng(0)
    for k in (1, 2, 3):
        (root / "photo" / f"photo{k}").mkdir(parents=True)
        (root / "sketch" / f"sketch{k}").mkdir(parents=True)
    annos = {"anno_train.json": [], "anno_test.json": []}
    counter = 0
    for fname, n in (("anno_train.json", n_train), ("anno_test.json", n_test)):
        for i in range(n):
            k, num = i % 3 + 1, f"{counter:04d}"
            counter += 1
            photo = (np.kron(r.random((4, 5, 3)), np.ones((10, 10, 1))) * 255).astype(np.uint8)  # smooth: JPEG-friendly
            Image.fromarray(photo).save(root / "photo" / f"photo{k}" / f"image{num}.jpg")
            Image.fromarray(255 - photo).save(root / "sketch" / f"sketch{k}" / f"sketch{num}.png")  # known relation
            annos[fname].append({"image_name": f"photo{k}/image{num}", "style": i % 3})
    for fname, a in annos.items():
        (root / fname).write_text(json.dumps(a))


def test_load_fs2k_pairs_split_and_cache(tmp_path):
    from src.data.fs2k import load_fs2k
    _write_fake_fs2k(tmp_path / "FS2K")                 # root given one level above, like an extracted archive
    d = load_fs2k(tmp_path)
    assert len(d["train"]["photo"]) + len(d["val"]["photo"]) == 30 and len(d["test"]["photo"]) == 9
    assert len(d["val"]["photo"]) == 3 * round(10 * 0.15)       # stratified: 10 per style, 15% each
    for part in d.values():
        assert part["photo"].shape[1:] == (128, 128, 3) and part["sketch"].shape == part["photo"].shape
        # pairing: the fake sketch is the inverted photo (up to JPEG compression and resizing)
        assert np.abs(part["photo"].astype(int) + part["sketch"].astype(int) - 255).mean() < 12
        assert set(np.unique(part["style"])) <= {0, 1, 2}
    assert not set(d["train"]["name"]) & set(d["val"]["name"]) and not set(d["train"]["name"]) & set(d["test"]["name"])
    again = load_fs2k(tmp_path)                                  # second call comes from the cache
    assert np.array_equal(again["val"]["name"], d["val"]["name"])
    assert (tmp_path / "FS2K" / "cache_128.npz").exists()


def test_evaluate_t4_outputs_and_independent_metric_check(tmp_path, monkeypatch):
    import json, sys
    import pandas as pd
    from src.data.fs2k import load_fs2k
    from src.eval import evaluate_t4 as ev

    _write_fake_fs2k(tmp_path / "FS2K", n_train=30, n_test=12)
    torch.save({"state_dict": UNetGenerator(8, 8, 0.0).state_dict(), "cfg": {"base_ch": 8, "emb_dim": 8, "dropout": 0.0}}, tmp_path / "gen.pt")
    out = tmp_path / "eval"
    monkeypatch.setattr(sys, "argv", ["evaluate_t4", "--gen", str(tmp_path / "gen.pt"), "--data-root", str(tmp_path), "--out", str(out)])
    ev.main()

    for f in ("summary.json", "per_image.csv", "examples_test.png", "failures_test.png", "style_conditioning_matrix.png"):
        assert (out / f).exists(), f
    s = json.loads((out / "summary.json").read_text())
    assert s["n_test_images"] == 12 and sum(s["images_per_style"].values()) == 12
    assert len(s["style_conditioning"]["mean_l1_matrix_true_vs_fed"]) == 3
    assert set(s["by_style"]) == {"Style 1", "Style 2", "Style 3"}

    # independent check of the grayscale-baseline L1 for the first test image, straight from the arrays
    test = load_fs2k(tmp_path)["test"]
    p, y = test["photo"][0].astype(np.float64) / 255, test["sketch"][0].astype(np.float64) / 255
    gray = 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]
    expected = np.abs(gray[..., None] - y).mean()
    got = pd.read_csv(out / "per_image.csv").loc[0, "gray_l1"]
    assert abs(expected - got) < 1e-4, (expected, got)


def test_make_samples_is_deterministic_square_and_sized(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_samples", Path(__file__).resolve().parents[1] / "scripts" / "make_samples.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    r = np.random.default_rng(0)
    ds = [(Image.fromarray((r.random((90 + i, 120 + 3 * i, 3)) * 255).astype(np.uint8)), 0) for i in range(20)]
    a = mod.make_samples(ds, tmp_path / "a", n=5)
    b = mod.make_samples(ds, tmp_path / "b", n=5)
    assert [p.name for p in a] == [f"pet_0{i}.jpg" for i in range(1, 6)]
    for pa, pb in zip(a, b):
        ia, ib = Image.open(pa), Image.open(pb)
        assert ia.size == (256, 256) and ia.mode == "RGB"
        assert np.array_equal(np.asarray(ia), np.asarray(ib))        # same seed -> same pictures
