"""Rebuild trained models from the checkpoints written by src.train.* (used by evaluation, export, app)."""
import torch

from .autoencoder import ConvAE
from .classifier import CorruptionClassifier


def load_ae(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    model = ConvAE(c["base_ch"], c.get("latent_dim", 512), c.get("dropout", 0.0),
                   bottleneck=c.get("bottleneck", "linear"), latent_ch=c.get("latent_ch", 16))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def load_classifier(path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    model = CorruptionClassifier(c["channels"], c.get("dropout", 0.0))
    model.load_state_dict(ck["state_dict"])
    return model.to(device).eval()


def load_soft_moe(path, device="cpu"):
    """Rebuild a trained SoftMoE from src.train.soft_moe's checkpoint."""
    from .soft_moe import SoftMoE
    ck = torch.load(path, map_location=device, weights_only=False)

    def ae(c):
        return ConvAE(c["base_ch"], c.get("latent_dim", 512), c.get("dropout", 0.0),
                      bottleneck=c.get("bottleneck", "linear"), latent_ch=c.get("latent_ch", 16))

    gc = ck["gate_cfg"]
    moe = SoftMoE(CorruptionClassifier(gc["channels"], gc.get("dropout", 0.0)),
                  [ae(c) for c in ck["expert_cfgs"]], ck["cfg"]["temperature"])
    moe.load_state_dict(ck["state_dict"])
    return moe.to(device).eval()


def load_generator(path, device="cpu"):
    """Rebuild the Task 4 U-Net generator from src.train.cgan's checkpoint."""
    from .cgan import UNetGenerator
    ck = torch.load(path, map_location=device, weights_only=False)
    c = ck["cfg"]
    g = UNetGenerator(c["base_ch"], c["emb_dim"], c.get("dropout", 0.0))
    g.load_state_dict(ck["state_dict"])
    return g.to(device).eval()
