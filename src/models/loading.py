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
