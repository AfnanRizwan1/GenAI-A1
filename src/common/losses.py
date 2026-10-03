import torch
import torch.nn.functional as F
from pytorch_msssim import ssim


def ssim_fn(a, b):
    """Mean SSIM over the batch, images in [0,1]. Always computed in fp32."""
    return ssim(a.float(), b.float(), data_range=1.0, size_average=True)


def restoration_loss(pred, target, alpha):
    """alpha * L1 + (1 - alpha) * (1 - SSIM)  (Tasks 1-2)."""
    return alpha * F.l1_loss(pred.float(), target.float()) + (1 - alpha) * (1 - ssim_fn(pred, target))
