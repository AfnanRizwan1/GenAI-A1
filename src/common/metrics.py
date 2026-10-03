import torch
from pytorch_msssim import ssim


def psnr_per_image(pred, target):
    mse = ((pred.float() - target.float()) ** 2).mean((1, 2, 3)).clamp_min(1e-10)
    return 10 * torch.log10(1.0 / mse)


def ssim_per_image(pred, target):
    return ssim(pred.float(), target.float(), data_range=1.0, size_average=False)


def l1_per_image(pred, target):
    return (pred.float() - target.float()).abs().mean((1, 2, 3))
