"""Convolutional denoising autoencoder with a genuine bottleneck (Task 1 + Task 2 specialists).

128x128x3 -> 4 stride-2 stages (128->8 spatial, channels c, 2c, 4c, 8c) -> bottleneck ->
mirrored decoder -> sigmoid. No skip connections: every output pixel has to pass through
the bottleneck code.

Two bottleneck types (compared in the report):
  * "linear": flatten -> Linear(latent_dim). A single vector, loses spatial layout.
  * "conv":   1x1 conv to `latent_ch` channels, i.e. a (latent_ch, 8, 8) code
              (latent_ch * 64 numbers, e.g. 16 -> 1024 vs 49152 input values).
              Keeps spatial structure while still compressing ~48x.
"""
import torch.nn as nn


def _down(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.LeakyReLU(0.2, inplace=True))


def _up(cin, cout):
    return nn.Sequential(
        nn.ConvTranspose2d(cin, cout, 4, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class ConvAE(nn.Module):
    def __init__(self, base_ch=32, latent_dim=512, dropout=0.1, img_size=128, bottleneck="linear", latent_ch=16):
        super().__init__()
        c = base_ch
        self.bottleneck = bottleneck
        self.c8, self.sp = 8 * c, img_size // 16
        self.encoder = nn.Sequential(_down(3, c), _down(c, 2 * c), _down(2 * c, 4 * c), _down(4 * c, 8 * c))
        if bottleneck == "linear":
            flat = self.c8 * self.sp * self.sp
            self.latent_size = latent_dim
            self.to_latent = nn.Sequential(nn.Flatten(), nn.Linear(flat, latent_dim))
            self.from_latent = nn.Sequential(nn.Linear(latent_dim, flat), nn.ReLU(inplace=True))
        elif bottleneck == "conv":
            self.latent_size = latent_ch * self.sp * self.sp
            self.to_latent = nn.Conv2d(self.c8, latent_ch, 1)
            self.from_latent = nn.Sequential(
                nn.Conv2d(latent_ch, self.c8, 3, 1, 1), nn.BatchNorm2d(self.c8), nn.ReLU(inplace=True))
        else:
            raise ValueError(bottleneck)
        self.drop = nn.Dropout(dropout)
        self.decoder = nn.Sequential(_up(8 * c, 4 * c), _up(4 * c, 2 * c), _up(2 * c, c), _up(c, c),
                                     nn.Conv2d(c, 3, 3, 1, 1), nn.Sigmoid())

    def encode(self, x):
        return self.to_latent(self.encoder(x))

    def decode(self, z):
        h = self.from_latent(z)
        if self.bottleneck == "linear":
            h = h.view(-1, self.c8, self.sp, self.sp)
        return self.decoder(h)

    def forward(self, x):
        return self.decode(self.drop(self.encode(x)))
