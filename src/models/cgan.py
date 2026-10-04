"""Style-conditioned face-to-sketch cGAN (Task 4).

Generator: U-Net, G(x, s) -> sketch.   Style s in {0,1,2} -> learned nn.Embedding(3, e).
  The embedding enters the generator in two places so it is part of the model, not an interface label:
    1. broadcast over the image and concatenated to the input photo,
    2. FiLM (per-channel scale and shift predicted from the embedding) in every decoder block.
Discriminator: PatchGAN on concat(photo, sketch, broadcast style-embedding map) -> grid of real/fake logits.

Images are in [-1, 1] (tanh output).
"""
import torch
import torch.nn as nn

NUM_STYLES = 3


class FiLMBlock(nn.Module):
    """Decoder block: upsample -> conv -> InstanceNorm -> FiLM(style) -> ReLU (+ dropout)."""

    def __init__(self, cin, cout, emb_dim, dropout=0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(cin, cout, 4, 2, 1, bias=False)
        self.norm = nn.InstanceNorm2d(cout, affine=False)
        self.film = nn.Linear(emb_dim, 2 * cout)
        nn.init.zeros_(self.film.weight), nn.init.zeros_(self.film.bias)  # starts as plain instance norm
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, e):
        h = self.norm(self.up(x))
        gamma, beta = self.film(e).chunk(2, dim=1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return self.drop(torch.relu(h))


class UNetGenerator(nn.Module):
    """6 down / 6 up levels for 128x128 inputs (128 -> 2 -> 128), skip connections as in pix2pix."""

    def __init__(self, base_ch=64, emb_dim=16, dropout=0.3, in_ch=3, out_ch=3):
        super().__init__()
        self.emb = nn.Embedding(NUM_STYLES, emb_dim)
        c = base_ch
        chs = [c, 2 * c, 4 * c, 8 * c, 8 * c, 8 * c]
        self.downs = nn.ModuleList()
        cin = in_ch + emb_dim
        for i, co in enumerate(chs):
            layers = [nn.Conv2d(cin, co, 4, 2, 1, bias=False)]
            if i > 0 and i < len(chs) - 1:
                layers.append(nn.InstanceNorm2d(co))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            self.downs.append(nn.Sequential(*layers))
            cin = co
        self.ups = nn.ModuleList()
        rev = chs[::-1]
        for i in range(len(rev) - 1):
            cin_up = rev[i] if i == 0 else 2 * rev[i]
            self.ups.append(FiLMBlock(cin_up, rev[i + 1], emb_dim, dropout if i < 3 else 0.0))
        self.final = nn.Sequential(nn.ConvTranspose2d(2 * chs[0], out_ch, 4, 2, 1), nn.Tanh())

    def forward(self, x, style):
        e = self.emb(style)                                           # (B, emb)
        h = torch.cat([x, e[:, :, None, None].expand(-1, -1, *x.shape[2:])], 1)
        skips = []
        for d in self.downs:
            h = d(h)
            skips.append(h)
        h = skips.pop()                                               # bottleneck
        for u in self.ups:
            h = u(h, e)
            h = torch.cat([h, skips.pop()], 1)
        return self.final(h)


class PatchDiscriminator(nn.Module):
    """PatchGAN: 4 stride-2 convs + 1 stride-1 conv -> (B,1,h,w) patch logits (receptive field ~ 70 px)."""

    def __init__(self, base_ch=64, emb_dim=16, in_ch=6):
        super().__init__()
        self.emb = nn.Embedding(NUM_STYLES, emb_dim)
        c = base_ch
        cfg = [(in_ch + emb_dim, c, 2, False), (c, 2 * c, 2, True), (2 * c, 4 * c, 2, True), (4 * c, 8 * c, 1, True)]
        layers = []
        for cin, cout, stride, norm in cfg:
            layers.append(nn.Conv2d(cin, cout, 4, stride, 1, bias=not norm))
            if norm:
                layers.append(nn.InstanceNorm2d(cout))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        layers.append(nn.Conv2d(8 * c, 1, 4, 1, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, photo, sketch, style):
        e = self.emb(style)
        emap = e[:, :, None, None].expand(-1, -1, *photo.shape[2:])
        return self.net(torch.cat([photo, sketch, emap], 1))
