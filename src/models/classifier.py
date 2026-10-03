"""Corruption classifier (Task 2 hard router, Task 3 gate initialisation).

Four conv blocks (stride-2) -> global average pool -> dropout -> linear(4).
Outputs logits for [clean, salt, blur, occlusion].
"""
import torch.nn as nn

CHANNEL_CONFIGS = {
    "small": (16, 32, 64, 128),
    "medium": (32, 64, 128, 256),
    "large": (48, 96, 192, 384),
}


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, 1, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, 2, 1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class CorruptionClassifier(nn.Module):
    def __init__(self, channels="medium", dropout=0.3, num_classes=4):
        super().__init__()
        ch = CHANNEL_CONFIGS[channels] if isinstance(channels, str) else tuple(channels)
        layers, cin = [], 3
        for c in ch:
            layers.append(_block(cin, c))
            cin = c
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, num_classes))

    def forward(self, x):
        return self.head(self.features(x))
