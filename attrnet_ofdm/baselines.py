"""Capacity-oriented convolutional and compact HA02-style baselines."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ResidualCNN(nn.Module):
    """Residual full-grid estimator for capacity-matched comparisons."""
    def __init__(self, features: int = 32, blocks: int = 4):
        super().__init__()
        self.stem = nn.Conv2d(2, features, 3, padding=1)
        self.blocks = nn.Sequential(*[nn.Sequential(nn.Conv2d(features, features, 3, padding=1), nn.GELU(), nn.Conv2d(features, features, 3, padding=1)) for _ in range(blocks)])
        self.head = nn.Conv2d(features, 2, 3, padding=1)

    def forward(self, x):
        if x.shape[-2:] != (72, 14):
            x = F.interpolate(x, size=(72, 14), mode="bilinear", align_corners=False)
        h = self.stem(x)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class CapacityAttRNet(nn.Module):
    """Thin named wrapper: configure existing AttRNet through features/blocks."""
    def __init__(self, attention: str = "fc", features: int = 24, blocks: int = 3):
        super().__init__()
        from .model import AttRNet
        self.model = AttRNet(attention=attention, features=features, blocks=blocks)

    def forward(self, x):
        return self.model(x)


class HA02Style(nn.Module):
    """Pilot-token Transformer encoder with a bilinear-upsample CNN decoder."""
    def __init__(self, d: int = 32, heads: int = 4, blocks: int = 2):
        super().__init__()
        self.embed = nn.Linear(28, d)
        layer = nn.TransformerEncoderLayer(d, heads, 2*d, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, blocks)
        self.map = nn.Linear(d, 64*6*4)
        self.decoder = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.GELU(), nn.Conv2d(32, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 2, 3, padding=1))

    def forward(self, h_ls, coords):
        from .pacenet import fourier_features
        if coords.ndim == 2:
            coords = coords.unsqueeze(0).expand(h_ls.shape[0], -1, -1)
        norm = coords.to(h_ls.dtype) / h_ls.new_tensor([71, 13])
        z = self.encoder(self.embed(torch.cat((h_ls, fourier_features(norm)), dim=-1)))
        grid = self.map(z.mean(1)).view(-1, 64, 6, 4)
        grid = F.interpolate(grid, size=(72, 14), mode="bilinear", align_corners=False)
        return self.decoder(grid)
