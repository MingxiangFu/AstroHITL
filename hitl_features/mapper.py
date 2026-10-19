"""MLP feature map with L2-normalized outputs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureMapper(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        if int(in_dim) < 1 or int(out_dim) < 1:
            raise ValueError("feature dimensions must be positive")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(features), dim=-1)
