from __future__ import annotations
import torch
from torch import nn


class IdentityEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


class MLPEncoder(nn.Module):
    def __init__(self, obs_dim: int, latent_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.float())
