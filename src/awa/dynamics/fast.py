from __future__ import annotations
import torch
from torch import nn


class GRUDynamics(nn.Module):
    def __init__(self, stochastic_dim: int, action_dim: int, deterministic_dim: int):
        super().__init__()
        self.cell = nn.GRUCell(stochastic_dim + action_dim, deterministic_dim)

    def forward(self, h, s, action):
        return self.cell(torch.cat([s, action], dim=-1), h)


class GatedSSMDynamics(nn.Module):
    """
    Lightweight SSM-like recurrent cell.
    It is deliberately dependency-free and serves as an ablation target.
    Replace with a production Mamba implementation behind the same interface.
    """
    def __init__(self, stochastic_dim: int, action_dim: int, deterministic_dim: int):
        super().__init__()
        inp = stochastic_dim + action_dim
        self.in_proj = nn.Linear(inp, deterministic_dim * 2)
        self.decay = nn.Parameter(torch.zeros(deterministic_dim))
        self.mix = nn.Linear(deterministic_dim, deterministic_dim)

    def forward(self, h, s, action):
        u, gate = self.in_proj(torch.cat([s, action], dim=-1)).chunk(2, dim=-1)
        decay = torch.sigmoid(self.decay)
        candidate = decay * h + (1.0 - decay) * torch.tanh(u)
        candidate = torch.tanh(self.mix(candidate))
        return torch.sigmoid(gate) * candidate + (1.0 - torch.sigmoid(gate)) * h
