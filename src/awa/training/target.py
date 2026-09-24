from __future__ import annotations
import copy
import torch
from torch import nn


class TargetValueNetwork(nn.Module):
    """EMA target critic used only for bootstrap targets."""
    def __init__(self, value_head: nn.Module):
        super().__init__()
        self.net = copy.deepcopy(value_head)
        for p in self.net.parameters(): p.requires_grad = False

    @torch.no_grad()
    def update(self, source: nn.Module, tau: float = 0.01):
        for t, s in zip(self.net.parameters(), source.parameters()):
            t.data.lerp_(s.data, tau)

    def forward(self, features):
        return self.net(features)
