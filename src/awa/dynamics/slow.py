from __future__ import annotations
import torch
from torch import nn


class SlowStateModel(nn.Module):
    """Slow event/context state updated at a configurable stride."""

    def __init__(self, input_dim: int, slow_dim: int = 128):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, slow_dim)

    def forward(self, slow_state: torch.Tensor, summary: torch.Tensor) -> torch.Tensor:
        return self.cell(summary, slow_state)
