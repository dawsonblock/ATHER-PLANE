from __future__ import annotations
from typing import Protocol
import torch


class ExternalEncoder(Protocol):
    @property
    def output_dim(self) -> int: ...
    def encode(self, observation) -> torch.Tensor: ...


class FrozenExternalAdapter(torch.nn.Module):
    """Wrap any external pretrained encoder behind the local interface."""

    def __init__(self, encoder: ExternalEncoder, projection: torch.nn.Module):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        for p in self.projection.parameters():
            p.requires_grad = True

    def forward(self, observation) -> torch.Tensor:
        with torch.no_grad():
            z = self.encoder.encode(observation)
        return self.projection(z)
