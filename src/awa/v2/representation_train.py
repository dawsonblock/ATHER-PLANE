from __future__ import annotations

from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class ProjectionMetrics:
    loss: float
    geometry_loss: float
    variance_penalty: float


def _off_diagonal_similarity(x: torch.Tensor) -> torch.Tensor:
    x = torch.nn.functional.normalize(x, dim=-1)
    sim = x @ x.transpose(0, 1)
    n = sim.shape[0]
    mask = ~torch.eye(n, dtype=torch.bool, device=sim.device)
    return sim[mask]


class GeometryPreservingProjectionTrainer:
    """Train only the projection while preserving frozen-backbone sample geometry.

    This provides a label-free representation projection objective for v2.1. It is
    intentionally modest: downstream world-model losses remain the real criterion.
    """

    def __init__(self, projection: nn.Module, lr: float = 3e-4, variance_floor: float = 0.2, variance_weight: float = 0.1):
        self.projection = projection
        self.optimizer = torch.optim.AdamW(self.projection.parameters(), lr=lr)
        self.variance_floor = float(variance_floor)
        self.variance_weight = float(variance_weight)

    def step(self, frozen_features: torch.Tensor) -> ProjectionMetrics:
        source = frozen_features.detach()
        projected = self.projection(source)
        if source.shape[0] < 2:
            geometry = projected.square().mean() * 0.0
        else:
            geometry = torch.nn.functional.mse_loss(
                _off_diagonal_similarity(projected),
                _off_diagonal_similarity(source),
            )
        std = projected.std(dim=0, unbiased=False)
        variance = torch.relu(self.variance_floor - std).mean()
        loss = geometry + self.variance_weight * variance
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return ProjectionMetrics(float(loss.detach()), float(geometry.detach()), float(variance.detach()))
