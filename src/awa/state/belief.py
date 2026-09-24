from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class Belief:
    deterministic: torch.Tensor
    stochastic: torch.Tensor
    slow: torch.Tensor | None = None

    @property
    def vector(self) -> torch.Tensor:
        parts = [self.deterministic, self.stochastic]
        if self.slow is not None:
            parts.append(self.slow)
        return torch.cat(parts, dim=-1)

    @property
    def fast_vector(self) -> torch.Tensor:
        return torch.cat([self.deterministic, self.stochastic], dim=-1)

    def detach(self) -> "Belief":
        return Belief(
            self.deterministic.detach(),
            self.stochastic.detach(),
            None if self.slow is None else self.slow.detach(),
        )


class StochasticBeliefState(nn.Module):
    def __init__(self, deterministic_dim: int, obs_latent_dim: int, stochastic_dim: int):
        super().__init__()
        self.stochastic_dim = stochastic_dim
        self.prior = nn.Linear(deterministic_dim, stochastic_dim * 2)
        self.posterior = nn.Sequential(
            nn.Linear(deterministic_dim + obs_latent_dim, deterministic_dim),
            nn.SiLU(),
            nn.Linear(deterministic_dim, stochastic_dim * 2),
        )

    @staticmethod
    def _stats(raw: torch.Tensor):
        mean, raw_scale = raw.chunk(2, dim=-1)
        scale = torch.nn.functional.softplus(raw_scale) + 0.05
        return mean, scale

    def prior_dist(self, h: torch.Tensor):
        mean, scale = self._stats(self.prior(h))
        return torch.distributions.Normal(mean, scale)

    def posterior_dist(self, h: torch.Tensor, z: torch.Tensor):
        mean, scale = self._stats(self.posterior(torch.cat([h, z], dim=-1)))
        return torch.distributions.Normal(mean, scale)

    def sample_posterior(self, h: torch.Tensor, z: torch.Tensor):
        dist = self.posterior_dist(h, z)
        return dist.rsample(), dist

    def sample_prior(self, h: torch.Tensor):
        dist = self.prior_dist(h)
        return dist.rsample(), dist
