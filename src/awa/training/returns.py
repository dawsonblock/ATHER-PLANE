from __future__ import annotations
import torch


def lambda_returns(rewards: torch.Tensor, values: torch.Tensor, discounts: torch.Tensor, lambda_: float = 0.95, bootstrap: torch.Tensor | None = None):
    """Compute generalized lambda returns. Shapes [T,B] or [T,B,1]."""
    if bootstrap is None:
        bootstrap = values[-1].detach()
    returns = []
    next_return = bootstrap
    for t in reversed(range(rewards.shape[0])):
        next_value = values[t + 1].detach() if t + 1 < values.shape[0] else bootstrap
        mixed = (1.0 - lambda_) * next_value + lambda_ * next_return
        next_return = rewards[t] + discounts[t] * mixed
        returns.append(next_return)
    return torch.stack(list(reversed(returns)), dim=0)
