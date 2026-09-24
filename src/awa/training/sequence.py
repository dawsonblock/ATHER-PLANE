from __future__ import annotations
import torch
import torch.nn.functional as F
from awa.state.belief import Belief


def continuation_mask(dones: torch.Tensor) -> torch.Tensor:
    if dones.ndim != 3:
        raise ValueError("dones must have shape [B,T,1]")
    return torch.cumprod(torch.cat([torch.ones_like(dones[:, :1]), 1.0 - dones[:, :-1]], dim=1), dim=1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    m = mask
    while m.ndim < x.ndim: m = m.unsqueeze(-1)
    if weights is not None:
        w = weights
        while w.ndim < x.ndim: w = w.unsqueeze(-1)
        m = m * w
    denom = m.expand_as(x).sum().clamp_min(1e-8)
    return (x * m).sum() / denom


def _model_action(actions: torch.Tensor, index: int, action_dim: int, action_type: str):
    a = actions[:, index]
    if action_type == "discrete":
        return F.one_hot(a.long(), action_dim).float()
    return a.float()


def overshooting_latent_loss(world_model, posterior_beliefs: list[Belief], observations: torch.Tensor,
                              actions: torch.Tensor, action_dim: int, horizon: int = 3,
                              burn_in: int = 0, action_type: str = "discrete") -> torch.Tensor:
    """Multi-step latent consistency for discrete or continuous recorded actions."""
    if horizon <= 1 or not posterior_beliefs:
        return torch.zeros((), device=observations.device)
    T = actions.shape[1]
    losses=[]
    for start in range(max(0, burn_in), T):
        b = posterior_beliefs[start].detach()
        for k in range(1, min(horizon, T-start)+1):
            a = _model_action(actions, start+k-1, action_dim, action_type)
            b, _, _, _, _, _ = world_model.imagine(b, a, step_index=start+k)
            target = world_model.encoder(observations[:, start+k]).detach()
            pred = world_model.latent_prediction(b)
            losses.append(F.mse_loss(pred, target))
    return torch.stack(losses).mean() if losses else torch.zeros((), device=observations.device)
