from __future__ import annotations
import math
import torch
from torch import nn


class CategoricalActor(nn.Module):
    action_type = "discrete"
    def __init__(self, belief_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__(); self.action_dim = action_dim
        self.net = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim))
    def logits(self, belief_vec): return self.net(belief_vec)
    def distribution(self, belief_vec): return torch.distributions.Categorical(logits=self.logits(belief_vec))
    def sample_onehot(self, belief_vec):
        idx = self.distribution(belief_vec).sample()
        return torch.nn.functional.one_hot(idx, self.action_dim).float(), idx
    def deterministic_action(self, belief_vec):
        idx = self.logits(belief_vec).argmax(-1)
        return torch.nn.functional.one_hot(idx, self.action_dim).float(), idx


class TanhGaussianActor(nn.Module):
    """Squashed Gaussian actor for bounded continuous actions."""
    action_type = "continuous"
    def __init__(self, belief_dim: int, action_dim: int, hidden_dim: int = 256,
                 low=None, high=None, log_std_min: float = -5.0, log_std_max: float = 2.0):
        super().__init__(); self.action_dim = int(action_dim)
        self.net = nn.Sequential(nn.Linear(belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.mean_head = nn.Linear(hidden_dim, self.action_dim); self.log_std_head = nn.Linear(hidden_dim, self.action_dim)
        self.log_std_min=float(log_std_min); self.log_std_max=float(log_std_max)
        lo = torch.full((self.action_dim,), -1.0) if low is None else torch.as_tensor(low, dtype=torch.float32).reshape(-1)
        hi = torch.full((self.action_dim,), 1.0) if high is None else torch.as_tensor(high, dtype=torch.float32).reshape(-1)
        if lo.numel()!=self.action_dim or hi.numel()!=self.action_dim or torch.any(hi<=lo): raise ValueError("invalid continuous action bounds")
        self.register_buffer("low", lo); self.register_buffer("high", hi)

    def _params(self, belief_vec):
        x=self.net(belief_vec); mean=self.mean_head(x); log_std=self.log_std_head(x).clamp(self.log_std_min,self.log_std_max)
        return mean, log_std

    def _scale(self, unit):
        return self.low + (unit + 1.0) * 0.5 * (self.high - self.low)

    def _unscale(self, action):
        unit = 2.0 * (action - self.low) / (self.high - self.low) - 1.0
        return unit.clamp(-0.999999, 0.999999)

    def sample_with_log_prob(self, belief_vec):
        mean,log_std=self._params(belief_vec); std=log_std.exp(); dist=torch.distributions.Normal(mean,std)
        raw=dist.rsample(); unit=torch.tanh(raw); action=self._scale(unit)
        # Change of variables for tanh plus affine action scaling.
        log_prob=dist.log_prob(raw) - torch.log(1.0-unit.pow(2)+1e-6)
        log_scale=torch.log((self.high-self.low)*0.5).sum()
        log_prob=log_prob.sum(-1)-log_scale
        entropy=dist.entropy().sum(-1)
        return action,log_prob,entropy

    def deterministic_action(self, belief_vec):
        mean,_=self._params(belief_vec); return self._scale(torch.tanh(mean))

    def distribution(self, belief_vec):
        # Exposed for diagnostics; callers requiring bounded samples should use sample_with_log_prob.
        mean,log_std=self._params(belief_vec); return torch.distributions.Independent(torch.distributions.Normal(mean,log_std.exp()),1)

    def log_prob(self, belief_vec, action):
        mean,log_std=self._params(belief_vec); std=log_std.exp(); dist=torch.distributions.Normal(mean,std)
        unit=self._unscale(action); raw=torch.atanh(unit)
        lp=dist.log_prob(raw)-torch.log(1.0-unit.pow(2)+1e-6)
        return lp.sum(-1)-torch.log((self.high-self.low)*0.5).sum()
