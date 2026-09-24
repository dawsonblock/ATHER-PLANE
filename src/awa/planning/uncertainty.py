from __future__ import annotations
import torch
from torch import nn


class UncertaintyEnsemble(nn.Module):
    """Ensemble predictor for epistemic transition uncertainty.

    v1.7 can condition predictions on the proposed action. `action_dim=0`
    preserves the v1.6 state-only API for backward compatibility.
    """
    def __init__(self, belief_dim: int, latent_dim: int, heads: int = 4, hidden_dim: int = 128,
                 action_dim: int = 0):
        super().__init__(); self.action_dim=int(action_dim)
        inp=belief_dim+self.action_dim
        self.heads = nn.ModuleList([
            nn.Sequential(nn.Linear(inp, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))
            for _ in range(heads)
        ])

    def _input(self, belief_vec: torch.Tensor, action: torch.Tensor | None=None):
        if self.action_dim<=0: return belief_vec
        if action is None:
            action=torch.zeros(belief_vec.shape[0],self.action_dim,device=belief_vec.device,dtype=belief_vec.dtype)
        if action.ndim==1: action=action.unsqueeze(0)
        if action.shape[-1]!=self.action_dim: raise ValueError('uncertainty action dimension mismatch')
        return torch.cat([belief_vec,action.to(device=belief_vec.device,dtype=belief_vec.dtype)],dim=-1)

    def forward(self, belief_vec, action: torch.Tensor | None=None):
        x=self._input(belief_vec,action)
        preds = torch.stack([h(x) for h in self.heads], dim=0)
        mean = preds.mean(dim=0)
        disagreement = preds.var(dim=0, unbiased=False).mean(dim=-1)
        return mean, disagreement
