from __future__ import annotations
import torch
from torch import nn


class ExpertMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, dim)
        )
    def forward(self, x):
        return self.net(x)


class TaskConditionedDynamicsExperts(nn.Module):
    """Sparse top-k residual experts with optional task/context conditioning.

    The router sees the current feature vector plus an optional task context. The
    experts themselves only transform the feature vector, keeping context out of
    the residual state representation.
    """
    def __init__(self, dim: int, experts: int = 4, hidden_dim: int = 256,
                 top_k: int = 2, context_dim: int = 0):
        super().__init__()
        self.top_k = min(top_k, experts)
        self.context_dim = context_dim
        self.router = nn.Linear(dim + context_dim, experts)
        self.experts = nn.ModuleList([ExpertMLP(dim, hidden_dim) for _ in range(experts)])

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None):
        if self.context_dim:
            if context is None:
                context = torch.zeros(x.shape[0], self.context_dim, device=x.device, dtype=x.dtype)
            if context.shape[-1] != self.context_dim:
                raise ValueError(f"Expected context_dim={self.context_dim}, got {context.shape[-1]}")
            router_in = torch.cat([x, context], dim=-1)
        else:
            router_in = x
        logits = self.router(router_in)
        weights = torch.softmax(logits, dim=-1)
        vals, idx = torch.topk(weights, self.top_k, dim=-1)
        out = torch.zeros_like(x)
        for expert_id, expert in enumerate(self.experts):
            # Each sample may route to this expert in any top-k slot.
            coeff = torch.zeros(x.shape[0], 1, device=x.device, dtype=x.dtype)
            for rank in range(self.top_k):
                mask = idx[:, rank] == expert_id
                if mask.any():
                    coeff[mask] += vals[mask, rank].unsqueeze(-1)
            mask_any = coeff.squeeze(-1) > 0
            if mask_any.any():
                out[mask_any] += coeff[mask_any] * expert(x[mask_any])
        return x + out, weights


# Backward-compatible alias.
SparseDynamicsExperts = TaskConditionedDynamicsExperts
