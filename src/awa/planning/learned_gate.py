from __future__ import annotations
import torch
from torch import nn


class LearnedArbitrator(nn.Module):
    """Learned probability that deliberative planning is worth its compute cost."""
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(5,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))

    def features(self, uncertainty, actor_entropy, value_spread, planning_cost=0.0, novelty=0.0):
        raw=[]
        for x in (uncertainty, actor_entropy, value_spread, planning_cost, novelty):
            if not torch.is_tensor(x):
                x=torch.tensor(x,dtype=torch.float32)
            raw.append(x.float().reshape(-1))
        n=max(x.numel() for x in raw)
        xs=[]
        for x in raw:
            if x.numel()==1 and n>1:
                x=x.expand(n)
            elif x.numel()!=n:
                raise ValueError("arbitrator diagnostic inputs must be scalar or share a batch size")
            xs.append(x.reshape(-1,1))
        return torch.cat(xs,dim=-1)

    def forward(self, uncertainty, actor_entropy, value_spread, planning_cost=0.0, novelty=0.0):
        return torch.sigmoid(self.net(self.features(uncertainty,actor_entropy,value_spread,planning_cost,novelty))).squeeze(-1)

    def loss(self, features: torch.Tensor, planner_beneficial: torch.Tensor):
        p=torch.sigmoid(self.net(features)).squeeze(-1)
        y=planner_beneficial.float().reshape_as(p)
        return torch.nn.functional.binary_cross_entropy(p,y)
