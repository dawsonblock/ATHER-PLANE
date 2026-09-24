from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
import torch.nn.functional as F

@dataclass
class FutureDistribution:
    means: torch.Tensor       # [B,K,D]
    scales: torch.Tensor      # [B,K,D]
    logits: torch.Tensor      # [B,K]

    @property
    def probs(self): return torch.softmax(self.logits,-1)
    def sample(self,n:int=1):
        cat=torch.distributions.Categorical(logits=self.logits)
        idx=cat.sample((n,)).transpose(0,1)  # [B,n]
        b=torch.arange(self.means.shape[0],device=self.means.device).unsqueeze(1)
        mu=self.means[b,idx]; sd=self.scales[b,idx]
        return mu+torch.randn_like(mu)*sd
    def mode(self):
        idx=self.logits.argmax(-1); b=torch.arange(self.means.shape[0],device=self.means.device)
        return self.means[b,idx]
    def expected(self):
        return (self.probs.unsqueeze(-1)*self.means).sum(1)
    def entropy_proxy(self):
        cat=-(self.probs*torch.log(self.probs+1e-8)).sum(-1)
        aleatoric=(self.probs.unsqueeze(-1)*torch.log(self.scales+1e-6)).sum((1,2))
        return cat+aleatoric/self.means.shape[-1]

class MonotoneHorizonUncertainty(nn.Module):
    """Predicts non-decreasing uncertainty as planning horizon increases."""
    def __init__(self, belief_dim:int, horizons=(1,5,10,20,50), hidden:int=128):
        super().__init__(); self.horizons=tuple(int(h) for h in horizons)
        self.net=nn.Sequential(nn.Linear(belief_dim,hidden),nn.SiLU(),nn.Linear(hidden,len(self.horizons)))
    def forward(self,belief):
        increments=F.softplus(self.net(belief))
        return torch.cumsum(increments,-1)
    def at_horizon(self,belief,h:int):
        u=self(belief); idx=min(range(len(self.horizons)),key=lambda i:abs(self.horizons[i]-h))
        return u[...,idx]
    def calibration_loss(self,belief,realized_errors):
        return F.smooth_l1_loss(self(belief),realized_errors)

class RiskConstraintModel(nn.Module):
    """Risk is deliberately separate from epistemic model uncertainty."""
    def __init__(self,belief_dim:int,action_dim:int,hidden:int=128,constraints:int=4):
        super().__init__(); self.constraints=constraints
        self.net=nn.Sequential(nn.Linear(belief_dim+action_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,constraints))
    def logits(self,belief,action): return self.net(torch.cat([belief,action],-1))
    def probabilities(self,belief,action): return torch.sigmoid(self.logits(belief,action))
    def aggregate_risk(self,belief,action): return self.probabilities(belief,action).max(-1).values

class MultimodalWorldModel(nn.Module):
    """Action-conditioned mixture world model over compact latent/belief state."""
    def __init__(self,belief_dim:int,action_dim:int,hidden:int=256,components:int=5,horizons=(1,5,10,20,50)):
        super().__init__(); self.belief_dim=belief_dim; self.action_dim=action_dim; self.components=components
        self.trunk=nn.Sequential(nn.Linear(belief_dim+action_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.mean=nn.Linear(hidden,components*belief_dim); self.scale=nn.Linear(hidden,components*belief_dim); self.mix=nn.Linear(hidden,components)
        self.reward=nn.Linear(hidden,1); self.value=nn.Linear(hidden,1); self.cont=nn.Linear(hidden,1)
        self.state_value=nn.Sequential(nn.Linear(belief_dim,hidden),nn.SiLU(),nn.Linear(hidden,1))
        self.uncertainty=MonotoneHorizonUncertainty(belief_dim,horizons=horizons)
        self.risk=RiskConstraintModel(belief_dim,action_dim)
    def distribution(self,belief,action):
        x=self.trunk(torch.cat([belief,action],-1)); B=belief.shape[0]
        means=self.mean(x).view(B,self.components,self.belief_dim)
        scales=F.softplus(self.scale(x).view(B,self.components,self.belief_dim))+0.02
        return FutureDistribution(means,scales,self.mix(x)),x
    def imagine_step(self,belief,action,deterministic=False):
        dist,x=self.distribution(belief,action)
        nxt=dist.expected() if deterministic else dist.sample(1).squeeze(1)
        return {
            'belief':nxt,'distribution':dist,'reward':self.reward(x),'value':self.value(x),
            'continuation':torch.sigmoid(self.cont(x)),'risk':self.risk.aggregate_risk(belief,action),
        }
    def terminal_value(self, belief):
        return self.state_value(belief).squeeze(-1)

    def nll_loss(self,belief,action,next_belief,reduction='mean'):
        dist,_=self.distribution(belief,action)
        target=next_belief.unsqueeze(1)
        log_comp=(-0.5*(((target-dist.means)/dist.scales)**2+2*torch.log(dist.scales)+math.log(2*math.pi))).sum(-1)
        rows=-torch.logsumexp(torch.log_softmax(dist.logits,-1)+log_comp,-1)
        if reduction == 'none': return rows
        if reduction == 'sum': return rows.sum()
        if reduction != 'mean': raise ValueError("reduction must be 'none', 'sum' or 'mean'")
        return rows.mean()
