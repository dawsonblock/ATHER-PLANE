from __future__ import annotations
import copy
import torch
from torch import nn
import torch.nn.functional as F


class RunningTargetScale(nn.Module):
    """Tracks return scale while keeping critic outputs in raw reward units."""
    def __init__(self, momentum: float = 0.01, eps: float = 1e-4):
        super().__init__()
        self.momentum=float(momentum); self.eps=float(eps)
        self.register_buffer('mean', torch.tensor(0.0))
        self.register_buffer('var', torch.tensor(1.0))
        self.register_buffer('initialized', torch.tensor(False))

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x=x.detach().float().reshape(-1)
        if x.numel()==0: return
        m=x.mean(); v=x.var(unbiased=False).clamp_min(self.eps)
        if not bool(self.initialized.item()):
            self.mean.copy_(m); self.var.copy_(v); self.initialized.fill_(True)
        else:
            a=self.momentum; self.mean.lerp_(m,a); self.var.lerp_(v,a)

    @property
    def std(self):
        return self.var.sqrt().clamp_min(self.eps)

    def scaled_huber(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.smooth_l1_loss((prediction-target.detach())/self.std, torch.zeros_like(prediction))


class _ActionBounds(nn.Module):
    def __init__(self, action_dim: int, low=None, high=None):
        super().__init__(); self.action_dim=int(action_dim)
        lo=torch.full((self.action_dim,),-1.0) if low is None else torch.as_tensor(low,dtype=torch.float32).reshape(-1)
        hi=torch.full((self.action_dim,),1.0) if high is None else torch.as_tensor(high,dtype=torch.float32).reshape(-1)
        if lo.numel()!=self.action_dim or hi.numel()!=self.action_dim or torch.any(hi<=lo):
            raise ValueError('invalid critic action bounds')
        self.register_buffer('low',lo); self.register_buffer('high',hi)

    def _norm_action(self, action):
        return (2.0*(action-self.low)/(self.high-self.low)-1.0).clamp(-1.25,1.25)


class TwinQCritic(_ActionBounds):
    """Twin scalar state-action critic for bounded continuous actions."""
    distributional=False
    def __init__(self, belief_dim: int, action_dim: int, hidden_dim: int = 256, low=None, high=None):
        super().__init__(action_dim,low,high)
        def net(): return nn.Sequential(nn.Linear(belief_dim+self.action_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,1))
        self.q1=net(); self.q2=net()

    def forward(self, belief_vec: torch.Tensor, action: torch.Tensor):
        x=torch.cat([belief_vec,self._norm_action(action)],dim=-1)
        return self.q1(x).squeeze(-1),self.q2(x).squeeze(-1)

    def minimum(self, belief_vec: torch.Tensor, action: torch.Tensor):
        q1,q2=self(belief_vec,action); return torch.minimum(q1,q2)


class QuantileTwinQCritic(_ActionBounds):
    """Twin quantile critic.

    Quantiles preserve multimodal/heteroscedastic return information while
    ``minimum`` exposes a conservative scalar value for actor optimization and
    planning diagnostics.
    """
    distributional=True
    def __init__(self, belief_dim: int, action_dim: int, hidden_dim: int = 256,
                 quantiles: int = 32, low=None, high=None):
        super().__init__(action_dim,low,high)
        self.num_quantiles=int(quantiles)
        if self.num_quantiles < 4: raise ValueError('quantiles must be >= 4')
        self.register_buffer('taus',(torch.arange(self.num_quantiles,dtype=torch.float32)+0.5)/self.num_quantiles)
        def net(): return nn.Sequential(nn.Linear(belief_dim+self.action_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,self.num_quantiles))
        self.q1=net(); self.q2=net()

    def forward(self, belief_vec: torch.Tensor, action: torch.Tensor):
        x=torch.cat([belief_vec,self._norm_action(action)],dim=-1)
        return self.q1(x),self.q2(x)

    def minimum(self, belief_vec: torch.Tensor, action: torch.Tensor):
        q1,q2=self(belief_vec,action)
        return torch.minimum(q1.mean(-1),q2.mean(-1))

    def conservative_quantiles(self, belief_vec: torch.Tensor, action: torch.Tensor):
        q1,q2=self(belief_vec,action)
        choose=(q1.mean(-1,keepdim=True)<=q2.mean(-1,keepdim=True))
        return torch.where(choose,q1,q2)


def quantile_huber_loss(prediction: torch.Tensor, target_samples: torch.Tensor, taus: torch.Tensor, kappa: float = 1.0):
    """QR-DQN style pairwise quantile Huber loss.

    prediction: [B,N]
    target_samples: [B,M] or [B,1]
    taus: [N]
    """
    if target_samples.ndim==1: target_samples=target_samples[:,None]
    delta=target_samples[:,None,:]-prediction[:,:,None]
    abs_delta=delta.abs()
    huber=torch.where(abs_delta<=kappa,0.5*delta.pow(2),kappa*(abs_delta-0.5*kappa))
    weight=(taus.view(1,-1,1)-(delta.detach()<0).float()).abs()
    return (weight*huber/max(kappa,1e-8)).mean()


class TargetTwinQCritic(nn.Module):
    def __init__(self, critic: nn.Module):
        super().__init__(); self.net=copy.deepcopy(critic)
        for p in self.net.parameters(): p.requires_grad=False

    @torch.no_grad()
    def update(self, critic: nn.Module, tau: float=0.01):
        for t,s in zip(self.net.parameters(),critic.parameters()): t.data.lerp_(s.data,float(tau))
        # buffers such as quantile locations and action bounds are immutable and
        # already copied; running stats are not stored in critic modules.

    def forward(self, belief_vec, action): return self.net(belief_vec,action)
    def minimum(self, belief_vec, action): return self.net.minimum(belief_vec,action)
    def conservative_quantiles(self, belief_vec, action):
        if not hasattr(self.net,'conservative_quantiles'):
            return self.minimum(belief_vec,action).unsqueeze(-1)
        return self.net.conservative_quantiles(belief_vec,action)
