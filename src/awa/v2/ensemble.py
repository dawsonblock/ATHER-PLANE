from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass(frozen=True)
class EnsembleUncertainty:
    next_state: torch.Tensor
    reward: torch.Tensor
    risk: torch.Tensor

class WorldModelEnsemble(nn.Module):
    """Small ensemble wrapper for epistemic uncertainty and robust planning.

    Members are ordinary Aether world models. Aleatoric uncertainty remains inside
    each member's mixture distribution; disagreement across members is treated as
    epistemic uncertainty.
    """
    def __init__(self, members):
        super().__init__()
        members=list(members)
        if len(members)<2:
            raise ValueError("ensemble requires at least two world models")
        dims={(int(m.belief_dim),int(m.action_dim)) for m in members}
        if len(dims)!=1:
            raise ValueError("ensemble members must share belief/action dimensions")
        self.members=nn.ModuleList(members)
        self.belief_dim,self.action_dim=next(iter(dims))

    @torch.no_grad()
    def disagreement(self, belief, action) -> EnsembleUncertainty:
        nxt=[]; rew=[]; risk=[]
        for m in self.members:
            out=m.imagine_step(belief,action,deterministic=True)
            nxt.append(out['belief']); rew.append(out['reward'].squeeze(-1)); risk.append(out['risk'])
        ns=torch.stack(nxt,0); rs=torch.stack(rew,0); ks=torch.stack(risk,0)
        return EnsembleUncertainty(ns.var(0,unbiased=False).mean(-1),rs.var(0,unbiased=False),ks.var(0,unbiased=False))


    @torch.no_grad()
    def imagine_step(self, belief, action, deterministic=False):
        # Mean prediction in the shared belief space; uncertainty remains available
        # separately through disagreement().
        rows=[m.imagine_step(belief,action,deterministic=deterministic) for m in self.members]
        return {
            'belief': torch.stack([r['belief'] for r in rows]).mean(0),
            'reward': torch.stack([r['reward'] for r in rows]).mean(0),
            'continuation': torch.stack([r['continuation'] for r in rows]).mean(0),
            'risk': torch.stack([r['risk'] for r in rows]).max(0).values,
            'epistemic': self.disagreement(belief,action).next_state,
        }

    @torch.no_grad()
    def terminal_value(self, belief):
        return torch.stack([m.terminal_value(belief) for m in self.members]).mean(0)

    @torch.no_grad()
    def aggregate_prediction(self, belief, action):
        rows=[m.imagine_step(belief,action,deterministic=True) for m in self.members]
        return {
            'belief': torch.stack([r['belief'] for r in rows]).mean(0),
            'reward': torch.stack([r['reward'] for r in rows]).mean(0),
            'continuation': torch.stack([r['continuation'] for r in rows]).mean(0),
            'risk': torch.stack([r['risk'] for r in rows]).max(0).values,
            'epistemic': self.disagreement(belief,action).next_state,
        }
