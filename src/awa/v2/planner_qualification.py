from __future__ import annotations
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn

from awa.v2.branching import collect_planner_benefits
from awa.v2.voc import ValueOfComputation, fit_value_of_computation


class _ZeroUncertainty:
    def at_horizon(self,belief,h): return torch.zeros(belief.shape[0],device=belief.device)

class _ZeroRisk:
    def aggregate_risk(self,belief,action): return torch.zeros(belief.shape[0],device=belief.device)

class PointOracleWorld(nn.Module):
    """Exact differentiable model of ContinuousPointEnv for planner infrastructure qualification.

    It is deliberately a benchmark oracle, not a learned-model performance claim.
    """
    def __init__(self,step_scale=.15,success_radius=.10):
        super().__init__(); self.step_scale=float(step_scale); self.success_radius=float(success_radius)
        self.uncertainty=_ZeroUncertainty(); self.risk=_ZeroRisk()
    def imagine_step(self,belief,action,deterministic=True):
        pos=belief[:,:2]; target=belief[:,2:4]; a=action.clamp(-1,1)
        before=torch.linalg.vector_norm(target-pos,dim=-1)
        npos=(pos+self.step_scale*a).clamp(-1.25,1.25)
        after=torch.linalg.vector_norm(target-npos,dim=-1)
        progress=before-after
        reward=2.0*progress-.01*a.pow(2).sum(-1)
        success=after<=self.success_radius
        reward=reward+success.float()
        nxt=torch.cat([npos,target],-1)
        value=(-after).unsqueeze(-1)
        return {'belief':nxt,'reward':reward.unsqueeze(-1),'value':value,
                'continuation':(~success).float().unsqueeze(-1),'risk':torch.zeros_like(after)}


class PointHeuristicActor(nn.Module):
    """Fast imperfect policy used as the no-search control in the planner smoke benchmark."""
    def __init__(self,gain=.55): super().__init__(); self.gain=float(gain)
    def deterministic_action(self,belief):
        delta=belief[:,2:4]-belief[:,:2]
        return (self.gain*delta/.15).clamp(-1,1)


@dataclass
class PlannerQualificationReport:
    samples: int
    branch_horizon: int
    summary: dict
    voc: dict | None
    def to_dict(self): return asdict(self)


def identity_encoder(obs):
    return torch.as_tensor(obs,dtype=torch.float32).reshape(1,-1)


def qualify_planners(env_factory,world,actor,planner_choices,encode_observation=identity_encoder,
                     episodes=4,max_states=64,branch_horizon=4,cost_per_1k_calls=.01,
                     fit_voc=True,voc_epochs=300):
    table=collect_planner_benefits(env_factory,encode_observation,world,actor,planner_choices,
                                   episodes=episodes,max_states=max_states,branch_horizon=branch_horizon,
                                   cost_per_1k_calls=cost_per_1k_calls)
    voc_report=None
    if fit_voc and len(table):
        x,g,c,_=table.matrix()
        model=ValueOfComputation(x.shape[1],table.choices,hidden=64)
        voc_report=fit_value_of_computation(model,x,g,c,epochs=voc_epochs,cost_weight=1.0).to_dict()
    return PlannerQualificationReport(len(table.records),int(branch_horizon),table.summary(),voc_report),table


def write_planner_report(report,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(report.to_dict(),indent=2,sort_keys=True),encoding='utf-8')
    return path
