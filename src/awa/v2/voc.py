from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
from torch import nn

@dataclass
class ComputeChoice:
    planner: str
    budget: int
    expected_gain: float
    expected_cost: float
    voc: float
    def to_dict(self): return asdict(self)

class ValueOfComputation(nn.Module):
    """Predict expected planner gain and subtract an explicit learned compute cost."""
    def __init__(self,feature_dim:int,choices:list[tuple[str,int]],hidden:int=128):
        super().__init__(); self.choices=list(choices)
        if not self.choices: raise ValueError('choices must not be empty')
        self.net=nn.Sequential(nn.Linear(feature_dim,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,len(choices)))
        self.log_cost_scale=nn.Parameter(torch.tensor(-3.0))
    def gains(self,features): return self.net(features)
    def utilities(self,features,costs=None):
        gains=self.gains(features)
        if costs is None:
            base=torch.tensor([max(0,b) for _,b in self.choices],device=features.device,dtype=features.dtype)
            denom=base.max().clamp_min(1.0); costs=(base/denom).unsqueeze(0).expand(gains.shape[0],-1)
        elif costs.ndim==1:
            costs=costs.unsqueeze(0).expand(gains.shape[0],-1)
        lam=torch.exp(self.log_cost_scale)
        return gains-lam*costs,gains,lam*costs
    def choose(self,features,costs=None):
        voc,gains,penalty=self.utilities(features,costs); idx=voc.argmax(-1); out=[]
        for row,i in enumerate(idx.tolist()):
            name,budget=self.choices[i]
            out.append(ComputeChoice(name,budget,float(gains[row,i].detach()),float(penalty[row,i].detach()),float(voc[row,i].detach())))
        return out,voc
    def supervised_loss(self,features,measured_gains,costs=None,target_cost_weight:float|None=None):
        pred=self.gains(features)
        loss=torch.nn.functional.smooth_l1_loss(pred,measured_gains)
        if costs is not None and target_cost_weight is not None:
            if costs.ndim==1: costs=costs.unsqueeze(0).expand_as(pred)
            pred_u=pred-torch.exp(self.log_cost_scale)*costs
            target_u=measured_gains-float(target_cost_weight)*costs
            loss=loss+torch.nn.functional.smooth_l1_loss(pred_u,target_u)
        return loss

@dataclass
class VOCFitReport:
    final_loss: float
    gain_mae: float
    utility_mae: float
    choice_accuracy: float
    learned_cost_scale: float
    epochs: int
    def to_dict(self): return asdict(self)


def fit_value_of_computation(model:ValueOfComputation,features,measured_gains,costs,
                             epochs:int=250,lr:float=3e-3,cost_weight:float=1.0):
    """Fit measured branch gains and cost-adjusted utility from exact branch data."""
    features=torch.as_tensor(features,dtype=torch.float32)
    measured_gains=torch.as_tensor(measured_gains,dtype=torch.float32)
    costs=torch.as_tensor(costs,dtype=torch.float32)
    if features.ndim!=2 or measured_gains.ndim!=2 or costs.shape!=measured_gains.shape:
        raise ValueError('expected features [N,F], gains/costs [N,C]')
    if measured_gains.shape[1]!=len(model.choices): raise ValueError('choice count mismatch')
    opt=torch.optim.Adam(model.parameters(),lr=float(lr)); last=None
    for _ in range(int(epochs)):
        last=model.supervised_loss(features,measured_gains,costs,target_cost_weight=cost_weight)
        opt.zero_grad(set_to_none=True); last.backward(); opt.step()
    with torch.no_grad():
        util,pred_gain,_=model.utilities(features,costs)
        target_util=measured_gains-float(cost_weight)*costs
        gain_mae=(pred_gain-measured_gains).abs().mean()
        utility_mae=(util-target_util).abs().mean()
        acc=(util.argmax(-1)==target_util.argmax(-1)).float().mean()
    return VOCFitReport(float(last.detach()),float(gain_mae),float(utility_mae),float(acc),float(torch.exp(model.log_cost_scale).detach()),int(epochs))
