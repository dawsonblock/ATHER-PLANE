from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from awa.planning.arbitrator import ReliabilityGate
from awa.planning.adaptive import AdaptivePlanningPolicy
from awa.safety.actions import ActionValidator, ContinuousActionValidator, ReflexLayer
from awa.actions import ActionSpec


class AgentEngine:
    def __init__(self, world_model, actor, uncertainty, planner, action_spec: ActionSpec, uncertainty_limit=1.5,
                 calibrator=None, learned_arbitrator=None, adaptive_policy: AdaptivePlanningPolicy | None=None,
                 learned_gate_threshold: float=0.5):
        self.world_model=world_model; self.actor=actor; self.uncertainty=uncertainty; self.planner=planner; self.action_spec=action_spec
        self.gate=ReliabilityGate(uncertainty_limit); self.reflex=ReflexLayer(); self.action_dim=action_spec.dim
        self.validator=ActionValidator(action_spec.dim) if action_spec.kind=="discrete" else ContinuousActionValidator(action_spec.low,action_spec.high)
        self.calibrator=calibrator; self.learned_arbitrator=learned_arbitrator; self.adaptive_policy=adaptive_policy
        self.learned_gate_threshold=float(learned_gate_threshold)

    @torch.no_grad()
    def _diagnostics(self, belief):
        if self.action_spec.kind=="discrete":
            proposed,_=self.actor.deterministic_action(belief.vector)
        else:
            proposed=self.actor.deterministic_action(belief.vector)
        _,u=self.uncertainty(belief.vector,proposed); raw=float(u.mean().item()); risk=float(self.calibrator(u).mean().item()) if self.calibrator is not None else raw/(1.0+raw)
        if self.action_spec.kind=="discrete":
            dist=self.actor.distribution(belief.vector); entropy=float(dist.entropy().mean().item()); vals=[]
            for i in range(self.action_dim):
                a=F.one_hot(torch.tensor([i],device=belief.vector.device),self.action_dim).float(); _,r,_,v,_,_=self.world_model.imagine(belief,a,step_index=0); vals.append(float((r+0.05*v).mean().item()))
        else:
            dist=self.actor.distribution(belief.vector); entropy=float(dist.entropy().mean().item()); vals=[]
            candidates=[self.actor.deterministic_action(belief.vector)]
            lo=torch.as_tensor(self.action_spec.low,device=belief.vector.device); hi=torch.as_tensor(self.action_spec.high,device=belief.vector.device)
            candidates += [lo.unsqueeze(0), hi.unsqueeze(0), torch.zeros(1,self.action_dim,device=belief.vector.device).clamp(lo,hi)]
            for a in candidates:
                _,r,_,v,_,_=self.world_model.imagine(belief,a,step_index=0); vals.append(float((r+0.05*v).mean().item()))
        value_spread=max(vals)-min(vals) if vals else 0.0
        return raw,risk,entropy,value_spread

    @torch.no_grad()
    def act(self, belief, observation=None, novelty: float=0.0):
        raw,risk,entropy,value_spread=self._diagnostics(belief); use_planner=self.planner is not None; reason="planner unavailable"; budget=None
        if use_planner and self.adaptive_policy is not None:
            budget=self.adaptive_policy.choose(risk); use_planner=budget.use_planner; reason=budget.reason
        elif use_planner:
            d=self.gate.choose(raw,True); use_planner=d.source=="planner"; reason=d.reason
        if use_planner and self.learned_arbitrator is not None:
            cost=(budget.candidates*budget.horizon if budget else self.planner.candidates*self.planner.horizon); norm_cost=cost/max(1.0,self.planner.candidates*self.planner.horizon)
            p=float(self.learned_arbitrator(risk,entropy,value_spread,planning_cost=norm_cost,novelty=novelty).mean().item()); use_planner=p>=self.learned_gate_threshold; reason=f"learned gate p={p:.3f}"
        if use_planner:
            h=None if budget is None else budget.horizon; c=None if budget is None else budget.candidates; model_action,env_action,_=self.planner.plan(belief,horizon=h,candidates=c)
        elif self.action_spec.kind=="discrete":
            model_action,idx_t=self.actor.sample_onehot(belief.vector); env_action=int(idx_t.item())
        else:
            model_action,_,_=self.actor.sample_with_log_prob(belief.vector); env_action=model_action.squeeze(0).cpu().numpy()
        env_action=self.validator.validate(env_action); env_action=self.reflex.override(observation,env_action)
        if self.action_spec.kind=="continuous":
            model_action=torch.as_tensor(np.asarray(env_action),dtype=torch.float32,device=belief.vector.device).reshape(1,-1)
        else:
            model_action=F.one_hot(torch.tensor([env_action],device=belief.vector.device),self.action_dim).float()
        return model_action,env_action,{"source":"planner" if use_planner else "actor","uncertainty_raw":raw,"uncertainty_risk":risk,"actor_entropy":entropy,"value_spread":value_spread,"reason":reason}
