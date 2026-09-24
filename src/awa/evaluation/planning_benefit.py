from __future__ import annotations
from dataclasses import dataclass
import copy
import math
import numpy as np
import torch

from awa.environments.factory import make_environment
from awa.planning.arbitrator_train import train_arbitrator_step


@dataclass
class PlanningBenefitData:
    diagnostics: torch.Tensor
    planner_return: torch.Tensor
    actor_return: torch.Tensor
    planning_cost: torch.Tensor
    branch_horizon: int = 1

    def __len__(self): return int(self.planner_return.numel())


def _clone_belief(belief):
    return type(belief)(
        belief.deterministic.clone(),
        belief.stochastic.clone(),
        None if belief.slow is None else belief.slow.clone(),
    )


def _actor_choice(components, belief):
    if components.action_spec.kind=='discrete':
        model,idx=components.actor.deterministic_action(belief.vector)
        return model,int(idx.item())
    model=components.actor.deterministic_action(belief.vector)
    return model,model.squeeze(0).detach().cpu().numpy()


def _planner_choice(components, belief):
    model,env_action,_=components.planner.plan(belief)
    return model,env_action


@torch.no_grad()
def _real_branch_return(env, components, belief, obs, mode: str, horizon: int,
                        discount: float, step_index: int):
    b=_clone_belief(belief); current_obs=np.asarray(obs).copy(); total=0.0; disc=1.0; steps=0
    for j in range(int(horizon)):
        model_action,env_action=(_planner_choice(components,b) if mode=='planner' else _actor_choice(components,b))
        next_obs,reward,done,_=env.step(env_action)
        total += disc*float(reward); steps += 1
        if done: break
        o=torch.as_tensor(next_obs,dtype=torch.float32,device=b.deterministic.device).unsqueeze(0)
        b=components.world_model.observe(b,model_action,o,step_index=step_index+j+1).belief
        current_obs=next_obs; disc*=float(discount)
    return total,steps


@torch.no_grad()
def collect_planning_benefit(cfg, components, device, episodes: int=5, max_samples: int=512,
                             branch_horizon: int=3, discount: float | None=None,
                             planning_cost_per_1k: float=0.0):
    """Measure planner-vs-actor return from identical real environment states.

    Each branch starts from an exact environment snapshot and the same posterior
    belief. Unlike the v1.7 one-step collector, this can measure delayed planner
    benefit over multiple real transitions. The main dataset trajectory still
    follows the deterministic actor so branch probes do not alter collection.
    """
    if components.planner is None: raise ValueError('planner must be enabled')
    branch_horizon=max(1,int(branch_horizon)); discount=float(cfg.get('training',{}).get('discount',0.99) if discount is None else discount)
    rows=[]; pr=[]; ar=[]; costs=[]
    for ep in range(int(episodes)):
        env=make_environment(cfg,cfg.get('seed',0)+12000+ep)
        if not hasattr(env,'state_dict') or not hasattr(env,'load_state_dict'):
            raise TypeError('planning-benefit collection requires state_dict/load_state_dict environment support')
        obs=env.reset(); belief=components.world_model.initial_belief(1,device); prev=components.action_spec.zero(1,device); done=False; t=0
        while not done and len(rows)<max_samples:
            o=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
            post=components.world_model.observe(belief,prev,o,step_index=t); belief=post.belief
            raw,risk,entropy,spread=components.engine._diagnostics(belief)
            snapshot=copy.deepcopy(env.state_dict())

            env.load_state_dict(copy.deepcopy(snapshot))
            actor_return,actor_steps=_real_branch_return(env,components,belief,obs,'actor',branch_horizon,discount,t)
            env.load_state_dict(copy.deepcopy(snapshot))
            planner_return,planner_steps=_real_branch_return(env,components,belief,obs,'planner',branch_horizon,discount,t)
            env.load_state_dict(copy.deepcopy(snapshot))

            rollout_ops=max(1,int(getattr(components.planner,'candidates',1))*int(getattr(components.planner,'horizon',1))*planner_steps)
            normalized_cost=math.log1p(rollout_ops)/math.log1p(100_000.0)
            return_cost=float(planning_cost_per_1k)*(rollout_ops/1000.0)
            novelty=0.0
            rows.append([risk,entropy,spread,normalized_cost,novelty])
            ar.append(float(actor_return)); pr.append(float(planner_return)); costs.append(return_cost)

            model_action,actor_env=_actor_choice(components,belief)
            obs,reward,done,_=env.step(actor_env); prev=model_action; t+=1
        close=getattr(env,'close',None)
        if close is not None: close()
        if len(rows)>=max_samples: break
    return PlanningBenefitData(
        torch.tensor(rows,dtype=torch.float32),torch.tensor(pr),torch.tensor(ar),torch.tensor(costs),branch_horizon
    )


@torch.no_grad()
def collect_one_step_planning_benefit(cfg, components, device, episodes: int=5, max_samples: int=512):
    return collect_planning_benefit(cfg,components,device,episodes,max_samples,branch_horizon=1)


def fit_arbitrator_dataset(arbitrator, data: PlanningBenefitData, epochs: int=100, lr: float=1e-2, margin: float=0.0):
    if len(data)==0: raise ValueError('empty planning-benefit dataset')
    opt=torch.optim.Adam(arbitrator.parameters(),lr=float(lr)); last=None
    for _ in range(int(epochs)):
        last=train_arbitrator_step(arbitrator,opt,data.diagnostics,data.planner_return,data.actor_return,data.planning_cost,margin)
    return last
