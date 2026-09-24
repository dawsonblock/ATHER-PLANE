from __future__ import annotations

from dataclasses import dataclass, asdict
import copy, time
import numpy as np
import torch

from awa.v2.branching import capture_snapshot, restore_snapshot, default_voc_features
from awa.v2.voc import ValueOfComputation, fit_value_of_computation
from .procedural_arena import ProceduralArenaEnv, ACTION_DIM


def _clone_temporal(state):
    return type(state)(state.local.clone(), state.global_ctx.clone(), state.event_buffer.clone())


@dataclass(frozen=True)
class GameVOCReport:
    samples: int
    choices: list[tuple[str,int]]
    fit: dict
    mean_actor_return: float
    mean_best_gain: float
    mean_world_calls: float
    def to_dict(self): return asdict(self)


@torch.no_grad()
def _real_branch(env, snapshot, encoder, temporal, belief, actor, planner=None, budget=0, horizon=4, gamma=.99):
    restore_snapshot(env,snapshot)
    temp=_clone_temporal(temporal); b=belief.clone(); total=0.0; disc=1.0; calls=0; done=False; step=0
    t0=time.perf_counter()
    for j in range(int(horizon)):
        if planner is None:
            action=actor.deterministic_action(b)
            if isinstance(action,tuple): action=action[0]
        else:
            res=planner.plan(b,actor=actor,budget=int(budget)); action=res.action; calls+=int(res.world_model_calls)
        arr=action.squeeze(0).detach().cpu().numpy()
        nxt,r,done,_=env.step(arr); total+=disc*float(r); step+=1
        if done: break
        temp,b=encoder.observe(temp,nxt,action,env.goal_vector(),j+1)
        disc*=float(gamma)
    restore_snapshot(env,snapshot)
    return float(total),calls,(time.perf_counter()-t0)*1000.0,step


def fit_game_voc(
    tasks,
    encoder,
    actor_baseline,
    world,
    planners: dict[str,object],
    *,
    choices=(('actor',0),('policy_mppi',16),('policy_mppi',48)),
    states_per_task=4,
    branch_horizon=4,
    horizon=100,
    seed_offset=0,
    epochs=200,
    cost_weight=1.0,
):
    """Train VOC from exact real-environment branches at identical game states."""
    device=next(encoder.parameters()).device; actor=actor_baseline.actor
    features=[]; gains=[]; costs=[]; actor_returns=[]; calls_all=[]
    for ti,task in enumerate(tasks):
        env=ProceduralArenaEnv(task,horizon=horizon); obs=env.reset(seed=task.seed+seed_offset+ti*1009)
        temporal=encoder.initial(1,device); prev=torch.zeros(1,ACTION_DIM,device=device); step=0; done=False; sampled=0
        while not done and sampled<int(states_per_task):
            temporal,belief=encoder.observe(temporal,obs,prev,env.goal_vector(),step)
            aa=actor.deterministic_action(belief)
            features.append(default_voc_features(world,belief,aa))
            snap=capture_snapshot(env); base_return,_,base_ms,_=_real_branch(env,snap,encoder,temporal,belief,actor,None,0,branch_horizon)
            actor_returns.append(base_return); row_gain=[]; row_cost=[]
            for name,budget in choices:
                if name=='actor':
                    ret=base_return; c=0; ms=base_ms
                else:
                    planner=planners[name]; ret,c,ms,_=_real_branch(env,snap,encoder,temporal,belief,actor,planner,budget,branch_horizon)
                row_gain.append(float(ret-base_return)); row_cost.append(float(c/1000.0 + ms/1000.0)); calls_all.append(c)
            gains.append(row_gain); costs.append(row_cost)
            # advance real episode with actor only
            arr=aa.squeeze(0).cpu().numpy(); obs,_,done,_=env.step(arr); prev=aa; step+=1; sampled+=1
    if not features: raise ValueError('VOC qualification produced no states')
    model=ValueOfComputation(len(features[0]),list(choices)).to(device)
    fit=fit_value_of_computation(model,features,gains,costs,epochs=int(epochs),cost_weight=float(cost_weight))
    report=GameVOCReport(
        len(features),list(choices),fit.to_dict(),float(np.mean(actor_returns)),
        float(np.mean(np.max(np.asarray(gains,dtype=np.float32),axis=1))),float(np.mean(calls_all)) if calls_all else 0.0,
    )
    return model,report
