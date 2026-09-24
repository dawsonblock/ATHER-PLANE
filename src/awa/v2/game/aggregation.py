from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from .loop import ReusableGameLoop

@dataclass(frozen=True)
class AggregationRound:
    round_index:int; episodes:int; success_rate:float; mean_return:float
    planner_dependency:float; replay_size:int
    def to_dict(self): return asdict(self)

class PlannerAssistedMixturePolicy:
    """Choose planner teacher on a controlled fraction of decisions.

    Actor and planner objects may implement ``act(env, obs)`` or be plain callables.
    Returned tuples preserve the used-planner bit expected by ReusableGameLoop.
    """
    def __init__(self,actor,planner,planner_probability=.5,seed=0):
        self.actor=actor; self.planner=planner; self.p=float(np.clip(planner_probability,0,1)); self.rng=np.random.default_rng(seed)
    def act(self,env,obs):
        use=self.planner is not None and self.rng.random()<self.p
        source=self.planner if use else self.actor
        out=source.act(env,obs) if hasattr(source,'act') else source(obs)
        if isinstance(out,tuple): action=out[0]
        else: action=out
        return action,bool(use)

class IterativeDataAggregator:
    """DAgger-like actor→planner→retrain orchestration around ReusableGameLoop.

    A caller-supplied retrain_fn receives (round_index, engine) and may return an
    updated actor/planner pair. This keeps the orchestration independent of model
    size while ensuring every round adds on-policy experience to the same replay.
    """
    def __init__(self,loop:ReusableGameLoop|None=None,seed=0): self.loop=loop or ReusableGameLoop(); self.seed=int(seed)
    def run(self,tasks,actor,planner=None,*,rounds=3,episodes_per_task=1,planner_schedule=None,retrain_fn=None,counterfactual_branches=0):
        reports=[]; current_actor=actor; current_planner=planner
        for r in range(int(rounds)):
            p=float(planner_schedule(r) if planner_schedule is not None else max(.1,.7-.2*r))
            policy=PlannerAssistedMixturePolicy(current_actor,current_planner,p,self.seed+r)
            rep=self.loop.run_tasks(list(tasks),episodes_per_task=episodes_per_task,policy=policy,counterfactual_branches=counterfactual_branches)
            # planner dependency is represented by the requested mixture probability;
            # per-transition planner bits are retained in the engine episode summaries.
            reports.append(AggregationRound(r,rep.episodes,rep.success_rate,rep.mean_return,p,rep.replay_size))
            if retrain_fn is not None:
                updated=retrain_fn(r,self.loop.engine)
                if updated is not None:
                    if isinstance(updated,tuple): current_actor,current_planner=updated
                    else: current_actor=updated
        return reports,current_actor,current_planner
