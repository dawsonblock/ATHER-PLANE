from __future__ import annotations
from dataclasses import dataclass, asdict
import time

@dataclass
class ArenaRecord:
    planner:str
    score:float
    world_model_calls:int
    latency_ms:float
    budget:int
    requested_world_model_calls:int | None = None
    score_per_1k_calls: float | None = None

    def to_dict(self): return asdict(self)


def backend_budget_for_calls(planner, max_world_model_calls:int) -> int:
    """Translate an equal transition-call budget into backend-specific search budget."""
    calls=max(1,int(max_world_model_calls)); horizon=max(1,int(getattr(planner,'horizon',1)))
    name=getattr(planner,'name','')
    if name in {'icem','policy_icem','cem'}:
        iterations=max(1,int(getattr(planner,'iterations',1)))
        return max(1,calls//(horizon*iterations))
    if name in {'mppi','policy_mppi','gradient'}:
        return max(1,calls//horizon)
    if name=='beam':
        action_dim=max(1,int(getattr(planner,'action_dim',1)))
        return max(1,calls//(horizon*action_dim))
    return calls


def compare_planners(planners,belief,budgets=(32,128),actor=None,equal_world_model_calls:bool=False):
    rows=[]
    for planner in planners:
        for requested in budgets:
            budget=backend_budget_for_calls(planner,requested) if equal_world_model_calls else int(requested)
            t=time.perf_counter(); result=planner.plan(belief,actor=actor,budget=budget); dt=(time.perf_counter()-t)*1000
            efficiency=(1000.0*float(result.score)/max(1,int(result.world_model_calls))) if result.score==result.score else None
            rows.append(ArenaRecord(result.planner,result.score,result.world_model_calls,dt,budget,
                                    int(requested) if equal_world_model_calls else None,efficiency))
    return rows
