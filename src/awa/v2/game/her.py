from __future__ import annotations
import numpy as np
from awa.v2.experience import ExperienceRecord
from .goal_program import GOAL_DIM, NAME_TO_CODE


def _reach_goal_vector() -> np.ndarray:
    g=np.zeros(GOAL_DIM,dtype=np.float32); g[int(NAME_TO_CODE['reach_goal'])]=1.0; g[-3]=0.0; g[-2]=.25; g[-1]=1.0
    return g


def augment_game_hindsight(engine, records, *, k_future: int = 2, seed: int = 0):
    """Goal-relabel failed game transitions into valid one-step reach-goal examples.

    Observation goal coordinates are rewritten along with the explicit goal vector,
    so these rows remain semantically consistent for actor training. They are marked
    as one-step episodes and should not be used for multi-step world-model sequences.
    """
    if not records: return []
    rng=np.random.default_rng(seed); out=[]; n=len(records)
    for i,base in enumerate(records):
        candidates=np.arange(i,n)
        if len(candidates)>int(k_future): candidates=rng.choice(candidates,size=int(k_future),replace=False)
        for j in sorted(map(int,np.atleast_1d(candidates))):
            achieved=np.asarray(records[j].actual_next,dtype=np.float32).reshape(-1)[:2]
            s=np.asarray(base.state,dtype=np.float32).copy().reshape(-1)
            ns=np.asarray(base.actual_next,dtype=np.float32).copy().reshape(-1)
            if s.size<11 or ns.size<11: continue
            s[8:10]=achieved; s[10]=1.0; ns[8:10]=achieved; ns[10]=1.0
            d0=float(np.linalg.norm(s[:2]-achieved)); d1=float(np.linalg.norm(ns[:2]-achieved))
            success=d1<=.10; reward=1.5*(d0-d1)+(2.2 if success else 0.0)
            md=dict(base.metadata or {}); md.update({
                'hindsight':True,'game_hindsight':True,'done':True,
                'episode_id':f"her:{md.get('episode_id','ep')}:{i}:{j}",'step':0,'source_future_index':j,
            })
            clone=ExperienceRecord(
                s,_reach_goal_vector(),np.asarray(base.action,dtype=np.float32),
                ns.copy(),ns,float(reward),bool(success),0.0,float(base.novelty),
                float(base.uncertainty),float(base.risk),float(base.td_error),
                float(base.task_importance),float(base.information_value),False,md,
            )
            engine.replay.add(clone); out.append(clone)
    return out
