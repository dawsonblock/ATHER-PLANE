from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import OrderedDict
from pathlib import Path
import numpy as np

from awa.v2.training import ReusableLearningEngine
from .goal_program import GOAL_DIM


@dataclass(frozen=True)
class PrioritizedReplayExportReport:
    transitions: int
    episodes: int
    mean_weight: float
    max_weight: float
    path: str
    def to_dict(self): return asdict(self)


def export_prioritized_game_replay(engine: ReusableLearningEngine, path: str | Path) -> PrioritizedReplayExportReport:
    """Export physically contiguous game replay with structural priorities as SGD weights.

    Hindsight rows with incompatible spatial-only goals are intentionally excluded;
    objective-aware game HER can be added separately without corrupting world dynamics.
    """
    pairs=[(r,float(p),i) for i,(r,p) in enumerate(zip(engine.replay.items,engine.replay.priorities)) if (not bool((r.metadata or {}).get("hindsight",False)) or bool((r.metadata or {}).get("game_hindsight",False)))]
    if not pairs: raise ValueError("no physical game replay rows to export")
    groups=OrderedDict()
    for r,p,i in pairs:
        md=r.metadata or {}; eid=str(md.get("episode_id",f"ungrouped:{i}"))
        groups.setdefault(eid,[]).append((r,p,i))
    ordered=[]
    for eid,rows in groups.items():
        rows=sorted(rows,key=lambda x:(int((x[0].metadata or {}).get("step",x[2])),x[2]))
        ordered.extend(rows)
    obs=[]; goals=[]; acts=[]; rews=[]; nxt=[]; ng=[]; dones=[]; weights=[]; epids=[]; constraints=[]; have_constraints=True
    for i,(r,p,_) in enumerate(ordered):
        g=np.asarray(r.goal,dtype=np.float32).reshape(-1)
        if g.shape!=(GOAL_DIM,):
            continue
        md=r.metadata or {}; eid=str(md.get("episode_id",f"row:{i}"))
        same_next=False; next_goal=g
        md_next_goal=(r.metadata or {}).get("next_goal")
        if md_next_goal is not None:
            candidate=np.asarray(md_next_goal,dtype=np.float32).reshape(-1)
            if candidate.shape==(GOAL_DIM,): next_goal=candidate
        if i+1<len(ordered):
            nr=ordered[i+1][0]; nmd=nr.metadata or {}
            same_next=str(nmd.get("episode_id",""))==eid
            if same_next:
                candidate=np.asarray(nr.goal,dtype=np.float32).reshape(-1)
                if candidate.shape==(GOAL_DIM,): next_goal=candidate
        done=bool(md.get("done",False) or not same_next)
        obs.append(np.asarray(r.state,dtype=np.float32).reshape(-1)); goals.append(g)
        acts.append(np.asarray(r.action,dtype=np.float32).reshape(-1)); rews.append(float(r.reward))
        nxt.append(np.asarray(r.actual_next,dtype=np.float32).reshape(-1)); ng.append(next_goal)
        dones.append(done); weights.append(max(1e-4,p)); epids.append(eid)
        raw_constraints=md.get("constraint_labels")
        if raw_constraints is None:
            have_constraints=False; constraints.append(None)
        else:
            constraints.append(np.asarray(raw_constraints,dtype=np.float32).reshape(-1))
    if not obs: raise ValueError("replay contained no compatible goal-conditioned game transitions")
    weights=np.asarray(weights,dtype=np.float32); weights=weights/float(weights.mean())
    target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(
        observations=np.asarray(obs,np.float32), goals=np.asarray(goals,np.float32),
        actions=np.asarray(acts,np.float32), rewards=np.asarray(rews,np.float32),
        next_observations=np.asarray(nxt,np.float32), next_goals=np.asarray(ng,np.float32),
        dones=np.asarray(dones,np.bool_), sample_weights=weights,
        episode_ids=np.asarray(epids),
    )
    if have_constraints and constraints:
        shapes={tuple(np.asarray(x).shape) for x in constraints}
        if len(shapes)==1:
            payload["constraints"]=np.asarray(constraints,np.float32)
    np.savez_compressed(target,**payload)
    return PrioritizedReplayExportReport(len(obs),len(groups),float(weights.mean()),float(weights.max()),str(target))
