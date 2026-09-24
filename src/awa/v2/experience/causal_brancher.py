from __future__ import annotations
from dataclasses import dataclass, asdict
import copy, time
import numpy as np
from awa.v2.branching import capture_snapshot, restore_snapshot, verify_snapshot_roundtrip

@dataclass(frozen=True)
class CounterfactualBranch:
    branch_id:str
    action:list[float]
    reward:float
    done:bool
    observation:list[float]
    latency_ms:float
    def to_dict(self): return asdict(self)

@dataclass
class CounterfactualBatch:
    state_id:str
    branches:list[CounterfactualBranch]
    def consequence_spread(self):
        if len(self.branches)<2: return 0.0
        xs=np.asarray([b.observation for b in self.branches],dtype=np.float32)
        return float(np.mean(np.linalg.norm(xs[:,None]-xs[None,:],axis=-1)))


def branch_actions(env, actions, state_id='state', verify=True):
    actions=[np.asarray(a,dtype=np.float32) for a in actions]
    if not actions: raise ValueError('actions cannot be empty')
    if verify: verify_snapshot_roundtrip(env,actions[0])
    snap=capture_snapshot(env); rows=[]
    try:
        for i,a in enumerate(actions):
            restore_snapshot(env,snap); t=time.perf_counter(); obs,r,d,_=env.step(a); latency=(time.perf_counter()-t)*1000
            rows.append(CounterfactualBranch(f'{state_id}:{i}',a.reshape(-1).tolist(),float(r),bool(d),np.asarray(obs,dtype=np.float32).reshape(-1).tolist(),float(latency)))
    finally: restore_snapshot(env,snap)
    return CounterfactualBatch(str(state_id),rows)
