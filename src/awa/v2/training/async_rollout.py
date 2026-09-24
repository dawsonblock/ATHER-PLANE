from __future__ import annotations
from collections import deque
from .staleness import StalenessPolicy, RolloutEnvelope

class ExperienceQueue:
    """Dependency-free bounded queue abstraction used by async simulator workers.

    Actual process/thread orchestration is intentionally left to the deployment layer;
    this class centralizes version/staleness semantics so every backend behaves the same.
    """
    def __init__(self,capacity=100_000,staleness=None):
        self.q=deque(maxlen=int(capacity)); self.staleness=staleness or StalenessPolicy()
    def push(self,envelope:RolloutEnvelope): self.q.append(envelope)
    def pop_batch(self,n,current_policy_version):
        out=[]; weights=[]
        while self.q and len(out)<int(n):
            x=self.q.popleft(); w=self.staleness.weight(current_policy_version,x)
            if w<=0: continue
            out.append(x); weights.append(w)
        return out,weights
    def __len__(self): return len(self.q)
