from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class ConfidenceInterval:
    mean: float; low: float; high: float; n: int
    def to_dict(self): return asdict(self)

def bootstrap_ci(values, *, confidence=.95, resamples=2000, seed=0):
    x=np.asarray(values,dtype=np.float64).reshape(-1)
    if len(x)==0: raise ValueError('values required')
    if len(x)==1: return ConfidenceInterval(float(x[0]),float(x[0]),float(x[0]),1)
    rng=np.random.default_rng(seed); idx=rng.integers(0,len(x),size=(int(resamples),len(x)))
    means=x[idx].mean(1); alpha=(1-float(confidence))/2
    return ConfidenceInterval(float(x.mean()),float(np.quantile(means,alpha)),float(np.quantile(means,1-alpha)),len(x))

def paired_bootstrap_ci(a,b,**kwargs):
    aa=np.asarray(a,dtype=np.float64); bb=np.asarray(b,dtype=np.float64)
    if aa.shape!=bb.shape: raise ValueError('paired arrays must match')
    return bootstrap_ci(aa-bb,**kwargs)
