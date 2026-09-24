from __future__ import annotations
from dataclasses import dataclass,asdict
import numpy as np
from awa.v2.evaluation_stats import bootstrap_ci

@dataclass(frozen=True)
class SeedMetric:
    seed:int; success_rate:float; mean_return:float; planner_dependency:float; prediction_error:float=0.0
    def to_dict(self): return asdict(self)

@dataclass(frozen=True)
class MultiSeedGameReport:
    seeds:int; success_rate:dict; mean_return:dict; planner_dependency:dict; prediction_error:dict
    def to_dict(self): return asdict(self)

def summarize_seed_metrics(rows,*,confidence=.95,resamples=2000,seed=0):
    rows=list(rows)
    if not rows: raise ValueError('at least one seed metric is required')
    def ci(name): return bootstrap_ci([float(getattr(r,name)) for r in rows],confidence=confidence,resamples=resamples,seed=seed).to_dict()
    return MultiSeedGameReport(len(rows),ci('success_rate'),ci('mean_return'),ci('planner_dependency'),ci('prediction_error'))
