from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class PlanningBudget:
    horizon: int
    candidates: int
    use_planner: bool
    reason: str


class AdaptivePlanningPolicy:
    """Maps calibrated uncertainty to bounded planning compute."""
    def __init__(self, base_horizon: int, base_candidates: int,
                 low: float = 0.25, medium: float = 0.60, high: float = 0.85,
                 min_horizon: int = 2, min_candidates: int = 16):
        if not (0 <= low <= medium <= high <= 1):
            raise ValueError("thresholds must satisfy 0 <= low <= medium <= high <= 1")
        self.base_horizon=base_horizon; self.base_candidates=base_candidates
        self.low=low; self.medium=medium; self.high=high
        self.min_horizon=min_horizon; self.min_candidates=min_candidates

    def choose(self, risk: float) -> PlanningBudget:
        r=float(max(0.0,min(1.0,risk)))
        if r >= self.high:
            return PlanningBudget(0,0,False,"uncertainty above safe planning boundary")
        if r >= self.medium:
            return PlanningBudget(self.min_horizon,self.min_candidates,True,"high uncertainty: short planning")
        if r >= self.low:
            return PlanningBudget(max(self.min_horizon,self.base_horizon//2),
                                  max(self.min_candidates,self.base_candidates//2),True,
                                  "moderate uncertainty: reduced planning")
        return PlanningBudget(self.base_horizon,self.base_candidates,True,"low uncertainty: full planning")
