from __future__ import annotations
from dataclasses import dataclass


@dataclass
class ArbitrationDecision:
    source: str
    reason: str


class ReliabilityGate:
    def __init__(self, uncertainty_limit: float = 1.5):
        self.uncertainty_limit = uncertainty_limit

    def choose(self, uncertainty: float, planner_enabled: bool = True) -> ArbitrationDecision:
        if planner_enabled and uncertainty <= self.uncertainty_limit:
            return ArbitrationDecision("planner", "world-model uncertainty is within planning boundary")
        return ArbitrationDecision("actor", "planner disabled or model uncertainty too high")
