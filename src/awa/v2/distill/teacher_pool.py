from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class TeacherCandidate:
    teacher: str
    action: np.ndarray
    value: float
    risk: float = 0.0
    compute_cost: float = 0.0
    success: bool = True
    metadata: dict | None = None

    @property
    def utility(self):
        return float(self.value) - float(self.risk) - float(self.compute_cost)


@dataclass(frozen=True)
class DistillationExample:
    state: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    teacher: str
    utility: float
    weight: float
    metadata: dict | None = None


class TeacherPool:
    """Select the best successful finite teacher under value-risk-compute utility."""

    def __init__(self, min_margin=0.0):
        self.min_margin = float(min_margin)
        if not math.isfinite(self.min_margin):
            raise ValueError("min_margin must be finite")

    @staticmethod
    def _valid(c: TeacherCandidate) -> bool:
        if not c.success:
            return False
        values = (float(c.value), float(c.risk), float(c.compute_cost), float(c.utility))
        if not all(math.isfinite(v) for v in values):
            return False
        action = np.asarray(c.action, dtype=np.float32)
        return action.size > 0 and bool(np.all(np.isfinite(action)))

    def select(self, candidates: list[TeacherCandidate], student_value: float = 0.0):
        student_value = float(student_value)
        if not math.isfinite(student_value):
            raise ValueError("student_value must be finite")
        valid = [c for c in candidates if self._valid(c)]
        if not valid:
            return None
        best = max(valid, key=lambda c: c.utility)
        return best if best.utility >= student_value + self.min_margin else None

    def example(self, state, goal, candidates, student_value=0.0, metadata=None):
        best = self.select(candidates, student_value)
        if best is None:
            return None
        margin = max(0.0, best.utility - float(student_value))
        state = np.asarray(state, dtype=np.float32)
        goal = np.asarray(goal, dtype=np.float32)
        action = np.asarray(best.action, dtype=np.float32)
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(goal)):
            raise ValueError("distillation state/goal must be finite")
        return DistillationExample(
            state,
            goal,
            action,
            best.teacher,
            best.utility,
            max(0.05, margin),
            metadata,
        )
