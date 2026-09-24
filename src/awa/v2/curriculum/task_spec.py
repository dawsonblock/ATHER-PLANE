from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable
import math


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    stage: int
    difficulty: float
    concepts: tuple[str, ...]
    seed: int
    environment: dict[str, Any] = field(default_factory=dict)
    goal: dict[str, Any] = field(default_factory=dict)
    verifier_name: str = "external"
    novelty_class: str = "train"

    def __post_init__(self):
        if not str(self.task_id):
            raise ValueError("task_id cannot be empty")
        difficulty = float(self.difficulty)
        if not math.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
            raise ValueError("difficulty must be finite and in [0,1]")
        if int(self.stage) < 1:
            raise ValueError("stage must be >= 1")
        if not str(self.verifier_name):
            raise ValueError("verifier_name cannot be empty")

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TaskOutcome:
    task_id: str
    success: bool
    reward: float
    prediction_error: float = 0.0
    novelty: float = 0.0
    planner_dependency: float = 0.0
    episodes_seen: int = 1

    def __post_init__(self):
        for name in ("reward", "prediction_error", "novelty", "planner_dependency"):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"{name} must be finite")
        if float(self.prediction_error) < 0 or float(self.novelty) < 0:
            raise ValueError("prediction_error/novelty must be >= 0")
        if not 0.0 <= float(self.planner_dependency) <= 1.0:
            raise ValueError("planner_dependency must be in [0,1]")
        if int(self.episodes_seen) <= 0:
            raise ValueError("episodes_seen must be > 0")


Verifier = Callable[[dict[str, Any]], bool]
