from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any
import math


class FailureClass(str, Enum):
    PERCEPTION = "perception"
    WORLD_MODEL = "world_model"
    MEMORY = "memory"
    ACTOR = "actor"
    PLANNER = "planner"
    SKILL = "skill"
    GOAL = "goal"
    RISK = "risk"
    ENVIRONMENT = "environment"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureEvidence:
    success: bool
    perception_error: float | None = None
    world_prediction_error: float | None = None
    memory_probe_error: float | None = None
    actor_return: float | None = None
    planner_return: float | None = None
    planner_available: bool = False
    planner_used: bool = False
    skill_attempted: bool = False
    skill_success: bool | None = None
    goal_progress: float | None = None
    risk_rejections: int = 0
    invalid_actions: int = 0
    environment_invalid: bool = False
    uncertainty: float | None = None

    def __post_init__(self):
        for name in (
            "perception_error", "world_prediction_error", "memory_probe_error",
            "actor_return", "planner_return", "goal_progress", "uncertainty",
        ):
            value = getattr(self, name)
            if value is not None and not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite when provided")
        if self.risk_rejections < 0 or self.invalid_actions < 0:
            raise ValueError("counts cannot be negative")


@dataclass(frozen=True)
class TriageResult:
    primary: FailureClass
    confidence: float
    scores: dict[str, float]
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["primary"] = self.primary.value
        data["reasons"] = list(self.reasons)
        return data


def triage_failure(e: FailureEvidence) -> TriageResult:
    """Evidence-based heuristic triage, not a claim of causal proof."""
    if e.success:
        return TriageResult(FailureClass.UNKNOWN, 0.0, {}, ("episode succeeded",))
    scores = {c.value: 0.0 for c in FailureClass}
    reasons: list[str] = []
    if e.environment_invalid:
        scores[FailureClass.ENVIRONMENT.value] += 1.0
        reasons.append("environment reported an invalid episode/state")
    if e.perception_error is not None and e.perception_error >= 0.35:
        scores[FailureClass.PERCEPTION.value] += min(1.0, float(e.perception_error))
        reasons.append("perception error is high")
    if e.world_prediction_error is not None and e.world_prediction_error >= 0.30:
        scores[FailureClass.WORLD_MODEL.value] += min(1.0, float(e.world_prediction_error))
        reasons.append("world-model prediction error is high")
    if e.memory_probe_error is not None and e.memory_probe_error >= 0.30:
        scores[FailureClass.MEMORY.value] += min(1.0, float(e.memory_probe_error))
        reasons.append("memory probe error is high")
    if e.skill_attempted and e.skill_success is False:
        scores[FailureClass.SKILL.value] += 0.65
        reasons.append("selected skill failed")
    if e.planner_available and e.actor_return is not None and e.planner_return is not None:
        gain = float(e.planner_return) - float(e.actor_return)
        if gain >= 0.20:
            scores[FailureClass.ACTOR.value] += min(1.0, 0.45 + gain)
            reasons.append("planner materially outperformed actor from comparable state")
        elif e.planner_used and gain <= 0.02 and (e.world_prediction_error or 0.0) < 0.20:
            scores[FailureClass.PLANNER.value] += 0.55
            reasons.append("planner failed to improve despite low model error")
    if e.goal_progress is not None and float(e.goal_progress) <= 0.05:
        scores[FailureClass.GOAL.value] += 0.35
        reasons.append("episode made almost no goal progress")
    if e.risk_rejections >= 3 and (e.goal_progress or 0.0) <= 0.10:
        scores[FailureClass.RISK.value] += min(1.0, 0.35 + 0.1 * e.risk_rejections)
        reasons.append("risk gate repeatedly rejected actions while progress stalled")
    if e.invalid_actions >= 2:
        scores[FailureClass.ACTOR.value] += min(0.7, 0.25 + 0.1 * e.invalid_actions)
        reasons.append("policy produced repeated invalid actions")
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]
    if best_score <= 0:
        return TriageResult(FailureClass.UNKNOWN, 0.0, scores, ("insufficient diagnostic evidence",))
    second = ranked[1][1]
    confidence = max(0.0, min(1.0, best_score - 0.5 * second))
    return TriageResult(FailureClass(best_name), confidence, scores, tuple(reasons))
