from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable
import math
import numpy as np

from .task_spec import TaskSpec


@dataclass(frozen=True)
class VerificationResult:
    success: bool
    score: float = 0.0
    reason: str = ""
    metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


VerifierFn = Callable[[TaskSpec, dict[str, Any]], VerificationResult | bool]


class TaskVerifierRegistry:
    """Named deterministic task verifiers for synthesized curriculum tasks."""

    def __init__(self, include_defaults: bool = True):
        self._verifiers: dict[str, VerifierFn] = {}
        if include_defaults:
            self.register("engine_verifier", self._engine_verifier)
            self.register("distance_goal", self._distance_goal)
            self.register("threshold", self._threshold)

    def register(self, name: str, fn: VerifierFn, *, replace: bool = False) -> None:
        key = str(name)
        if key in self._verifiers and not replace:
            raise ValueError(f"verifier already registered: {key}")
        if not callable(fn):
            raise TypeError("verifier must be callable")
        self._verifiers[key] = fn

    def verify(self, task: TaskSpec, context: dict[str, Any]) -> VerificationResult:
        if task.verifier_name not in self._verifiers:
            raise KeyError(f"unknown task verifier: {task.verifier_name}")
        result = self._verifiers[task.verifier_name](task, dict(context))
        if isinstance(result, VerificationResult):
            return result
        return VerificationResult(bool(result), float(bool(result)), task.verifier_name)

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._verifiers))

    @staticmethod
    def _engine_verifier(task: TaskSpec, context: dict[str, Any]) -> VerificationResult:
        if "success" not in context:
            raise KeyError("engine_verifier requires context['success']")
        success = bool(context["success"])
        score = float(context.get("score", context.get("reward", float(success))))
        return VerificationResult(success, score, "engine_success_flag")

    @staticmethod
    def _distance_goal(task: TaskSpec, context: dict[str, Any]) -> VerificationResult:
        if "position" not in context:
            raise KeyError("distance_goal requires context['position']")
        target_value = context.get("target", task.goal.get("target"))
        if target_value is None:
            raise KeyError("distance_goal requires context['target'] or task.goal['target']")
        p = np.asarray(context["position"], dtype=np.float32).reshape(-1)
        target = np.asarray(target_value, dtype=np.float32).reshape(-1)
        if p.shape != target.shape:
            raise ValueError("position/target shape mismatch")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(target)):
            raise ValueError("position/target must be finite")
        tolerance = float(context.get("tolerance", task.goal.get("tolerance", 0.5)))
        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("distance tolerance must be finite and >= 0")
        distance = float(np.linalg.norm(p - target))
        score = 1.0 / (1.0 + distance)
        return VerificationResult(
            distance <= tolerance,
            score,
            "distance_goal",
            {"distance": distance, "tolerance": tolerance},
        )

    @staticmethod
    def _threshold(task: TaskSpec, context: dict[str, Any]) -> VerificationResult:
        metric = str(task.goal.get("metric", context.get("metric", "score")))
        op = str(task.goal.get("op", context.get("op", ">=")))
        threshold = float(
            task.goal.get(
                "threshold",
                task.goal.get("success_threshold", context.get("threshold", 0.0)),
            )
        )
        if metric not in context:
            raise KeyError(f"threshold verifier requires context[{metric!r}]")
        value = float(context[metric])
        if not math.isfinite(value) or not math.isfinite(threshold):
            raise ValueError("threshold value and target must be finite")
        ops = {
            ">=": lambda a, b: a >= b,
            ">": lambda a, b: a > b,
            "<=": lambda a, b: a <= b,
            "<": lambda a, b: a < b,
            "==": lambda a, b: math.isclose(a, b),
        }
        if op not in ops:
            raise ValueError(f"unsupported threshold op: {op}")
        return VerificationResult(
            bool(ops[op](value, threshold)),
            value,
            f"{metric}{op}{threshold}",
            {"metric": metric, "threshold": threshold},
        )
