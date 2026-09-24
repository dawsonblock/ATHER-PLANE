from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass
class ExperienceRecord:
    state: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    predicted_next: np.ndarray
    actual_next: np.ndarray
    reward: float
    success: bool
    prediction_error: float
    novelty: float
    uncertainty: float
    risk: float
    td_error: float = 0.0
    task_importance: float = 0.0
    information_value: float = 0.0
    repeated_surprise: bool = False
    metadata: dict | None = None

    def to_dict(self):
        d = asdict(self)
        for k in ("state", "goal", "action", "predicted_next", "actual_next"):
            d[k] = np.asarray(d[k]).tolist()
        return d


class ExperienceAnalyzer:
    def __init__(self, surprise_threshold=0.25, repeat_window=4):
        self.surprise_threshold = float(surprise_threshold)
        self.repeat_window = int(repeat_window)
        if not np.isfinite(self.surprise_threshold) or self.surprise_threshold < 0:
            raise ValueError("surprise_threshold must be finite and >= 0")
        if self.repeat_window <= 0:
            raise ValueError("repeat_window must be > 0")
        self._recent = []

    @staticmethod
    def prediction_error(predicted_next, actual_next):
        p = np.asarray(predicted_next, dtype=np.float32)
        a = np.asarray(actual_next, dtype=np.float32)
        if p.shape != a.shape:
            raise ValueError(
                f"predicted_next shape {p.shape} must match actual_next shape {a.shape}"
            )
        if p.size == 0 or not np.all(np.isfinite(p)) or not np.all(np.isfinite(a)):
            raise ValueError("predicted/actual next state must be non-empty and finite")
        return float(np.sqrt(np.mean(np.square(p - a))))

    def classify_surprise(self, error: float):
        error = float(error)
        if not np.isfinite(error) or error < 0:
            raise ValueError("prediction error must be finite and >= 0")
        self._recent.append(error)
        self._recent = self._recent[-self.repeat_window :]
        repeated = (
            len(self._recent) >= self.repeat_window
            and sum(e >= self.surprise_threshold for e in self._recent)
            >= max(2, self.repeat_window - 1)
        )
        if error < 0.5 * self.surprise_threshold:
            kind = "known"
        elif error < 2 * self.surprise_threshold:
            kind = "useful_surprise"
        else:
            kind = "large_surprise"
        return kind, repeated

    def make(
        self,
        *,
        state,
        goal,
        action,
        predicted_next,
        actual_next,
        reward,
        success,
        novelty,
        uncertainty,
        risk,
        td_error=0.0,
        task_importance=0.0,
        information_value=0.0,
        metadata=None,
    ):
        err = self.prediction_error(predicted_next, actual_next)
        kind, repeated = self.classify_surprise(err)
        scalars = {
            "reward": reward,
            "novelty": novelty,
            "uncertainty": uncertainty,
            "risk": risk,
            "td_error": td_error,
            "task_importance": task_importance,
            "information_value": information_value,
        }
        for name, value in scalars.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        md = dict(metadata or {})
        md["surprise_kind"] = kind
        return ExperienceRecord(
            np.asarray(state, dtype=np.float32),
            np.asarray(goal, dtype=np.float32),
            np.asarray(action, dtype=np.float32),
            np.asarray(predicted_next, dtype=np.float32),
            np.asarray(actual_next, dtype=np.float32),
            float(reward),
            bool(success),
            err,
            float(novelty),
            float(uncertainty),
            float(risk),
            float(td_error),
            float(task_importance),
            float(information_value),
            bool(repeated),
            md,
        )
