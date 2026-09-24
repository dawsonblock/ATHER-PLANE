from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PriorityWeights:
    td_error: float = 0.30
    prediction_error: float = 0.25
    novelty: float = 0.15
    failure: float = 0.10
    task_importance: float = 0.10
    information_value: float = 0.10

    def __post_init__(self):
        values = np.asarray(list(self.__dict__.values()), dtype=np.float64)
        if np.any(~np.isfinite(values)) or np.any(values < 0):
            raise ValueError("priority weights must be finite and non-negative")
        if float(values.sum()) <= 0:
            raise ValueError("at least one priority weight must be positive")


class StructuralPriority:
    def __init__(self, weights=PriorityWeights(), clip=5.0, epsilon=1e-4):
        self.weights = weights
        self.clip = float(clip)
        self.epsilon = float(epsilon)
        if not np.isfinite(self.clip) or self.clip <= 0:
            raise ValueError("priority clip must be finite and > 0")
        if not np.isfinite(self.epsilon) or self.epsilon <= 0 or self.epsilon > self.clip:
            raise ValueError("priority epsilon must be finite, > 0 and <= clip")

    @staticmethod
    def _nonnegative(value, name):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return max(0.0, value)

    def __call__(self, r):
        failure = 0.0 if bool(r.success) else 1.0
        w = self.weights
        raw = (
            w.td_error * abs(float(r.td_error))
            + w.prediction_error * self._nonnegative(r.prediction_error, "prediction_error")
            + w.novelty * self._nonnegative(r.novelty, "novelty")
            + w.failure * failure
            + w.task_importance * self._nonnegative(r.task_importance, "task_importance")
            + w.information_value * self._nonnegative(r.information_value, "information_value")
        )
        if getattr(r, "repeated_surprise", False):
            raw *= 1.15
        return float(np.clip(raw, self.epsilon, self.clip))


class PrioritizedExperienceBuffer:
    def __init__(self, capacity=100_000, priority=None, seed=0):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("replay capacity must be > 0")
        self.priority = priority or StructuralPriority()
        self.rng = np.random.default_rng(seed)
        self.items = []
        self.priorities = []

    def add(self, item):
        p = self.priority(item)
        self.items.append(item)
        self.priorities.append(p)
        if len(self.items) > self.capacity:
            self.items.pop(0)
            self.priorities.pop(0)

    def sample(self, n, beta=0.4):
        n = int(n)
        beta = float(beta)
        if not self.items:
            raise ValueError("empty replay")
        if n <= 0:
            raise ValueError("sample size must be > 0")
        if not np.isfinite(beta) or beta < 0:
            raise ValueError("beta must be finite and >= 0")
        p = np.asarray(self.priorities, dtype=np.float64)
        if np.any(~np.isfinite(p)) or float(p.sum()) <= 0:
            raise ValueError("replay priorities are invalid")
        p = p / p.sum()
        idx = self.rng.choice(
            len(self.items),
            size=n,
            replace=len(self.items) < n,
            p=p,
        )
        weights = np.power(len(self.items) * p[idx], -beta)
        weights = weights / (weights.max() + 1e-8)
        return [self.items[int(i)] for i in idx], idx.astype(np.int64), weights.astype(np.float32)

    def update(self, indices, items=None):
        for j, i in enumerate(indices):
            i = int(i)
            if i < 0 or i >= len(self.items):
                raise IndexError("replay priority index out of range")
            item = self.items[i] if items is None else items[j]
            self.priorities[i] = self.priority(item)

    def __len__(self):
        return len(self.items)
