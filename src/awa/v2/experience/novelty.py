from __future__ import annotations

from collections import deque
import numpy as np


class RunningNovelty:
    """Cosine-distance novelty against a bounded recent prototype bank.

    A single Aether run can contain tasks with different goal/latent widths.  v2.7
    compared every vector against every prototype, which made a dimension change
    crash with a matrix-multiplication shape error.  v2.8 keeps one bounded bank
    but only compares shape-compatible prototypes.
    """

    def __init__(self, capacity: int = 4096):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("novelty capacity must be > 0")
        self.items = deque(maxlen=capacity)

    @staticmethod
    def _unit(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            raise ValueError("novelty vector cannot be empty")
        if not np.all(np.isfinite(x)):
            raise ValueError("novelty vector must be finite")
        return x / (np.linalg.norm(x) + 1e-8)

    def score(self, x):
        q = self._unit(x)
        compatible = [p for p in self.items if np.asarray(p).shape == q.shape]
        if not compatible:
            return 1.0
        sims = np.asarray([float(q @ p) for p in compatible], dtype=np.float32)
        return float(np.clip(1.0 - np.max(sims), 0.0, 2.0) / 2.0)

    def observe(self, x):
        q = self._unit(x)
        s = self.score(q)
        self.items.append(q)
        return s
