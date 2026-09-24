from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from .task_spec import TaskSpec, TaskOutcome


@dataclass(frozen=True)
class ConceptStats:
    concept: str
    attempts: int
    success_rate: float
    recent_success: float

    def to_dict(self):
        return asdict(self)


class ConceptCoverageTracker:
    """Tracks curriculum competence by reusable concept, not only by task id."""

    def __init__(self, history: int = 100):
        self.history = int(history)
        self._rows = defaultdict(lambda: deque(maxlen=self.history))

    def record(self, task: TaskSpec, outcome: TaskOutcome) -> None:
        for concept in task.concepts:
            self._rows[str(concept)].append(bool(outcome.success))

    def stats(self, concept: str) -> ConceptStats:
        rows = list(self._rows[str(concept)])
        if not rows:
            return ConceptStats(str(concept), 0, 0.0, 0.0)
        recent = rows[-min(20, len(rows)):]
        return ConceptStats(str(concept), len(rows), sum(rows) / len(rows), sum(recent) / len(recent))

    def weakest(self, k: int = 5) -> list[ConceptStats]:
        rows = [self.stats(c) for c in self._rows]
        rows.sort(key=lambda x: (x.recent_success, x.attempts))
        return rows[: int(k)]
