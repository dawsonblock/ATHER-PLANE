from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .coverage import ConceptCoverageTracker
from .task_factory import ProceduralTaskFactory, STAGE_CONCEPTS
from .task_spec import TaskSpec


@dataclass
class AdaptiveTaskGenerator:
    """Generate fresh practice tasks around weak reusable concepts.

    v2.7 recreated this generator on every call with the same seed offset, causing
    identical adaptive tasks to be emitted repeatedly.  ``counter`` is now part of
    the resumable curriculum state and advances for every generated task.
    """

    factory: ProceduralTaskFactory
    coverage: ConceptCoverageTracker
    seed_offset: int = 1_000_000
    counter: int = 0

    def _stage_for_concept(self, concept: str) -> int:
        candidates = [stage for stage, concepts in STAGE_CONCEPTS.items() if concept in concepts]
        return min(candidates) if candidates else 1

    @staticmethod
    def _difficulty(success: float, attempts: int) -> float:
        if attempts == 0:
            return 0.30
        if success < 0.30:
            return 0.25
        if success <= 0.80:
            return float(np.clip(0.40 + 0.35 * success, 0.45, 0.70))
        return 0.80

    def generate(self, count: int = 4) -> list[TaskSpec]:
        count = int(count)
        if count < 0:
            raise ValueError("count must be >= 0")
        if count == 0:
            return []
        weakest = self.coverage.weakest(max(1, count))
        tasks: list[TaskSpec] = []
        for i in range(count):
            index = int(self.seed_offset + self.counter)
            self.counter += 1
            if not weakest:
                tasks.append(self.factory.make(1, 0.30, index, "adaptive"))
                continue
            stat = weakest[i % len(weakest)]
            stage = self._stage_for_concept(stat.concept)
            difficulty = self._difficulty(stat.recent_success, stat.attempts)
            tasks.append(
                self.factory.make(
                    stage,
                    difficulty,
                    index,
                    f"adaptive:{stat.concept}",
                )
            )
        return tasks
