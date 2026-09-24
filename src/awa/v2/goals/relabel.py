from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np


@dataclass(frozen=True)
class RelabeledTransition:
    transition_index: int
    source_future_index: int
    action: np.ndarray
    achieved_goal: np.ndarray
    original_goal: np.ndarray
    relabeled_goal: np.ndarray
    reward: float
    success: bool

    def to_dict(self):
        d = asdict(self)
        for key in ("action", "achieved_goal", "original_goal", "relabeled_goal"):
            d[key] = np.asarray(d[key]).tolist()
        return d


class HindsightGoalRelabeler:
    """Future-goal hindsight relabeling for goal-conditioned trajectories.

    A transition that failed its original objective can still teach how to reach an
    outcome that was actually achieved later in the same episode.  Relabeling is kept
    separate from the environment reward function so game-specific semantics can be
    supplied through ``reward_fn`` when needed.
    """

    def __init__(self, k_future: int = 4, tolerance: float = 0.05, seed: int = 0):
        self.k_future = max(1, int(k_future))
        self.tolerance = float(tolerance)
        self.rng = np.random.default_rng(seed)

    def relabel_episode(
        self,
        achieved_goals,
        actions,
        desired_goal,
        *,
        reward_fn=None,
        include_original: bool = False,
    ) -> list[RelabeledTransition]:
        achieved = np.asarray(achieved_goals, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if achieved.ndim == 1:
            achieved = achieved[:, None]
        if actions.shape[0] != achieved.shape[0]:
            raise ValueError("actions and achieved_goals must have the same transition count")
        desired = np.asarray(desired_goal, dtype=np.float32).reshape(-1)
        if desired.shape[-1] != achieved.shape[-1]:
            raise ValueError("desired_goal dimension must match achieved_goal dimension")
        rows: list[RelabeledTransition] = []
        n = len(achieved)
        for t in range(n):
            candidates = np.arange(t, n, dtype=np.int64)
            take = min(self.k_future, len(candidates))
            chosen = self.rng.choice(candidates, size=take, replace=False)
            if include_original:
                chosen = np.concatenate([np.asarray([-1], dtype=np.int64), np.asarray(chosen, dtype=np.int64)])
            for j in chosen:
                goal = desired if int(j) < 0 else achieved[int(j)]
                distance = float(np.linalg.norm(achieved[t] - goal))
                success = distance <= self.tolerance
                reward = float(reward_fn(achieved[t], goal, success)) if reward_fn is not None else float(success)
                rows.append(RelabeledTransition(
                    t,
                    t if int(j) < 0 else int(j),
                    np.asarray(actions[t], dtype=np.float32),
                    np.asarray(achieved[t], dtype=np.float32),
                    desired.copy(),
                    np.asarray(goal, dtype=np.float32).copy(),
                    reward,
                    success,
                ))
        return rows
