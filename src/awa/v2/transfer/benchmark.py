from __future__ import annotations

from dataclasses import dataclass, asdict, replace
from enum import Enum
from collections import defaultdict
import numpy as np

from awa.v2.curriculum.task_factory import ProceduralTaskFactory
from awa.v2.curriculum.task_spec import TaskSpec


class OODCategory(str, Enum):
    SEEN = "seen"
    VISUAL = "visual_ood"
    LAYOUT = "layout_ood"
    DYNAMICS = "dynamics_ood"
    COMPOSITIONAL = "compositional_ood"
    TASK = "task_ood"


@dataclass(frozen=True)
class TransferEpisode:
    category: OODCategory
    task_id: str
    episode_index: int
    success: bool
    reward: float
    planner_dependency: float = 0.0
    prediction_error: float = 0.0


@dataclass(frozen=True)
class TransferStats:
    category: str
    tasks: int
    episodes: int
    first_episode_success: float
    final_window_success: float
    adaptation_auc: float
    episodes_to_target: float | None
    mean_reward: float
    mean_prediction_error: float
    planner_dependency_start: float
    planner_dependency_end: float

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class TransferReport:
    categories: dict[str, TransferStats]
    macro_adaptation_auc: float
    macro_final_success: float

    def to_dict(self):
        return {
            "categories": {k: v.to_dict() for k, v in self.categories.items()},
            "macro_adaptation_auc": self.macro_adaptation_auc,
            "macro_final_success": self.macro_final_success,
        }


class TransferBenchmark:
    """Few-shot transfer metrics computed per held-out task, then macro-averaged.

    Computing adaptation per task avoids an easy methodological error: interleaving
    several held-out tasks and treating their episode numbers as one learning curve.
    """

    def __init__(self, target_success: float = 0.8, window: int = 5):
        self.target_success = float(target_success)
        self.window = max(1, int(window))

    def _task_metrics(self, rows: list[TransferEpisode]):
        rows = sorted(rows, key=lambda r: r.episode_index)
        success = np.asarray([float(r.success) for r in rows], dtype=np.float32)
        reward = np.asarray([r.reward for r in rows], dtype=np.float32)
        pred = np.asarray([r.prediction_error for r in rows], dtype=np.float32)
        dep = np.asarray([r.planner_dependency for r in rows], dtype=np.float32)
        rolling = np.asarray([success[max(0, i - self.window + 1):i + 1].mean() for i in range(len(success))], dtype=np.float32)
        hit = np.flatnonzero(rolling >= self.target_success)
        w = min(self.window, len(rows))
        return {
            "episodes": len(rows),
            "first": float(success[0]),
            "final": float(success[-w:].mean()),
            "auc": float(rolling.mean()),
            "target": None if len(hit) == 0 else float(int(hit[0]) + 1),
            "reward": float(reward.mean()),
            "pred": float(pred.mean()),
            "dep_start": float(dep[:w].mean()),
            "dep_end": float(dep[-w:].mean()),
        }

    def evaluate(self, episodes: list[TransferEpisode]) -> TransferReport:
        by_category: dict[OODCategory, dict[str, list[TransferEpisode]]] = defaultdict(lambda: defaultdict(list))
        for row in episodes:
            by_category[OODCategory(row.category)][row.task_id].append(row)
        stats: dict[str, TransferStats] = {}
        for category, tasks in by_category.items():
            metrics = [self._task_metrics(rows) for rows in tasks.values()]
            targets = [m["target"] for m in metrics if m["target"] is not None]
            stats[category.value] = TransferStats(
                category.value,
                len(metrics),
                int(sum(m["episodes"] for m in metrics)),
                float(np.mean([m["first"] for m in metrics])),
                float(np.mean([m["final"] for m in metrics])),
                float(np.mean([m["auc"] for m in metrics])),
                None if not targets else float(np.mean(targets)),
                float(np.mean([m["reward"] for m in metrics])),
                float(np.mean([m["pred"] for m in metrics])),
                float(np.mean([m["dep_start"] for m in metrics])),
                float(np.mean([m["dep_end"] for m in metrics])),
            )
        if not stats:
            raise ValueError("transfer benchmark requires at least one episode")
        return TransferReport(
            stats,
            float(np.mean([s.adaptation_auc for s in stats.values()])),
            float(np.mean([s.final_window_success for s in stats.values()])),
        )


def build_transfer_suite(factory: ProceduralTaskFactory, count_per_category: int = 4, difficulty: float = 0.6) -> dict[OODCategory, list[TaskSpec]]:
    """Build executable held-out OOD tasks; no category is metadata-only."""
    base = factory.make(8, difficulty, 50_000, OODCategory.SEEN.value)
    out: dict[OODCategory, list[TaskSpec]] = {c: [] for c in OODCategory}
    for i in range(int(count_per_category)):
        out[OODCategory.SEEN].append(replace(base, task_id=f"seen-{i}", novelty_class=OODCategory.SEEN.value))

        visual_env = dict(base.environment)
        visual_env["texture_seed"] = 700_000 + i; visual_env["lighting_seed"] = 800_000 + i
        out[OODCategory.VISUAL].append(replace(base, task_id=f"visual-{i}", environment=visual_env, novelty_class=OODCategory.VISUAL.value))

        layout_env = dict(base.environment); layout_env["layout_seed"] = 900_000 + i
        out[OODCategory.LAYOUT].append(replace(base, task_id=f"layout-{i}", environment=layout_env, novelty_class=OODCategory.LAYOUT.value))

        dynamics_env = dict(base.environment)
        rng = np.random.default_rng(factory.base_seed + 100_000 + i)
        dynamics_env.update({
            "gravity_scale": float(rng.uniform(.55, 1.45)),
            "friction_scale": float(rng.uniform(.35, 1.65)),
            "movement_scale": float(rng.uniform(.70, 1.30)),
        })
        out[OODCategory.DYNAMICS].append(replace(base, task_id=f"dynamics-{i}", environment=dynamics_env, concepts=base.concepts + ("dynamics_shift", "adaptation"), novelty_class=OODCategory.DYNAMICS.value))

        # Same enemy/cover world, but a genuinely longer unseen objective composition.
        comp_goal = dict(base.goal); comp_goal.update({"objectives":["take_cover","defeat_enemy","reach_goal"],"heldout_composition":i})
        out[OODCategory.COMPOSITIONAL].append(replace(
            base, task_id=f"composition-{i}", concepts=base.concepts + ("composition","multi_step"),
            goal=comp_goal, novelty_class=OODCategory.COMPOSITIONAL.value,
        ))

        # A different executable objective: collect N objects before extraction.
        task_goal = dict(base.goal); task_goal.update({"objectives":["collect_object","reach_goal"],"target_count":2+(i%2),"heldout_objective":f"collect_{i}"})
        out[OODCategory.TASK].append(replace(
            base, task_id=f"task-{i}", concepts=base.concepts + ("objects","collection"),
            goal=task_goal, novelty_class=OODCategory.TASK.value,
        ))
    return out
