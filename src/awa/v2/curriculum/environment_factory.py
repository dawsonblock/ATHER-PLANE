from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .task_spec import TaskSpec


FACTOR_OBJECTIVES: dict[str, tuple[str, ...]] = {
    "navigation": ("reach_goal",),
    "memory": ("remember_goal", "reach_goal"),
    "combat": ("defeat_enemy", "reach_goal"),
    "scarcity": ("collect_resource", "defeat_enemy", "reach_goal"),
    "key_door": ("collect_key", "open_door", "reach_goal"),
    "hazards": ("avoid_enemy", "reach_goal"),
    "dynamics_shift": ("reach_goal",),
}

FACTOR_ORDER = tuple(FACTOR_OBJECTIVES)


@dataclass(frozen=True)
class GeneratedTaskReceipt:
    task_id: str
    split: str
    factor_signature: tuple[str, ...]
    task_sha256: str
    constructible: bool
    finite_observation: bool
    finite_goal: bool
    reachable_targets: bool

    @property
    def valid(self) -> bool:
        return bool(
            self.constructible
            and self.finite_observation
            and self.finite_goal
            and self.reachable_targets
        )

    def to_dict(self) -> dict:
        return asdict(self) | {"valid": self.valid}


@dataclass(frozen=True)
class CompositionalBenchmark:
    train: tuple[TaskSpec, ...]
    heldout: tuple[TaskSpec, ...]
    transfer: tuple[TaskSpec, ...]
    receipts: tuple[GeneratedTaskReceipt, ...]

    @property
    def sha256(self) -> str:
        payload = {
            "train": [task.to_dict() for task in self.train],
            "heldout": [task.to_dict() for task in self.heldout],
            "transfer": [task.to_dict() for task in self.transfer],
            "receipts": [row.to_dict() for row in self.receipts],
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict:
        train_signatures = sorted({_signature(t.concepts) for t in self.train})
        heldout_signatures = sorted({_signature(t.concepts) for t in self.heldout})
        transfer_signatures = sorted({_signature(t.concepts) for t in self.transfer})
        return {
            "format": "awa-v2.30-compositional-benchmark-v1",
            "sha256": self.sha256,
            "train_tasks": [task.to_dict() for task in self.train],
            "heldout_tasks": [task.to_dict() for task in self.heldout],
            "transfer_tasks": [task.to_dict() for task in self.transfer],
            "train_factor_signatures": [list(x) for x in train_signatures],
            "heldout_factor_signatures": [list(x) for x in heldout_signatures],
            "transfer_factor_signatures": [list(x) for x in transfer_signatures],
            "exact_signature_leakage": bool(
                set(train_signatures) & (set(heldout_signatures) | set(transfer_signatures))
            ),
            "all_tasks_valid": all(row.valid for row in self.receipts),
            "receipts": [row.to_dict() for row in self.receipts],
        }

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target


def _signature(factors: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({str(f) for f in factors}, key=FACTOR_ORDER.index))


def _task_sha256(task: TaskSpec) -> str:
    raw = json.dumps(task.to_dict(), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


class FactorizedEnvironmentFactory:
    """Generate factor-controlled procedural tasks with explicit split semantics.

    This is intentionally separate from the stage curriculum.  The factory is for
    causal/compositional evaluation where the factor combination itself is the
    controlled variable.  Exact factor signatures are never shared across train,
    heldout and transfer splits in the canonical benchmark.
    """

    def __init__(self, base_seed: int = 2300):
        self.base_seed = int(base_seed)

    def make(
        self,
        factors: Iterable[str],
        *,
        difficulty: float,
        index: int,
        split: str,
    ) -> TaskSpec:
        signature = _signature(factors)
        if not signature:
            raise ValueError("at least one factor is required")
        unknown = [factor for factor in signature if factor not in FACTOR_OBJECTIVES]
        if unknown:
            raise ValueError(f"unknown factors: {unknown}")
        difficulty = float(np.clip(float(difficulty), 0.0, 1.0))
        split = str(split)
        seed_raw = json.dumps(
            [self.base_seed, signature, difficulty, int(index), split],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        seed = int.from_bytes(hashlib.sha256(seed_raw).digest()[:4], "big") & 0x7FFFFFFF
        rng = np.random.default_rng(seed)

        objectives: list[str] = []
        for factor in signature:
            for objective in FACTOR_OBJECTIVES[factor]:
                if objective not in objectives:
                    objectives.append(objective)

        # Ensure prerequisite-sensitive objectives preserve a sensible sequence.
        ordered = []
        for objective in (
            "remember_goal",
            "collect_resource",
            "collect_key",
            "open_door",
            "defeat_enemy",
            "avoid_enemy",
            "reach_goal",
        ):
            if objective in objectives:
                ordered.append(objective)
        objectives = ordered or ["reach_goal"]

        dynamics = "dynamics_shift" in signature
        memory = "memory" in signature
        combat = bool({"combat", "scarcity", "hazards"} & set(signature))
        environment = {
            "layout_seed": int(rng.integers(0, 2**31 - 1)),
            "obstacle_density": float(0.10 + 0.42 * difficulty),
            "object_density": float(0.12 + 0.48 * difficulty),
            "enemy_density": float((0.15 + 0.55 * difficulty) if combat else 0.0),
            "enemy_aggression": float((0.25 + 0.65 * difficulty) if combat else 0.0),
            "gravity_scale": float(rng.uniform(0.60, 1.40) if dynamics else 1.0),
            "friction_scale": float(rng.uniform(0.45, 1.55) if dynamics else 1.0),
            "movement_scale": float(rng.uniform(0.70, 1.30) if dynamics else 1.0),
            "lighting_seed": int(rng.integers(0, 2**31 - 1)),
            "texture_seed": int(rng.integers(0, 2**31 - 1)),
            "hide_goal_after_steps": int(5 + round(8 * difficulty)) if memory else None,
            "resource_scarcity": float(0.30 + 0.60 * difficulty) if "scarcity" in signature else 0.0,
            "hazard_density": float(0.20 + 0.55 * difficulty) if "hazards" in signature else 0.0,
        }
        goal = {
            "objectives": objectives,
            "target_count": 1,
            "success_threshold": 0.9,
            "factor_signature": list(signature),
        }
        task_raw = json.dumps(
            [signature, difficulty, index, split, seed, environment, goal],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        task_id = f"cmp-{split}-{hashlib.sha256(task_raw).hexdigest()[:14]}"
        # Stage 12 is retained only for compatibility with the arena schema.  The
        # executable semantics come from goal/objective factors, not the stage id.
        return TaskSpec(
            task_id=task_id,
            stage=12,
            difficulty=difficulty,
            concepts=signature,
            seed=seed,
            environment=environment,
            goal=goal,
            verifier_name="engine_verifier",
            novelty_class=split,
        )

    def validate(self, task: TaskSpec, *, horizon: int = 64) -> GeneratedTaskReceipt:
        from awa.v2.game.procedural_arena import ProceduralArenaEnv

        constructible = False
        finite_observation = False
        finite_goal = False
        reachable_targets = False
        try:
            env = ProceduralArenaEnv(task, horizon=horizon)
            obs = env.reset(seed=task.seed)
            constructible = True
            finite_observation = bool(np.all(np.isfinite(np.asarray(obs))))
            finite_goal = bool(np.all(np.isfinite(np.asarray(env.goal_vector()))))
            targets = []
            for objective in env.program.objectives:
                target = env._objective_target(objective)  # deterministic geometry audit
                if target is not None:
                    targets.append(target)
            reachable_targets = all(env._grid_reachable(env.player, target) for target in targets)
        except Exception:
            pass
        return GeneratedTaskReceipt(
            task_id=task.task_id,
            split=task.novelty_class,
            factor_signature=_signature(task.concepts),
            task_sha256=_task_sha256(task),
            constructible=constructible,
            finite_observation=finite_observation,
            finite_goal=finite_goal,
            reachable_targets=reachable_targets,
        )

    def build_canonical_benchmark(
        self,
        *,
        train_replicates: int = 2,
        eval_replicates: int = 1,
        train_difficulty: float = 0.50,
        heldout_difficulty: float = 0.70,
        transfer_difficulty: float = 0.82,
    ) -> CompositionalBenchmark:
        if train_replicates <= 0 or eval_replicates <= 0:
            raise ValueError("replicate counts must be positive")

        train_signatures = [
            ("navigation",),
            ("memory",),
            ("combat",),
            ("scarcity",),
            ("key_door",),
            ("hazards",),
            ("dynamics_shift",),
            ("navigation", "memory"),
            ("combat", "scarcity"),
            ("key_door", "hazards"),
        ]
        heldout_signatures = [
            ("memory", "combat"),
            ("key_door", "dynamics_shift"),
            ("scarcity", "hazards"),
        ]
        transfer_signatures = [
            ("memory", "combat", "scarcity"),
            ("key_door", "hazards", "dynamics_shift"),
            ("navigation", "memory", "key_door", "combat"),
        ]

        train: list[TaskSpec] = []
        heldout: list[TaskSpec] = []
        transfer: list[TaskSpec] = []
        for signature in train_signatures:
            for rep in range(int(train_replicates)):
                train.append(
                    self.make(
                        signature,
                        difficulty=train_difficulty,
                        index=rep,
                        split="train",
                    )
                )
        for signature in heldout_signatures:
            for rep in range(int(eval_replicates)):
                heldout.append(
                    self.make(
                        signature,
                        difficulty=heldout_difficulty,
                        index=rep,
                        split="heldout_composition",
                    )
                )
        for signature in transfer_signatures:
            for rep in range(int(eval_replicates)):
                transfer.append(
                    self.make(
                        signature,
                        difficulty=transfer_difficulty,
                        index=rep,
                        split="transfer_composition",
                    )
                )

        split_signatures = {
            "train": {_signature(task.concepts) for task in train},
            "heldout": {_signature(task.concepts) for task in heldout},
            "transfer": {_signature(task.concepts) for task in transfer},
        }
        if split_signatures["train"] & split_signatures["heldout"]:
            raise RuntimeError("train/heldout factor-signature leakage")
        if split_signatures["train"] & split_signatures["transfer"]:
            raise RuntimeError("train/transfer factor-signature leakage")
        if split_signatures["heldout"] & split_signatures["transfer"]:
            raise RuntimeError("heldout/transfer factor-signature leakage")

        receipts = tuple(self.validate(task) for task in [*train, *heldout, *transfer])
        if not all(row.valid for row in receipts):
            invalid = [row.task_id for row in receipts if not row.valid]
            raise RuntimeError(f"generated invalid tasks: {invalid}")
        return CompositionalBenchmark(tuple(train), tuple(heldout), tuple(transfer), receipts)
