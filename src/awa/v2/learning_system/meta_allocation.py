"""Declarative training-allocation interface used online and in historical replay.

The contract intentionally contains no executable callbacks or arbitrary code.  It is the
shared decision surface between real Aether training and DREAM-RSI-style historical replay.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
import math
from typing import Iterable, Sequence

_ALLOWED_COLLECTORS = {
    "coverage",
    "random",
    "teacher",
    "aether_actor",
    "aether_planner",
}


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class TrainingAllocationDecision:
    """One executable training-allocation action.

    ``stop=True`` is represented explicitly so replay and online execution use the same
    interface, mirroring DREAM-RSI's common decision surface for online and replay phases.
    """

    factor_signature: tuple[str, ...] = ()
    collector: str = "coverage"
    transition_budget: int = 0
    planner_budget: int = 0
    parallel_worlds: int = 1
    stop: bool = False

    def __post_init__(self) -> None:
        canonical = tuple(sorted(set(str(x) for x in self.factor_signature if str(x))))
        if canonical != self.factor_signature:
            raise ValueError("factor_signature must be sorted, unique, and non-empty strings")
        if self.collector not in _ALLOWED_COLLECTORS:
            raise ValueError(f"unsupported collector: {self.collector}")
        if self.transition_budget < 0 or self.planner_budget < 0:
            raise ValueError("budgets must be >= 0")
        if self.parallel_worlds < 1:
            raise ValueError("parallel_worlds must be >= 1")
        if self.stop:
            if self.factor_signature or self.transition_budget or self.planner_budget:
                raise ValueError("stop decisions cannot allocate factors or compute")
        elif not self.factor_signature or self.transition_budget < 1:
            raise ValueError("non-stop decisions require factors and transition_budget >= 1")

    @classmethod
    def stop_decision(cls) -> "TrainingAllocationDecision":
        return cls(stop=True)

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingAllocationDecision":
        row = dict(payload)
        row["factor_signature"] = tuple(row.get("factor_signature", ()))
        return cls(**row)

    def to_dict(self) -> dict:
        out = asdict(self)
        out["factor_signature"] = list(self.factor_signature)
        return out

    @property
    def execution_key(self) -> str:
        """Exact online-execution identity for historical support accounting."""
        return _sha(self.to_dict())


@dataclass(frozen=True)
class TrainingAllocationOption:
    """Observable state plus one executable decision offered to a meta-policy.

    Every score input must be available *before* the decision executes.  Outcome metrics are
    deliberately absent so replay cannot leak a historical child's result into the decision.
    """

    decision: TrainingAllocationDecision
    capability_deficit: float = 0.0
    uncertainty: float = 0.0
    transfer_gap: float = 0.0
    novelty: float = 0.0
    expected_cost: float = 0.0
    historical_support: int = 0
    age_iterations: int = 0

    def __post_init__(self) -> None:
        values = (
            self.capability_deficit,
            self.uncertainty,
            self.transfer_gap,
            self.novelty,
            self.expected_cost,
        )
        if not all(math.isfinite(float(x)) for x in values):
            raise ValueError("allocation option signals must be finite")
        if self.expected_cost < 0 or self.historical_support < 0 or self.age_iterations < 0:
            raise ValueError("cost/support/age must be >= 0")
        if self.decision.stop:
            raise ValueError("TrainingAllocationOption cannot wrap a stop decision")

    @classmethod
    def from_dict(cls, payload: dict) -> "TrainingAllocationOption":
        row = dict(payload)
        row["decision"] = TrainingAllocationDecision.from_dict(row["decision"])
        return cls(**row)

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.to_dict(),
            "capability_deficit": float(self.capability_deficit),
            "uncertainty": float(self.uncertainty),
            "transfer_gap": float(self.transfer_gap),
            "novelty": float(self.novelty),
            "expected_cost": float(self.expected_cost),
            "historical_support": int(self.historical_support),
            "age_iterations": int(self.age_iterations),
        }


@dataclass(frozen=True)
class DeclarativeTrainingAllocationPolicy:
    """Bounded, hashable meta-policy; no arbitrary source-code rewriting."""

    policy_id: str
    deficit_weight: float = 1.0
    uncertainty_weight: float = 0.25
    transfer_weight: float = 0.35
    novelty_weight: float = 0.20
    cost_weight: float = 0.10
    support_weight: float = 0.05
    staleness_weight: float = 0.02
    stop_threshold: float = -0.05
    max_parallel: int = 4
    worker_budget: int | None = None
    transition_budget_cap: int | None = None

    def __post_init__(self) -> None:
        if not self.policy_id:
            raise ValueError("policy_id required")
        numeric = (
            self.deficit_weight,
            self.uncertainty_weight,
            self.transfer_weight,
            self.novelty_weight,
            self.cost_weight,
            self.support_weight,
            self.staleness_weight,
            self.stop_threshold,
        )
        if not all(math.isfinite(float(x)) for x in numeric):
            raise ValueError("policy parameters must be finite")
        if self.cost_weight < 0 or self.staleness_weight < 0:
            raise ValueError("cost/staleness weights must be >= 0")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if self.worker_budget is not None and int(self.worker_budget) < 1:
            raise ValueError("worker_budget must be >= 1 when provided")
        if self.transition_budget_cap is not None and int(self.transition_budget_cap) < 1:
            raise ValueError("transition_budget_cap must be >= 1 when provided")

    def score(self, option: TrainingAllocationOption) -> float:
        return float(
            self.deficit_weight * option.capability_deficit
            + self.uncertainty_weight * option.uncertainty
            + self.transfer_weight * option.transfer_gap
            + self.novelty_weight * option.novelty
            + self.support_weight * math.log1p(option.historical_support)
            - self.cost_weight * option.expected_cost
            - self.staleness_weight * option.age_iterations
        )

    def choose_batch(self, options: Sequence[TrainingAllocationOption]) -> tuple[TrainingAllocationDecision, ...]:
        """Choose a bounded batch using the same method online and in replay.

        ``max_parallel`` caps the number of meta-decisions. ``worker_budget`` caps the
        *actual parallel worlds* requested by those decisions. ``transition_budget_cap``
        independently caps the total real environment transitions allocated by the selected
        batch.  Both are hard resource ceilings shared by online execution and replay.
        """
        scored = sorted(
            ((self.score(option), option.decision.execution_key, option.decision) for option in options),
            key=lambda row: (-row[0], row[1]),
        )
        selected: list[TrainingAllocationDecision] = []
        workers = 0
        transitions = 0
        for score, _, decision in scored:
            if len(selected) >= self.max_parallel:
                break
            if score <= self.stop_threshold:
                continue
            requested = int(decision.parallel_worlds)
            if self.worker_budget is not None and workers + requested > int(self.worker_budget):
                continue
            requested_transitions = int(decision.transition_budget)
            if (
                self.transition_budget_cap is not None
                and transitions + requested_transitions > int(self.transition_budget_cap)
            ):
                continue
            selected.append(decision)
            workers += requested
            transitions += requested_transitions
        return tuple(selected) or (TrainingAllocationDecision.stop_decision(),)

    @classmethod
    def from_dict(cls, payload: dict) -> "DeclarativeTrainingAllocationPolicy":
        return cls(**dict(payload))

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())


class DeclarativeTrainingPolicyMutator:
    """Deterministic bounded neighborhood for DREAM-RSI policy improvement."""

    _FIELDS = (
        "deficit_weight",
        "uncertainty_weight",
        "transfer_weight",
        "novelty_weight",
        "cost_weight",
        "support_weight",
        "staleness_weight",
        "stop_threshold",
    )

    def __init__(self, step: float = 0.10, fields: Iterable[str] | None = None):
        if not math.isfinite(float(step)) or step <= 0:
            raise ValueError("step must be finite and > 0")
        self.step = float(step)
        self.fields = tuple(fields or self._FIELDS)
        unknown = set(self.fields) - set(self._FIELDS)
        if unknown:
            raise ValueError(f"unknown mutation fields: {sorted(unknown)}")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("mutation fields must be unique")

    def neighbors(self, base: DeclarativeTrainingAllocationPolicy) -> tuple[DeclarativeTrainingAllocationPolicy, ...]:
        rows = [base]
        for field_name in self.fields:
            for sign in (-1.0, 1.0):
                value = float(getattr(base, field_name)) + sign * self.step
                if field_name in {"cost_weight", "staleness_weight"}:
                    value = max(0.0, value)
                rows.append(
                    replace(
                        base,
                        policy_id=f"{base.policy_id}:{field_name}:{'plus' if sign > 0 else 'minus'}",
                        **{field_name: value},
                    )
                )
        return tuple(rows)


@dataclass(frozen=True)
class MaterializedTrainingAllocation:
    """Concrete task/collector plan produced from a declarative decision."""

    decision: TrainingAllocationDecision
    tasks: tuple[object, ...]
    split: str
    difficulty: float

    def __post_init__(self) -> None:
        if self.decision.stop:
            if self.tasks:
                raise ValueError("stop allocation cannot materialize tasks")
            return
        if len(self.tasks) != self.decision.parallel_worlds:
            raise ValueError("task count must equal decision.parallel_worlds")


def materialize_training_allocation(
    decision: TrainingAllocationDecision,
    *,
    base_seed: int,
    difficulty: float,
    split: str = "train",
    index_offset: int = 0,
    validate_tasks: bool = False,
) -> MaterializedTrainingAllocation:
    """Translate the shared meta decision into real factorized Aether tasks.

    This is the online half of the shared interface. Historical replay consumes the exact same
    ``TrainingAllocationDecision`` objects, while real execution materializes them into
    `FactorizedEnvironmentFactory` tasks.
    """
    if decision.stop:
        return MaterializedTrainingAllocation(decision, (), str(split), float(difficulty))
    from awa.v2.curriculum.environment_factory import FactorizedEnvironmentFactory

    factory = FactorizedEnvironmentFactory(base_seed=int(base_seed))
    tasks = tuple(
        factory.make(
            decision.factor_signature,
            difficulty=float(difficulty),
            index=int(index_offset) + idx,
            split=str(split),
        )
        for idx in range(decision.parallel_worlds)
    )
    if validate_tasks:
        receipts = [factory.validate(task) for task in tasks]
        invalid = [receipt.task_id for receipt in receipts if not receipt.valid]
        if invalid:
            raise RuntimeError(f"materialized invalid tasks: {invalid}")
    return MaterializedTrainingAllocation(decision, tasks, str(split), float(difficulty))


def build_collector_from_decision(
    decision: TrainingAllocationDecision,
    *,
    seed: int,
    world_checkpoint: str | None = None,
    actor_checkpoint: str | None = None,
    device: str = "cpu",
):
    """Build the real collector named by a training-allocation decision.

    Aether collectors require grounded checkpoints for actor/planner modes. Missing checkpoints
    fail closed instead of silently substituting teacher or coverage collection.
    """
    if decision.stop:
        raise ValueError("stop decision has no collector")
    from awa.v2.game.collectors import CoverageArenaPolicy, FrozenAetherArenaPolicy
    from awa.v2.game.teacher import LogicalArenaTeacher, RandomArenaPolicy

    if decision.collector == "coverage":
        return CoverageArenaPolicy(seed=int(seed))
    if decision.collector == "random":
        return RandomArenaPolicy(seed=int(seed))
    if decision.collector == "teacher":
        return LogicalArenaTeacher()
    if world_checkpoint is None or actor_checkpoint is None:
        raise ValueError("Aether collector decisions require world_checkpoint and actor_checkpoint")
    return FrozenAetherArenaPolicy(
        world_checkpoint,
        actor_checkpoint,
        device=device,
        use_planner=decision.collector == "aether_planner",
        planner_budget=max(1, int(decision.planner_budget or 1)),
    )
