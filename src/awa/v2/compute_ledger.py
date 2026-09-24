from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable
import math


def chunk_schedule(total: int, max_batch_size: int | None = None) -> tuple[int, ...]:
    """Return the physical call schedule for ``total`` independent items.

    ``None`` or ``0`` means one maximally batched call.  A positive limit creates
    the minimum number of sequential calls needed to cover the same logical work.
    The function is intentionally hardware-agnostic: it describes execution, not
    an estimate of latency or energy.
    """
    total = int(total)
    if total < 0:
        raise ValueError("total must be >= 0")
    if total == 0:
        return ()
    if max_batch_size in (None, 0):
        return (total,)
    limit = int(max_batch_size)
    if limit <= 0:
        raise ValueError("max_batch_size must be positive, 0, or None")
    full, rem = divmod(total, limit)
    out = [limit] * full
    if rem:
        out.append(rem)
    return tuple(out)


@dataclass(frozen=True)
class PlannerExecutionFingerprint:
    """Physical schedule for a logically equivalent planner workload.

    The distinction mirrors the systems point in arXiv:2609.19499: a logical
    candidate/sample count does not identify how that work was grouped into
    physical calls.  For Aether, the analogous quantities are imagined world-model
    transitions and actual ``world.imagine_step`` invocations with a batch size.
    """

    logical_world_model_transitions: int = 0
    physical_world_model_forwards: int = 0
    batch_size_histogram: tuple[tuple[int, int], ...] = ()
    max_batch_size: int = 0
    mean_batch_size: float = 0.0
    logical_transitions_per_forward: float = 0.0
    world_batch_limit: int | None = None
    schedule_kind: str = "none"

    @classmethod
    def from_batches(
        cls,
        batches: Iterable[int],
        *,
        logical_world_model_transitions: int | None = None,
        world_batch_limit: int | None = None,
        schedule_kind: str = "batched",
    ) -> "PlannerExecutionFingerprint":
        rows = tuple(int(x) for x in batches if int(x) > 0)
        logical = int(sum(rows) if logical_world_model_transitions is None else logical_world_model_transitions)
        physical = len(rows)
        hist = tuple(sorted((int(k), int(v)) for k, v in Counter(rows).items()))
        mean = float(sum(rows) / physical) if physical else 0.0
        return cls(
            logical_world_model_transitions=logical,
            physical_world_model_forwards=physical,
            batch_size_histogram=hist,
            max_batch_size=max(rows, default=0),
            mean_batch_size=mean,
            logical_transitions_per_forward=(float(logical) / physical if physical else 0.0),
            world_batch_limit=(None if world_batch_limit in (None, 0) else int(world_batch_limit)),
            schedule_kind=str(schedule_kind),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["batch_size_histogram"] = [list(x) for x in self.batch_size_histogram]
        return payload

    def plus(self, other: "PlannerExecutionFingerprint", *, schedule_kind: str | None = None) -> "PlannerExecutionFingerprint":
        hist: Counter[int] = Counter(dict(self.batch_size_histogram))
        hist.update(dict(other.batch_size_histogram))
        physical = self.physical_world_model_forwards + other.physical_world_model_forwards
        logical = self.logical_world_model_transitions + other.logical_world_model_transitions
        weighted = sum(k * v for k, v in hist.items())
        return PlannerExecutionFingerprint(
            logical_world_model_transitions=int(logical),
            physical_world_model_forwards=int(physical),
            batch_size_histogram=tuple(sorted((int(k), int(v)) for k, v in hist.items())),
            max_batch_size=max(self.max_batch_size, other.max_batch_size),
            mean_batch_size=(float(weighted) / physical if physical else 0.0),
            logical_transitions_per_forward=(float(logical) / physical if physical else 0.0),
            world_batch_limit=self.world_batch_limit if self.world_batch_limit == other.world_batch_limit else None,
            schedule_kind=schedule_kind or (self.schedule_kind if self.schedule_kind == other.schedule_kind else "mixed"),
        )


@dataclass
class PhysicalComputeLedger:
    """Aggregate measured/observed compute without inventing unavailable energy data."""

    planner_calls: int = 0
    logical_world_model_transitions: int = 0
    physical_world_model_forwards: int = 0
    batch_histogram: Counter[int] = field(default_factory=Counter)
    inference_wall_seconds: float = 0.0
    training_wall_seconds: float = 0.0
    peak_cuda_memory_bytes: int = 0
    accelerator_energy_joules: float | None = None
    energy_measurement: str = "not_measured"

    def add_planner_result(self, result: Any, *, elapsed_seconds: float = 0.0) -> None:
        self.planner_calls += 1
        metadata = dict(getattr(result, "metadata", {}) or {})
        logical = int(metadata.get("logical_world_model_transitions", getattr(result, "world_model_calls", 0)))
        physical = int(metadata.get("physical_world_model_forwards", 0))
        self.logical_world_model_transitions += max(0, logical)
        self.physical_world_model_forwards += max(0, physical)
        for row in metadata.get("batch_size_histogram", []) or []:
            if isinstance(row, (list, tuple)) and len(row) == 2:
                self.batch_histogram[int(row[0])] += int(row[1])
        self.inference_wall_seconds += max(0.0, float(elapsed_seconds))

    def add_trace(self, trace: Any) -> None:
        if trace is None:
            return
        self.planner_calls += int(getattr(trace, "planner", "actor") != "actor")
        self.logical_world_model_transitions += max(0, int(getattr(trace, "logical_world_model_transitions", getattr(trace, "world_model_calls", 0))))
        self.physical_world_model_forwards += max(0, int(getattr(trace, "physical_world_model_forwards", 0)))
        max_batch = max(0, int(getattr(trace, "max_world_batch_size", 0)))
        physical = max(0, int(getattr(trace, "physical_world_model_forwards", 0)))
        if max_batch and physical:
            # A trace does not retain the complete histogram.  Record an upper-bound
            # bucket only when detailed planner metadata is unavailable.
            self.batch_histogram[max_batch] += physical
        self.inference_wall_seconds += max(0.0, float(getattr(trace, "latency_ms", 0.0))) / 1000.0

    @property
    def mean_batch_size(self) -> float:
        calls = sum(self.batch_histogram.values())
        return float(sum(k * v for k, v in self.batch_histogram.items()) / calls) if calls else 0.0

    @property
    def logical_transitions_per_forward(self) -> float:
        return (
            float(self.logical_world_model_transitions) / self.physical_world_model_forwards
            if self.physical_world_model_forwards else 0.0
        )

    def to_dict(self) -> dict[str, Any]:
        if self.accelerator_energy_joules is not None and not math.isfinite(float(self.accelerator_energy_joules)):
            raise ValueError("accelerator_energy_joules must be finite when measured")
        return {
            "format": "awa-v2.29-physical-compute-ledger-v1",
            "planner_calls": int(self.planner_calls),
            "logical_world_model_transitions": int(self.logical_world_model_transitions),
            "physical_world_model_forwards": int(self.physical_world_model_forwards),
            "batch_size_histogram": [[int(k), int(v)] for k, v in sorted(self.batch_histogram.items())],
            "max_world_batch_size": max(self.batch_histogram, default=0),
            "mean_world_batch_size": float(self.mean_batch_size),
            "logical_transitions_per_forward": float(self.logical_transitions_per_forward),
            "inference_wall_seconds": float(self.inference_wall_seconds),
            "training_wall_seconds": float(self.training_wall_seconds),
            "accelerator_wall_hours": float((self.training_wall_seconds + self.inference_wall_seconds) / 3600.0),
            "peak_cuda_memory_bytes": int(self.peak_cuda_memory_bytes),
            "accelerator_energy_joules": self.accelerator_energy_joules,
            "energy_measurement": self.energy_measurement,
        }
