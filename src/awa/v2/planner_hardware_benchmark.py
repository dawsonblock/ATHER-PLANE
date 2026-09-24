from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import statistics
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from .planners import PolicySeededMPPI


@dataclass(frozen=True)
class PlannerHardwareBenchmarkConfig:
    candidates: int = 128
    horizon: int = 8
    batch_limits: tuple[int, ...] = (0, 32, 8, 4, 1)
    warmup: int = 5
    repeats: int = 30
    seed: int = 2300

    def __post_init__(self) -> None:
        if self.candidates <= 0 or self.horizon <= 0:
            raise ValueError("candidates and horizon must be positive")
        if self.warmup < 0 or self.repeats <= 0:
            raise ValueError("warmup must be >= 0 and repeats must be > 0")
        if any(int(x) < 0 for x in self.batch_limits):
            raise ValueError("batch limits must be >= 0")

    def to_dict(self) -> dict:
        return asdict(self)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def benchmark_planner_schedules(
    world,
    actor,
    belief: torch.Tensor,
    *,
    action_low: Iterable[float],
    action_high: Iterable[float],
    config: PlannerHardwareBenchmarkConfig | None = None,
    require_cuda: bool = True,
) -> dict:
    cfg = config or PlannerHardwareBenchmarkConfig()
    device = belief.device
    if require_cuda and (device.type != "cuda" or not torch.cuda.is_available()):
        raise RuntimeError("hardware planner benchmark requires a CUDA device unless require_cuda=False")

    rows = []
    action_fingerprints: dict[int, str] = {}
    action_sequences: dict[int, torch.Tensor] = {}
    for batch_limit in cfg.batch_limits:
        planner = PolicySeededMPPI(
            world,
            actor,
            list(action_low),
            list(action_high),
            horizon=cfg.horizon,
            candidates=cfg.candidates,
            world_batch_size=(None if int(batch_limit) == 0 else int(batch_limit)),
        )
        for warm in range(cfg.warmup):
            torch.manual_seed(cfg.seed + warm)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(cfg.seed + warm)
            planner.plan(belief, budget=cfg.candidates)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        latencies: list[float] = []
        physical: list[int] = []
        logical: list[int] = []
        batch_means: list[float] = []
        actions: list[torch.Tensor] = []
        for repeat in range(cfg.repeats):
            run_seed = cfg.seed + 10_000 + repeat
            torch.manual_seed(run_seed)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(run_seed)
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            result = planner.plan(belief, budget=cfg.candidates)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            latencies.append(float(elapsed_ms))
            physical.append(int(result.metadata.get("physical_world_model_forwards", 0)))
            logical.append(int(result.metadata.get("logical_world_model_transitions", result.world_model_calls)))
            batch_means.append(float(result.metadata.get("mean_world_model_batch_size", 0.0)))
            actions.append(result.action.detach().cpu())

        action_blob = torch.stack(actions).numpy().tobytes()
        action_hash = hashlib.sha256(action_blob).hexdigest()
        action_fingerprints[int(batch_limit)] = action_hash
        action_sequences[int(batch_limit)] = torch.stack(actions)
        peak_memory = (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" and torch.cuda.is_available()
            else 0
        )
        rows.append(
            {
                "planner_world_batch_size": int(batch_limit),
                "schedule_semantics": "maximally_batched" if int(batch_limit) == 0 else "microbatched",
                "median_latency_ms": float(statistics.median(latencies)),
                "p95_latency_ms": _percentile(latencies, 95),
                "mean_latency_ms": float(statistics.fmean(latencies)),
                "mean_logical_world_model_transitions": float(statistics.fmean(logical)),
                "mean_physical_world_model_forwards": float(statistics.fmean(physical)),
                "mean_world_model_batch_size": float(statistics.fmean(batch_means)),
                "logical_transitions_per_physical_forward": (
                    float(sum(logical) / max(1, sum(physical)))
                ),
                "peak_cuda_memory_bytes": peak_memory,
                "action_sequence_sha256": action_hash,
            }
        )

    logical_values = {round(row["mean_logical_world_model_transitions"], 9) for row in rows}
    reference_limit = int(cfg.batch_limits[0])
    reference_actions = action_sequences[reference_limit]
    max_action_deviation = 0.0
    for row in rows:
        limit = int(row["planner_world_batch_size"])
        deviation = float((action_sequences[limit] - reference_actions).abs().max().item())
        row["max_action_abs_deviation_from_reference"] = deviation
        row["action_numerically_equivalent"] = bool(deviation <= 1e-6)
        max_action_deviation = max(max_action_deviation, deviation)
        mean_latency_s = max(float(row["mean_latency_ms"]) / 1000.0, 1e-12)
        row["logical_transitions_per_second"] = float(
            row["mean_logical_world_model_transitions"] / mean_latency_s
        )
        row["physical_forwards_per_second"] = float(
            row["mean_physical_world_model_forwards"] / mean_latency_s
        )

    cuda_properties = None
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        cuda_properties = {
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": int(props.total_memory),
            "compute_capability": list(torch.cuda.get_device_capability(device)),
        }
    report = {
        "format": "awa-v2.35-planner-hardware-benchmark-v2",
        "device": str(device),
        "cuda_device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" and torch.cuda.is_available() else None
        ),
        "cuda_properties": cuda_properties,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "config": cfg.to_dict(),
        "logical_work_matched": len(logical_values) == 1,
        "action_schedule_equivalent": bool(max_action_deviation <= 1e-6),
        "max_action_abs_deviation": float(max_action_deviation),
        "action_fingerprints": {str(k): v for k, v in sorted(action_fingerprints.items())},
        "rows": rows,
        "energy_measurement": "not_measured",
        "accelerator_energy_joules": None,
        "claim_boundary": (
            "This benchmark measures execution scheduling on the current device. It does not infer energy from TDP or wall time. "
            "Matched logical work and numerical action equivalence are reported separately from physical latency/memory."
        ),
    }
    return report


def write_hardware_benchmark(report: dict, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
