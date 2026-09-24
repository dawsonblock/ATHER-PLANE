from __future__ import annotations

from typing import Any, Iterable
import numpy as np

from .empirical_closure import RunRecord
from .evaluation_stats import bootstrap_ci, paired_bootstrap_ci


LADDER = (
    ("temporal_memory", "actor_only", "belief_actor"),
    ("world_model", "belief_actor", "world_actor"),
    ("fixed_planner", "world_actor", "world_planner"),
    ("adaptive_compute", "world_planner", "adaptive_compute"),
    ("grounded_hindsight_replay", "adaptive_compute", "reusable_replay"),
    ("adaptive_curriculum", "reusable_replay", "full_curriculum"),
)


def _key(r: RunRecord) -> tuple[int, str, int]:
    return (int(r.seed), str(r.split), int(r.transitions))


def _metric_map(rows: Iterable[RunRecord], system: str, metric: str) -> dict[tuple[int, str, int], float]:
    return {_key(r): float(r.metrics[metric]) for r in rows if r.system == system and metric in r.metrics}


def _curve_auc(by_key: dict[tuple[int, str, int], float], *, seed: int, split: str, milestones: tuple[int, ...]) -> float | None:
    xs = np.asarray(milestones, dtype=np.float64)
    vals = []
    for m in milestones:
        key = (int(seed), split, int(m))
        if key not in by_key:
            return None
        vals.append(by_key[key])
    y = np.asarray(vals, dtype=np.float64)
    if len(xs) == 1:
        return float(y[0])
    span = float(xs[-1] - xs[0])
    if span <= 0:
        return float(np.mean(y))
    return float(np.sum((y[1:] + y[:-1]) * 0.5 * np.diff(xs)) / span)


def _mean_ci(values, *, seed: int, resamples: int) -> dict[str, Any]:
    ci = bootstrap_ci(values, resamples=resamples, seed=seed)
    return ci.to_dict()


def _relative_reduction(candidate: np.ndarray, baseline: np.ndarray) -> float:
    denom = max(1e-9, float(np.mean(baseline)))
    return float((np.mean(baseline) - np.mean(candidate)) / denom)


def build_ablation_decision_report(records: Iterable[RunRecord], protocol: Any, config: Any) -> dict[str, Any]:
    rows = list(records)
    expected = protocol.cells()
    actual = {(r.system, r.seed, r.task, r.split, r.transitions) for r in rows}
    complete = expected == actual and len(rows) == len(expected)
    available_seeds = sorted({r.seed for r in rows})
    enough_seeds = len(available_seeds) >= int(config.minimum_seeds)
    base = {
        "format": "awa-v2.30-ablation-decision-report-v1",
        "protocol_sha256": protocol.sha256,
        "records": len(rows), "expected_records": len(expected),
        "complete": bool(complete), "minimum_seeds": int(config.minimum_seeds),
        "observed_seeds": available_seeds,
        "thresholds": {
            "minimum_final_success_gain": float(config.minimum_final_success_gain),
            "minimum_relative_curve_gain": float(config.minimum_relative_curve_gain),
            "noninferiority_margin": float(config.noninferiority_margin),
            "meaningful_latency_reduction": float(config.meaningful_latency_reduction),
            "meaningful_planner_call_reduction": float(config.meaningful_planner_call_reduction),
            "meaningful_physical_forward_reduction": float(config.meaningful_physical_forward_reduction),
            "meaningful_logical_work_reduction": float(config.meaningful_logical_work_reduction),
        },
    }
    if not complete or not enough_seeds:
        return base | {
            "status": "INSUFFICIENT_EVIDENCE",
            "reason": "The preregistered matrix is incomplete; no keep/remove decisions are issued.",
            "marginal_components": [],
        }

    milestones = tuple(sorted(int(x) for x in protocol.milestones))
    final_m = milestones[-1]
    components = []
    for idx, (mechanism, baseline, candidate) in enumerate(LADDER):
        if baseline not in protocol.systems or candidate not in protocol.systems:
            continue
        b_success = _metric_map(rows, baseline, "success_rate")
        c_success = _metric_map(rows, candidate, "success_rate")
        pair_keys = [(seed, split, final_m) for seed in protocol.seeds for split in protocol.splits]
        bv = np.asarray([b_success[k] for k in pair_keys], dtype=np.float64)
        cv = np.asarray([c_success[k] for k in pair_keys], dtype=np.float64)
        final_delta = cv - bv
        final_ci = paired_bootstrap_ci(cv, bv, resamples=config.bootstrap_resamples, seed=2280 + idx).to_dict()

        curve_b = []
        curve_c = []
        rel_curve = []
        for seed in protocol.seeds:
            for split in protocol.splits:
                ba = _curve_auc(b_success, seed=seed, split=split, milestones=milestones)
                ca = _curve_auc(c_success, seed=seed, split=split, milestones=milestones)
                if ba is None or ca is None:
                    continue
                curve_b.append(ba); curve_c.append(ca)
                rel_curve.append((ca - ba) / max(abs(ba), 1e-6))
        rel_curve_arr = np.asarray(rel_curve, dtype=np.float64)
        rel_curve_ci = _mean_ci(rel_curve_arr, seed=2380 + idx, resamples=config.bootstrap_resamples)

        b_lat = _metric_map(rows, baseline, "inference_latency_ms")
        c_lat = _metric_map(rows, candidate, "inference_latency_ms")
        latency_reduction = _relative_reduction(
            np.asarray([c_lat[k] for k in pair_keys]), np.asarray([b_lat[k] for k in pair_keys])
        ) if all(k in b_lat and k in c_lat for k in pair_keys) else 0.0
        b_plan = _metric_map(rows, baseline, "planner_calls_per_episode")
        c_plan = _metric_map(rows, candidate, "planner_calls_per_episode")
        planner_reduction = _relative_reduction(
            np.asarray([c_plan[k] for k in pair_keys]), np.asarray([b_plan[k] for k in pair_keys])
        ) if all(k in b_plan and k in c_plan for k in pair_keys) and np.mean([b_plan[k] for k in pair_keys]) > 1e-9 else 0.0
        b_phys = _metric_map(rows, baseline, "physical_world_model_forwards_per_episode")
        c_phys = _metric_map(rows, candidate, "physical_world_model_forwards_per_episode")
        physical_forward_reduction = _relative_reduction(
            np.asarray([c_phys[k] for k in pair_keys]), np.asarray([b_phys[k] for k in pair_keys])
        ) if all(k in b_phys and k in c_phys for k in pair_keys) and np.mean([b_phys[k] for k in pair_keys]) > 1e-9 else 0.0
        b_logical = _metric_map(rows, baseline, "logical_world_model_transitions_per_episode")
        c_logical = _metric_map(rows, candidate, "logical_world_model_transitions_per_episode")
        logical_work_reduction = _relative_reduction(
            np.asarray([c_logical[k] for k in pair_keys]), np.asarray([b_logical[k] for k in pair_keys])
        ) if all(k in b_logical and k in c_logical for k in pair_keys) and np.mean([b_logical[k] for k in pair_keys]) > 1e-9 else 0.0

        mean_final = float(np.mean(final_delta))
        mean_curve = float(np.mean(rel_curve_arr))
        noninferior = float(final_ci["low"]) >= -float(config.noninferiority_margin)
        capability_value = mean_final >= float(config.minimum_final_success_gain)
        sample_value = mean_curve >= float(config.minimum_relative_curve_gain)
        efficiency_value = noninferior and (
            latency_reduction >= float(config.meaningful_latency_reduction)
            or planner_reduction >= float(config.meaningful_planner_call_reduction)
            or physical_forward_reduction >= float(config.meaningful_physical_forward_reduction)
            or logical_work_reduction >= float(config.meaningful_logical_work_reduction)
        )
        # Removal requires the bootstrap upper bound to remain below the capability
        # threshold as well as no sample/compute advantage. This deliberately avoids
        # deleting a mechanism merely because a five-seed mean is noisy.
        clearly_below_capability = float(final_ci["high"]) < float(config.minimum_final_success_gain)
        if capability_value or sample_value or efficiency_value:
            decision = "KEEP"
        elif clearly_below_capability and mean_curve < float(config.minimum_relative_curve_gain) and not efficiency_value:
            decision = "REMOVE_CANDIDATE"
        else:
            decision = "UNCERTAIN"
        components.append({
            "mechanism": mechanism, "baseline": baseline, "candidate": candidate,
            "decision": decision,
            "final_success_delta_mean": mean_final,
            "final_success_delta_ci": final_ci,
            "relative_curve_gain_mean": mean_curve,
            "relative_curve_gain_ci": rel_curve_ci,
            "latency_reduction_fraction": float(latency_reduction),
            "planner_call_reduction_fraction": float(planner_reduction),
            "physical_forward_reduction_fraction": float(physical_forward_reduction),
            "logical_world_model_work_reduction_fraction": float(logical_work_reduction),
            "noninferior_success": bool(noninferior),
            "paired_final_cells": len(pair_keys),
        })

    return base | {
        "status": "QUALIFIED",
        "reason": "Preregistered matrix complete; decisions apply to this benchmark and threshold contract.",
        "marginal_components": components,
    }


def build_compute_schedule_report(records: Iterable[RunRecord], protocol: Any, config: Any) -> dict[str, Any]:
    """Summarize logical work separately from the physical execution schedule."""
    rows = list(records)
    expected = protocol.cells()
    actual = {(r.system, r.seed, r.task, r.split, r.transitions) for r in rows}
    systems = []
    for system in protocol.systems:
        sr = [r for r in rows if r.system == system]
        def vals(name):
            return np.asarray([float(r.metrics[name]) for r in sr if name in r.metrics], dtype=np.float64)
        logical = vals("logical_world_model_transitions_per_episode")
        physical = vals("physical_world_model_forwards_per_episode")
        batch = vals("mean_world_model_batch_size")
        latency = vals("inference_latency_ms")
        accel = vals("accelerator_wall_hours")
        peak = vals("peak_cuda_memory_bytes")
        systems.append({
            "system": system,
            "records": len(sr),
            "mean_logical_world_model_transitions_per_episode": float(logical.mean()) if logical.size else 0.0,
            "mean_physical_world_model_forwards_per_episode": float(physical.mean()) if physical.size else 0.0,
            "mean_world_model_batch_size": float(batch.mean()) if batch.size else 0.0,
            "logical_transitions_per_physical_forward": (float(logical.sum()/physical.sum()) if physical.size and physical.sum() > 0 else 0.0),
            "mean_inference_latency_ms": float(latency.mean()) if latency.size else 0.0,
            "mean_training_accelerator_wall_hours": float(accel.mean()) if accel.size else 0.0,
            "max_peak_cuda_memory_bytes": int(peak.max()) if peak.size else 0,
            "accelerator_energy_joules": None,
            "energy_measurement": "not_measured",
        })
    return {
        "format": "awa-v2.30-compute-schedule-report-v1",
        "status": "COMPLETE" if actual == expected and len(rows) == len(expected) else "PARTIAL",
        "protocol_sha256": protocol.sha256,
        "logical_data_budget_milestones": [int(x) for x in protocol.milestones],
        "planner_world_batch_size": int(getattr(config, "planner_world_batch_size", 0)),
        "planner_world_batch_size_semantics": "0=maximally_batched",
        "systems": systems,
        "reporting_boundary": "Energy is never estimated from elapsed time. accelerator_energy_joules remains null unless a future hardware meter records it.",
    }
