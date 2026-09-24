from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import hashlib
import json
import math

import numpy as np

from .empirical_closure import RunRecord
from .evaluation_stats import bootstrap_ci


def _sha(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _record_key(record: RunRecord) -> tuple[str, int, str, str, int]:
    return (record.system, record.seed, record.task, record.split, record.transitions)


def _normalized_auc(milestones: list[int], values: list[float]) -> float:
    if not milestones or len(milestones) != len(values):
        raise ValueError("milestones/values must be non-empty and aligned")
    x = np.asarray(milestones, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("milestone curve contains non-finite values")
    if len(x) == 1:
        return float(y[0])
    if np.any(np.diff(x) <= 0):
        raise ValueError("milestones must be strictly increasing")
    span = float(x[-1] - x[0])
    if span <= 0:
        raise ValueError("milestone span must be positive")
    # Area normalized by the transition span, so the score remains in metric units.
    area = np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) * 0.5)
    return float(area / span)


@dataclass(frozen=True)
class MilestonePromotionConfig:
    minimum_final_mean_gain: float = 0.0
    minimum_curve_mean_gain: float = 0.0
    maximum_seed_final_regression: float = 0.05
    maximum_split_final_regression: float = 0.02
    confidence: float = 0.95
    bootstrap_resamples: int = 4000
    require_positive_final_ci: bool = False

    def __post_init__(self):
        for name in (
            "minimum_final_mean_gain",
            "minimum_curve_mean_gain",
            "maximum_seed_final_regression",
            "maximum_split_final_regression",
            "confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.maximum_seed_final_regression < 0 or self.maximum_split_final_regression < 0:
            raise ValueError("regression tolerances must be >= 0")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0,1)")
        if int(self.bootstrap_resamples) < 200:
            raise ValueError("bootstrap_resamples must be >= 200")


@dataclass(frozen=True)
class MilestonePromotionReceipt:
    status: str
    candidate_policy_id: str
    baseline_policy_id: str
    seeds: tuple[int, ...]
    final_milestone: int
    mean_paired_gain: float
    worst_paired_gain: float
    curve_mean_paired_gain: float
    final_ci_low: float
    final_ci_high: float
    split_final_gains: dict[str, float]
    failures: tuple[str, ...]
    scorecard_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_milestone_scorecard(
    records: Iterable[RunRecord],
    *,
    baseline: str,
    candidate: str,
    primary_metric: str = "success_rate",
    minimum_seeds: int = 5,
    confidence: float = 0.95,
    resamples: int = 4000,
) -> dict[str, Any]:
    """Build a paired learning-curve scorecard over a complete preregistered matrix.

    The scorecard intentionally separates three questions that older promotion logic
    collapsed into one average: final performance, sample-efficiency across milestones,
    and split-specific regressions. Structural incompleteness fails closed.
    """
    rows = [r for r in records if r.system in {baseline, candidate}]
    failures: list[str] = []
    if baseline == candidate:
        failures.append("baseline and candidate must differ")
    if not rows:
        failures.append("no records for baseline/candidate")

    systems = {r.system for r in rows}
    if baseline not in systems:
        failures.append("baseline records missing")
    if candidate not in systems:
        failures.append("candidate records missing")

    seeds = tuple(sorted({r.seed for r in rows}))
    tasks = tuple(sorted({r.task for r in rows}))
    splits = tuple(sorted({r.split for r in rows}))
    milestones = tuple(sorted({int(r.transitions) for r in rows}))
    if len(seeds) < int(minimum_seeds):
        failures.append(f"paired seeds {len(seeds)} < required {minimum_seeds}")
    if not milestones:
        failures.append("no milestones")

    by: dict[tuple[str, int, str, str, int], RunRecord] = {}
    duplicates: list[tuple[str, int, str, str, int]] = []
    for row in rows:
        key = _record_key(row)
        if key in by:
            duplicates.append(key)
        by[key] = row
    if duplicates:
        failures.append(f"duplicate scorecard cells: {len(duplicates)}")

    expected = {
        (system, seed, task, split, milestone)
        for system in (baseline, candidate)
        for seed in seeds
        for task in tasks
        for split in splits
        for milestone in milestones
    }
    missing = sorted(expected - set(by))
    if missing:
        failures.append(f"missing scorecard cells: {len(missing)}")

    metric_missing = [key for key, row in by.items() if primary_metric not in row.metrics]
    if metric_missing:
        failures.append(f"primary metric missing from cells: {len(metric_missing)}")

    structure_ok = not failures
    per_milestone: list[dict[str, Any]] = []
    final_seed_gains: dict[int, float] = {}
    curve_seed_gains: dict[int, float] = {}
    split_final_gains: dict[str, float] = {}
    planner_dependence: dict[str, Any] | None = None

    if structure_ok:
        final_milestone = int(milestones[-1])
        for milestone in milestones:
            deltas: list[float] = []
            for seed in seeds:
                for task in tasks:
                    for split in splits:
                        b = by[(baseline, seed, task, split, milestone)]
                        c = by[(candidate, seed, task, split, milestone)]
                        deltas.append(float(c.metrics[primary_metric]) - float(b.metrics[primary_metric]))
            ci = bootstrap_ci(deltas, confidence=confidence, resamples=resamples, seed=9100 + int(milestone) % 1009)
            per_milestone.append({
                "transitions": int(milestone),
                "paired_delta": ci.to_dict(),
            })

        for seed in seeds:
            final_deltas: list[float] = []
            curve_deltas: list[float] = []
            for task in tasks:
                for split in splits:
                    b_values = [float(by[(baseline, seed, task, split, m)].metrics[primary_metric]) for m in milestones]
                    c_values = [float(by[(candidate, seed, task, split, m)].metrics[primary_metric]) for m in milestones]
                    final_deltas.append(c_values[-1] - b_values[-1])
                    curve_deltas.append(
                        _normalized_auc(list(milestones), c_values)
                        - _normalized_auc(list(milestones), b_values)
                    )
            final_seed_gains[int(seed)] = float(np.mean(final_deltas))
            curve_seed_gains[int(seed)] = float(np.mean(curve_deltas))

        for split in splits:
            deltas = []
            for seed in seeds:
                for task in tasks:
                    b = by[(baseline, seed, task, split, final_milestone)]
                    c = by[(candidate, seed, task, split, final_milestone)]
                    deltas.append(float(c.metrics[primary_metric]) - float(b.metrics[primary_metric]))
            split_final_gains[split] = float(np.mean(deltas))

        # Planner dependence is diagnostic only; missing metrics never block promotion.
        planner_metric = "planner_calls_per_episode"
        if all(planner_metric in row.metrics for row in rows):
            planner_dependence = {"baseline": [], "candidate": []}
            for system in (baseline, candidate):
                bucket = planner_dependence["baseline" if system == baseline else "candidate"]
                for milestone in milestones:
                    vals = [
                        float(by[(system, seed, task, split, milestone)].metrics[planner_metric])
                        for seed in seeds for task in tasks for split in splits
                    ]
                    bucket.append({"transitions": int(milestone), "mean": float(np.mean(vals))})

        final_ci = bootstrap_ci(
            list(final_seed_gains.values()),
            confidence=confidence,
            resamples=resamples,
            seed=9241,
        )
        curve_ci = bootstrap_ci(
            list(curve_seed_gains.values()),
            confidence=confidence,
            resamples=resamples,
            seed=9257,
        )
        aggregates = {
            "final_milestone": final_milestone,
            "final_seed_paired_gain": final_seed_gains,
            "final_paired_gain": final_ci.to_dict(),
            "curve_seed_paired_gain": curve_seed_gains,
            "curve_paired_gain": curve_ci.to_dict(),
            "split_final_gains": split_final_gains,
        }
    else:
        aggregates = {
            "final_milestone": int(milestones[-1]) if milestones else 0,
            "final_seed_paired_gain": {},
            "final_paired_gain": None,
            "curve_seed_paired_gain": {},
            "curve_paired_gain": None,
            "split_final_gains": {},
        }

    body: dict[str, Any] = {
        "format": "awa-v2.24-milestone-scorecard-v1",
        "status": "PASS" if structure_ok else "FAIL",
        "baseline": baseline,
        "candidate": candidate,
        "primary_metric": primary_metric,
        "minimum_seeds": int(minimum_seeds),
        "seeds": list(seeds),
        "tasks": list(tasks),
        "splits": list(splits),
        "milestones": list(milestones),
        "per_milestone": per_milestone,
        "aggregates": aggregates,
        "planner_dependence": planner_dependence,
        "missing_cells": [list(x) for x in missing],
        "failures": failures,
    }
    body["scorecard_sha256"] = _sha(body)
    return body


def evaluate_milestone_promotion(
    scorecard: dict[str, Any],
    config: MilestonePromotionConfig | None = None,
) -> MilestonePromotionReceipt:
    cfg = config or MilestonePromotionConfig()
    failures = list(scorecard.get("failures") or [])
    if scorecard.get("status") != "PASS":
        failures.append("milestone scorecard did not pass structural checks")

    agg = scorecard.get("aggregates") or {}
    final = agg.get("final_paired_gain") or {}
    curve = agg.get("curve_paired_gain") or {}
    final_seed = {int(k): float(v) for k, v in (agg.get("final_seed_paired_gain") or {}).items()}
    split_final = {str(k): float(v) for k, v in (agg.get("split_final_gains") or {}).items()}

    mean_gain = float(final.get("mean", float("-inf")))
    low = float(final.get("low", float("-inf")))
    high = float(final.get("high", float("-inf")))
    curve_gain = float(curve.get("mean", float("-inf")))
    worst_gain = min(final_seed.values()) if final_seed else float("-inf")

    if not failures:
        if mean_gain <= cfg.minimum_final_mean_gain:
            failures.append("candidate final mean paired gain did not clear promotion threshold")
        if curve_gain < cfg.minimum_curve_mean_gain:
            failures.append("candidate learning-curve gain regressed below threshold")
        if worst_gain < -abs(cfg.maximum_seed_final_regression):
            failures.append("candidate exceeded allowed final per-seed regression")
        bad_splits = {
            split: gain
            for split, gain in split_final.items()
            if gain < -abs(cfg.maximum_split_final_regression)
        }
        if bad_splits:
            failures.append("candidate exceeded allowed final split regression: " + ",".join(sorted(bad_splits)))
        if cfg.require_positive_final_ci and low <= 0.0:
            failures.append("candidate final paired confidence interval does not exclude zero")

    return MilestonePromotionReceipt(
        status="PASS" if not failures else "FAIL",
        candidate_policy_id=str(scorecard.get("candidate", "")),
        baseline_policy_id=str(scorecard.get("baseline", "")),
        seeds=tuple(sorted(final_seed)),
        final_milestone=int(agg.get("final_milestone", 0)),
        mean_paired_gain=mean_gain,
        worst_paired_gain=float(worst_gain),
        curve_mean_paired_gain=curve_gain,
        final_ci_low=low,
        final_ci_high=high,
        split_final_gains=split_final,
        failures=tuple(failures),
        scorecard_sha256=str(scorecard.get("scorecard_sha256", "")),
    )
