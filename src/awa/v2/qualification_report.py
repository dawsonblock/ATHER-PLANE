from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import math

from .evaluation_stats import bootstrap_ci


METRICS = (
    "actor_success_rate",
    "planner_success_rate",
    "heldout_success_rate",
    "prediction_error",
    "planner_dependency",
    "skill_reuse_rate",
    "adaptation_episodes",
    "compute_per_success",
)


@dataclass(frozen=True)
class MilestoneRow:
    transitions: int
    seed: int
    actor_success_rate: float
    planner_success_rate: float
    heldout_success_rate: float
    prediction_error: float
    planner_dependency: float
    skill_reuse_rate: float
    adaptation_episodes: float
    compute_per_success: float

    def __post_init__(self):
        if int(self.transitions) <= 0: raise ValueError("transitions must be positive")
        for name in METRICS:
            value = float(getattr(self, name))
            if not math.isfinite(value): raise ValueError(f"{name} must be finite")
        for name in ("actor_success_rate", "planner_success_rate", "heldout_success_rate", "planner_dependency", "skill_reuse_rate"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0: raise ValueError(f"{name} must be in [0,1]")
        if min(self.prediction_error, self.adaptation_episodes, self.compute_per_success) < 0:
            raise ValueError("error/adaptation/compute metrics cannot be negative")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "MilestoneRow":
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__})


DIRECTION = {
    "actor_success_rate": "up", "planner_success_rate": "up", "heldout_success_rate": "up",
    "prediction_error": "down", "planner_dependency": "down", "skill_reuse_rate": "up",
    "adaptation_episodes": "down", "compute_per_success": "down",
}


def build_milestone_report(rows: list[MilestoneRow], *, confidence: float = 0.95, resamples: int = 2000) -> dict[str, Any]:
    if not rows: raise ValueError("milestone rows required")
    groups: dict[int, list[MilestoneRow]] = {}
    for row in rows: groups.setdefault(int(row.transitions), []).append(row)
    milestones = []
    for transitions in sorted(groups):
        grp = groups[transitions]; metrics = {}
        for metric in METRICS:
            ci = bootstrap_ci([float(getattr(x, metric)) for x in grp], confidence=confidence, resamples=resamples, seed=transitions + len(metric))
            metrics[metric] = ci.to_dict()
        milestones.append({"transitions": transitions, "seeds": sorted(int(x.seed) for x in grp), "metrics": metrics})
    first, last = milestones[0], milestones[-1]
    trends = {}
    for metric in METRICS:
        start = float(first["metrics"][metric]["mean"]); end = float(last["metrics"][metric]["mean"])
        raw_delta = end - start
        desired_delta = raw_delta if DIRECTION[metric] == "up" else -raw_delta
        trends[metric] = {"direction": DIRECTION[metric], "first": start, "last": end, "raw_delta": raw_delta, "improved": desired_delta > 0}
    return {"format": "awa-v2.14-milestone-report-v1", "confidence": confidence, "milestones": milestones, "trends": trends}


def write_milestone_report(report: dict[str, Any], out_dir: str | Path) -> tuple[Path, Path]:
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    jp = out / "qualification_report.json"; mp = out / "qualification_report.md"
    jp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# Aether milestone qualification", "", "| Transitions | Actor | Planner | Held-out | Pred. error | Planner dep. | Skill reuse | Adapt episodes | Compute/success |", "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for milestone in report["milestones"]:
        m = milestone["metrics"]
        def mean(k): return float(m[k]["mean"])
        lines.append(
            f"| {milestone['transitions']:,} | {mean('actor_success_rate'):.3f} | {mean('planner_success_rate'):.3f} | "
            f"{mean('heldout_success_rate'):.3f} | {mean('prediction_error'):.4f} | {mean('planner_dependency'):.3f} | "
            f"{mean('skill_reuse_rate'):.3f} | {mean('adaptation_episodes'):.2f} | {mean('compute_per_success'):.2f} |"
        )
    lines += ["", "## Trend checks", ""]
    for metric, trend in report["trends"].items():
        mark = "PASS" if trend["improved"] else "CHECK"
        lines.append(f"- **{metric}** ({trend['direction']} desired): {trend['first']:.4f} → {trend['last']:.4f} — {mark}")
    mp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return jp, mp


def read_milestone_jsonl(path: str | Path) -> list[MilestoneRow]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip(): rows.append(MilestoneRow.from_dict(json.loads(line)))
    return rows
