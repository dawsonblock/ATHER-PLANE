from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable
import math
import numpy as np

from .evaluation_stats import bootstrap_ci, paired_bootstrap_ci


@dataclass(frozen=True)
class AblationSpec:
    name: str
    temporal_memory: bool = True
    planner: bool = True
    skills: bool = True
    ensemble: bool = True
    prioritized_replay: bool = True
    counterfactual: bool = True
    hindsight: bool = True
    voc: bool = True

    def __post_init__(self):
        if not self.name: raise ValueError("ablation name cannot be empty")

    def to_dict(self): return asdict(self)


DEFAULT_ABLATIONS = (
    AblationSpec("full"),
    AblationSpec("no_temporal", temporal_memory=False),
    AblationSpec("no_planner", planner=False),
    AblationSpec("no_skills", skills=False),
    AblationSpec("no_ensemble", ensemble=False),
    AblationSpec("no_priority", prioritized_replay=False),
    AblationSpec("no_counterfactual", counterfactual=False),
    AblationSpec("no_hindsight", hindsight=False),
    AblationSpec("no_voc", voc=False),
)


def run_ablation_suite(
    evaluator: Callable[[AblationSpec, int], dict[str, float]],
    *, seeds: list[int], specs: tuple[AblationSpec, ...] = DEFAULT_ABLATIONS,
    primary_metric: str = "success_rate", resamples: int = 1000,
) -> dict[str, Any]:
    if not seeds: raise ValueError("at least one seed is required")
    if not specs or specs[0].name != "full": raise ValueError("first ablation must be full baseline")
    raw: dict[str, list[dict[str, float]]] = {}
    for spec in specs:
        rows = []
        for seed in seeds:
            metrics = {str(k): float(v) for k, v in evaluator(spec, int(seed)).items()}
            if primary_metric not in metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise ValueError("ablation evaluator returned invalid metrics")
            rows.append(metrics)
        raw[spec.name] = rows
    baseline = np.asarray([r[primary_metric] for r in raw["full"]], dtype=np.float64)
    summary = {}
    for spec in specs:
        vals = np.asarray([r[primary_metric] for r in raw[spec.name]], dtype=np.float64)
        ci = bootstrap_ci(vals, resamples=resamples, seed=sum(seeds) + len(spec.name))
        delta = paired_bootstrap_ci(vals, baseline, resamples=resamples, seed=sum(seeds) + 17 + len(spec.name))
        summary[spec.name] = {"spec": spec.to_dict(), "primary": ci.to_dict(), "delta_vs_full": delta.to_dict()}
    return {"format": "awa-v2.14-ablation-report-v1", "primary_metric": primary_metric, "seeds": list(map(int, seeds)), "raw": raw, "summary": summary}


@dataclass(frozen=True)
class EmpiricalAblationSpec:
    """Preregistered executable component configuration for v2.29.

    The fields mirror mechanisms that have a concrete runtime in the procedural
    ablation harness. Broad experimental families such as skill discovery or MoE
    are intentionally excluded until they have their own isolated executable path.
    """
    ablation_id: str
    representation: str
    world_model: bool = False
    fixed_planner: bool = False
    adaptive_compute: bool = False
    hindsight_replay: bool = False
    adaptive_curriculum: bool = False

    def __post_init__(self):
        if not self.ablation_id:
            raise ValueError("ablation_id required")
        if self.representation not in {"raw", "belief", "world"}:
            raise ValueError("representation must be raw/belief/world")
        if self.world_model != (self.representation == "world"):
            raise ValueError("world_model must match representation")
        if (self.fixed_planner or self.adaptive_compute) and not self.world_model:
            raise ValueError("planning requires world_model")
        if self.fixed_planner and self.adaptive_compute:
            raise ValueError("fixed_planner and adaptive_compute are mutually exclusive")

    def to_dict(self):
        return asdict(self)

    @property
    def sha256(self) -> str:
        import hashlib, json
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(",",":")).encode()
        return hashlib.sha256(raw).hexdigest()


EMPIRICAL_ABLATIONS = (
    EmpiricalAblationSpec("actor_only", "raw"),
    EmpiricalAblationSpec("belief_actor", "belief"),
    EmpiricalAblationSpec("world_actor", "world", world_model=True),
    EmpiricalAblationSpec("world_planner", "world", world_model=True, fixed_planner=True),
    EmpiricalAblationSpec("adaptive_compute", "world", world_model=True, adaptive_compute=True),
    EmpiricalAblationSpec("reusable_replay", "world", world_model=True, adaptive_compute=True, hindsight_replay=True),
    EmpiricalAblationSpec("full_curriculum", "world", world_model=True, adaptive_compute=True, hindsight_replay=True, adaptive_curriculum=True),
)


def build_empirical_ablation_protocol(*, seeds, tasks=("procedural",), splits=("heldout","transfer"), milestones=(25000,100000), minimum_seeds=5):
    """Bind every executable ablation configuration hash before results are accepted."""
    from .experiment_protocol import ExperimentProtocol
    specs=EMPIRICAL_ABLATIONS
    return ExperimentProtocol(
        protocol_id="aether-v2.31-executable-component-ablation-v1",
        systems=tuple(s.ablation_id for s in specs),
        seeds=tuple(int(x) for x in seeds),
        tasks=tuple(str(x) for x in tasks),
        splits=tuple(str(x) for x in splits),
        milestones=tuple(int(x) for x in milestones),
        primary_metric="success_rate",
        minimum_seeds=int(minimum_seeds),
        system_config_sha256=tuple((s.ablation_id,s.sha256) for s in specs),
    )
