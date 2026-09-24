"""Explicit ownership and dependency contract for Aether v2.31.

The project has three stable layers:

* agent_runtime: inference-time cognition and action selection.
* learning_system: data, environment generation, replay, curriculum, and training.
* research_os: experiment governance, provenance, ablation, reporting, and release evidence.

Experimental families are outside this contract and must not be pulled into the
canonical agent runtime implicitly.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable


class SystemLayer(StrEnum):
    AGENT_RUNTIME = "agent_runtime"
    LEARNING_SYSTEM = "learning_system"
    RESEARCH_OS = "research_os"
    EXPERIMENTAL = "experimental"
    SHARED = "shared"


@dataclass(frozen=True)
class SplitViolation:
    source: str
    source_layer: str
    target: str
    target_layer: str
    reason: str

    def to_dict(self) -> dict:
        return asdict(self)


# Ownership is intentionally explicit.  Prefixes are longest-match wins.
OWNERSHIP: tuple[tuple[str, SystemLayer], ...] = (
    ("awa.v2.agent_runtime", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.agent", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.world", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.temporal", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.representation", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.video_representation", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.planners", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.voc", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.safety", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.reasoning", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.game.action_codec", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.game.belief", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.game.controller", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.game.hybrid_control", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.game.vizdoom_runtime", SystemLayer.AGENT_RUNTIME),
    ("awa.v2.game.vizdoom_semantics", SystemLayer.AGENT_RUNTIME),

    ("awa.v2.learning_system", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.actor_baseline", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.datasets", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.world_training", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.risk_training", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.representation_train", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.curriculum", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.replay", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.goals", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.collectors", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.dataset", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.training", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.her", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.teacher", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.procedural_arena", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.procedural_runtime", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.vizdoom_dataset", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.vizdoom_training", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.vizdoom_vjepa", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.vizdoom_campaign", SystemLayer.LEARNING_SYSTEM),
    ("awa.v2.game.voc_training", SystemLayer.LEARNING_SYSTEM),

    ("awa.v2.research_os", SystemLayer.RESEARCH_OS),
    ("awa.v2.ablation", SystemLayer.RESEARCH_OS),
    ("awa.v2.ablation_campaign", SystemLayer.RESEARCH_OS),
    ("awa.v2.ablation_report", SystemLayer.RESEARCH_OS),
    ("awa.v2.benchmark_qualification", SystemLayer.RESEARCH_OS),
    ("awa.v2.benchmark_tracks", SystemLayer.RESEARCH_OS),
    ("awa.v2.campaign", SystemLayer.RESEARCH_OS),
    ("awa.v2.checkpoints", SystemLayer.RESEARCH_OS),
    ("awa.v2.compute_ledger", SystemLayer.RESEARCH_OS),
    ("awa.v2.empirical_closure", SystemLayer.RESEARCH_OS),
    ("awa.v2.empirical_status", SystemLayer.RESEARCH_OS),
    ("awa.v2.game.vizdoom_qualification", SystemLayer.RESEARCH_OS),
    ("awa.v2.evaluation_stats", SystemLayer.RESEARCH_OS),
    ("awa.v2.evidence_integrity", SystemLayer.RESEARCH_OS),
    ("awa.v2.evolving_campaign", SystemLayer.RESEARCH_OS),
    ("awa.v2.experiment_ledger", SystemLayer.RESEARCH_OS),
    ("awa.v2.experiment_protocol", SystemLayer.RESEARCH_OS),
    ("awa.v2.failure_triage", SystemLayer.RESEARCH_OS),
    ("awa.v2.milestone_analysis", SystemLayer.RESEARCH_OS),
    ("awa.v2.native_campaign", SystemLayer.RESEARCH_OS),
    ("awa.v2.planner_hardware_benchmark", SystemLayer.RESEARCH_OS),
    ("awa.v2.planner_qualification", SystemLayer.RESEARCH_OS),
    ("awa.v2.promotion", SystemLayer.RESEARCH_OS),
    ("awa.v2.qualification_report", SystemLayer.RESEARCH_OS),
    ("awa.v2.telemetry", SystemLayer.RESEARCH_OS),

    ("awa.v2.experimental", SystemLayer.EXPERIMENTAL),
    ("awa.v2.engram", SystemLayer.EXPERIMENTAL),
    ("awa.v2.ensemble", SystemLayer.EXPERIMENTAL),
    ("awa.v2.modular_world", SystemLayer.EXPERIMENTAL),
    ("awa.v2.skills", SystemLayer.EXPERIMENTAL),
    ("awa.v2.distill", SystemLayer.EXPERIMENTAL),
    ("awa.v2.training.reusable_engine", SystemLayer.EXPERIMENTAL),
)

_ALLOWED: dict[SystemLayer, frozenset[SystemLayer]] = {
    SystemLayer.AGENT_RUNTIME: frozenset({SystemLayer.AGENT_RUNTIME, SystemLayer.SHARED}),
    SystemLayer.LEARNING_SYSTEM: frozenset({SystemLayer.AGENT_RUNTIME, SystemLayer.LEARNING_SYSTEM, SystemLayer.SHARED}),
    SystemLayer.RESEARCH_OS: frozenset({SystemLayer.AGENT_RUNTIME, SystemLayer.LEARNING_SYSTEM, SystemLayer.RESEARCH_OS, SystemLayer.SHARED}),
    SystemLayer.EXPERIMENTAL: frozenset(SystemLayer),
    SystemLayer.SHARED: frozenset({SystemLayer.SHARED}),
}


def classify_module(module_name: str) -> SystemLayer:
    matches = [(prefix, layer) for prefix, layer in OWNERSHIP if module_name == prefix or module_name.startswith(prefix + ".")]
    if not matches:
        return SystemLayer.SHARED
    return max(matches, key=lambda x: len(x[0]))[1]


def dependency_allowed(source: SystemLayer, target: SystemLayer) -> bool:
    return target in _ALLOWED[source]


def _module_name(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _imports(path: Path) -> Iterable[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for name in node.names:
                yield name.name
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module


def validate_split_packages(src_root: str | Path) -> list[SplitViolation]:
    """Validate only the canonical split packages.

    Historical flat modules remain compatibility implementation details in v2.31;
    new canonical code must enter through one of these three packages.  This keeps
    the migration bounded while enforcing dependency direction on the new surface.
    """
    root = Path(src_root)
    package_roots = (
        root / "awa" / "v2" / "agent_runtime",
        root / "awa" / "v2" / "learning_system",
        root / "awa" / "v2" / "research_os",
    )
    violations: list[SplitViolation] = []
    for package_root in package_roots:
        for path in sorted(package_root.rglob("*.py")):
            source = _module_name(path, root)
            source_layer = classify_module(source)
            for target in _imports(path):
                if not target.startswith("awa.v2"):
                    continue
                target_layer = classify_module(target)
                if not dependency_allowed(source_layer, target_layer):
                    violations.append(SplitViolation(
                        source=source,
                        source_layer=source_layer.value,
                        target=target,
                        target_layer=target_layer.value,
                        reason=f"{source_layer.value} may not depend on {target_layer.value}",
                    ))
    return violations


def split_manifest() -> dict:
    return {
        "format": "awa-v2.31-system-split-v1",
        "layers": {
            SystemLayer.AGENT_RUNTIME.value: {
                "purpose": "inference-time cognition and action selection",
                "may_depend_on": sorted(x.value for x in _ALLOWED[SystemLayer.AGENT_RUNTIME]),
            },
            SystemLayer.LEARNING_SYSTEM.value: {
                "purpose": "experience, datasets, replay, environment generation, curriculum, and training",
                "may_depend_on": sorted(x.value for x in _ALLOWED[SystemLayer.LEARNING_SYSTEM]),
            },
            SystemLayer.RESEARCH_OS.value: {
                "purpose": "protocols, provenance, ablations, compute accounting, qualification, and release evidence",
                "may_depend_on": sorted(x.value for x in _ALLOWED[SystemLayer.RESEARCH_OS]),
            },
            SystemLayer.EXPERIMENTAL.value: {
                "purpose": "quarantined hypotheses that are not part of the canonical agent",
                "may_depend_on": sorted(x.value for x in _ALLOWED[SystemLayer.EXPERIMENTAL]),
            },
        },
    }
