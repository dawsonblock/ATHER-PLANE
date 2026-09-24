from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Hashable, Iterable, Protocol
import hashlib
import json
import math
import os
import subprocess
import tempfile

from .empirical_closure import RunRecord, load_jsonl
from .evidence_integrity import RunProvenance, canonical_sha256
from .experiment_protocol import ExperimentProtocol, build_protocol_receipt
from .milestone_analysis import (
    MilestonePromotionConfig,
    build_milestone_scorecard,
    evaluate_milestone_promotion,
)
from .meta_exploration import (
    DeclarativeExplorationPolicy,
    DeclarativePolicyMutator,
    DiscoveryNode,
    DiscoveryTree,
    EvidenceClass,
    EvolvingExplorationController,
    ExplorationPolicyPromotionGate,
    MetaPolicyOptimizer,
    OnlinePolicyResult,
    PolicyProposal,
    PromotionReceipt,
    ReplayObjective,
    ReplaySimulatorPool,
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _run_id(record: RunRecord) -> str:
    return "|".join(map(str, (record.system, record.seed, record.task, record.split, record.transitions)))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class MetricProjection:
    """Explicit projection from empirical RunRecord metrics into replay-tree metrics.

    Missing optional metrics map to zero rather than being invented. Wall-clock latency is
    scaled to hours by default so the replay objective is not numerically dominated by seconds.
    """

    quality_metric: str = "success_rate"
    novelty_metric: str = "novelty"
    uncertainty_reduction_metric: str = "uncertainty_reduction"
    transfer_metric: str = "success_rate"
    information_gain_metric: str = "information_gain"
    constraint_metric: str = "constraint_violations"
    compute_cost_metric: str = "normalized_compute_cost"
    latency_metric: str = "wall_clock_seconds"
    latency_scale: float = 1.0 / 3600.0

    def __post_init__(self):
        if not math.isfinite(float(self.latency_scale)) or self.latency_scale < 0:
            raise ValueError("latency_scale must be finite and >= 0")

    def project(self, record: RunRecord) -> tuple[dict[str, float], float, float]:
        m = record.metrics

        def get(name: str, default: float = 0.0) -> float:
            value = float(m.get(name, default))
            if not math.isfinite(value):
                raise ValueError(f"non-finite projected metric: {name}")
            return value

        quality = get(self.quality_metric)
        transfer = get(self.transfer_metric) if record.split == "transfer" else 0.0
        metrics = {
            "quality": quality,
            "novelty": get(self.novelty_metric),
            "uncertainty_reduction": get(self.uncertainty_reduction_metric),
            "transfer": transfer,
            "information_gain": get(self.information_gain_metric),
            "constraint_violations": get(self.constraint_metric),
        }
        compute = get(self.compute_cost_metric)
        latency = get(self.latency_metric) * float(self.latency_scale)
        if compute < 0 or latency < 0:
            raise ValueError("projected compute/latency costs must be >= 0")
        return metrics, compute, latency


@dataclass(frozen=True)
class CampaignCell:
    policy_id: str
    seed: int
    task: str
    split: str
    transitions: int

    def __post_init__(self):
        if not self.policy_id or not self.task:
            raise ValueError("policy_id and task are required")
        if self.split not in {"train", "validation", "heldout", "transfer"}:
            raise ValueError("invalid split")
        if self.transitions < 0:
            raise ValueError("transitions must be non-negative")

    @property
    def key(self) -> tuple[str, int, str, str, int]:
        return (self.policy_id, int(self.seed), self.task, self.split, int(self.transitions))

    @property
    def cell_id(self) -> str:
        return _sha(self.key)[:24]

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "cell_id": self.cell_id}


@dataclass(frozen=True)
class PairedCampaignPlan:
    active_policy_id: str
    candidate_policy_id: str
    seeds: tuple[int, ...]
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    milestones: tuple[int, ...]
    primary_metric: str = "success_rate"
    minimum_seeds: int = 5

    def __post_init__(self):
        if not self.active_policy_id or not self.candidate_policy_id:
            raise ValueError("active/candidate policy IDs are required")
        if self.active_policy_id == self.candidate_policy_id:
            raise ValueError("candidate policy must differ from active policy")
        for name in ("seeds", "tasks", "splits", "milestones"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)):
                raise ValueError(f"{name} must be non-empty and unique")
        if len(self.seeds) < self.minimum_seeds:
            raise ValueError("validation plan has fewer seeds than minimum_seeds")
        if self.minimum_seeds < 2:
            raise ValueError("minimum_seeds must be >= 2")

    @property
    def systems(self) -> tuple[str, str]:
        return (self.active_policy_id, self.candidate_policy_id)

    @property
    def cells(self) -> tuple[CampaignCell, ...]:
        return tuple(
            CampaignCell(system, seed, task, split, milestone)
            for system in self.systems
            for seed in self.seeds
            for task in self.tasks
            for split in self.splits
            for milestone in self.milestones
        )

    @property
    def protocol(self) -> ExperimentProtocol:
        return ExperimentProtocol(
            protocol_id=f"evolving-{_sha(self.to_dict(include_cells=False))[:16]}",
            systems=self.systems,
            seeds=self.seeds,
            tasks=self.tasks,
            splits=self.splits,
            milestones=self.milestones,
            primary_metric=self.primary_metric,
            minimum_seeds=self.minimum_seeds,
        )

    def to_dict(self, *, include_cells: bool = True) -> dict[str, Any]:
        out = {
            "format": "awa-v2.21-paired-campaign-plan-v1",
            "active_policy_id": self.active_policy_id,
            "candidate_policy_id": self.candidate_policy_id,
            "seeds": list(self.seeds),
            "tasks": list(self.tasks),
            "splits": list(self.splits),
            "milestones": list(self.milestones),
            "primary_metric": self.primary_metric,
            "minimum_seeds": self.minimum_seeds,
        }
        if include_cells:
            out["cells"] = [c.to_dict() for c in self.cells]
        return out


@dataclass(frozen=True)
class CampaignEvidence:
    """One real campaign cell plus the v2.18 provenance required to trust it."""

    record: RunRecord
    provenance: RunProvenance
    evidence: EvidenceClass = EvidenceClass.VALIDATED

    def __post_init__(self):
        if not self.evidence.grounded:
            raise ValueError("campaign qualification evidence must be OBSERVED or VALIDATED")
        if self.provenance.run_id != _run_id(self.record):
            raise ValueError("provenance run_id does not match RunRecord key")

    @property
    def cell(self) -> CampaignCell:
        r = self.record
        return CampaignCell(r.system, r.seed, r.task, r.split, r.transitions)

    @property
    def provenance_sha256(self) -> str:
        return canonical_sha256(self.provenance.to_dict())

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "awa-v2.21-campaign-evidence-v1",
            "evidence": self.evidence.value,
            "record": self.record.to_dict(),
            "provenance": self.provenance.to_dict(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CampaignEvidence":
        if raw.get("format") != "awa-v2.21-campaign-evidence-v1":
            raise ValueError("unsupported campaign evidence format")
        p = dict(raw["provenance"])
        p["scenario_ids"] = tuple(p["scenario_ids"])
        return cls(
            record=RunRecord(**raw["record"]),
            provenance=RunProvenance(**p),
            evidence=EvidenceClass(raw.get("evidence", "validated")),
        )


class CampaignEvidenceStore:
    """Content-addressed, append-only evidence store with conflict rejection."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, cell: CampaignCell) -> Path:
        return self.root / f"{cell.cell_id}.json"

    def put(self, evidence: CampaignEvidence) -> Path:
        path = self.path_for(evidence.cell)
        payload = evidence.to_dict()
        if path.exists():
            existing = CampaignEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
            if existing.sha256 != evidence.sha256:
                raise ValueError(f"evidence conflict for cell {evidence.cell.cell_id}")
            return path
        _atomic_json(path, payload)
        return path

    def get(self, cell: CampaignCell) -> CampaignEvidence | None:
        path = self.path_for(cell)
        if not path.exists():
            return None
        out = CampaignEvidence.from_dict(json.loads(path.read_text(encoding="utf-8")))
        if out.cell.key != cell.key:
            raise ValueError(f"evidence cell-key mismatch in {path}")
        return out

    def all(self) -> list[CampaignEvidence]:
        rows = []
        for path in sorted(self.root.glob("*.json")):
            rows.append(CampaignEvidence.from_dict(json.loads(path.read_text(encoding="utf-8"))))
        return rows


class CampaignRunner(Protocol):
    def run(
        self,
        cell: CampaignCell,
        policy: DeclarativeExplorationPolicy,
        work_dir: Path,
    ) -> CampaignEvidence: ...


class SubprocessCampaignRunner:
    """Run an explicit argv template without a shell and ingest one evidence JSON.

    Supported placeholders: {policy_json}, {policy_id}, {seed}, {task}, {split},
    {transitions}, {output}, {work_dir}. The child must write a
    ``awa-v2.21-campaign-evidence-v1`` JSON document to {output}.
    """

    PLACEHOLDERS = {
        "policy_json", "policy_id", "seed", "task", "split", "transitions", "output", "work_dir"
    }

    def __init__(self, argv_template: Iterable[str], *, timeout_seconds: float | None = None, env: dict[str, str] | None = None):
        self.argv_template = tuple(str(x) for x in argv_template)
        if not self.argv_template:
            raise ValueError("runner argv_template cannot be empty")
        if timeout_seconds is not None and (timeout_seconds <= 0 or not math.isfinite(float(timeout_seconds))):
            raise ValueError("timeout_seconds must be finite and > 0")
        self.timeout_seconds = None if timeout_seconds is None else float(timeout_seconds)
        self.env = dict(env or {})

    def _argv(self, cell: CampaignCell, policy_path: Path, output: Path, work_dir: Path) -> list[str]:
        values = {
            "policy_json": str(policy_path),
            "policy_id": cell.policy_id,
            "seed": str(cell.seed),
            "task": cell.task,
            "split": cell.split,
            "transitions": str(cell.transitions),
            "output": str(output),
            "work_dir": str(work_dir),
        }
        return [token.format(**values) for token in self.argv_template]

    def run(self, cell: CampaignCell, policy: DeclarativeExplorationPolicy, work_dir: Path) -> CampaignEvidence:
        work_dir.mkdir(parents=True, exist_ok=True)
        policy_path = work_dir / "policy.json"
        output = work_dir / "evidence.json"
        _atomic_json(policy_path, {"format": "awa-v2.20-exploration-policy-v1", **policy.to_dict()})
        argv = self._argv(cell, policy_path, output, work_dir)
        env = os.environ.copy()
        env.update(self.env)
        proc = subprocess.run(
            argv,
            cwd=str(work_dir),
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
        )
        log = {
            "argv": argv,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
        _atomic_json(work_dir / "runner_receipt.json", log)
        if proc.returncode != 0:
            raise RuntimeError(f"campaign runner failed for {cell.cell_id}: exit {proc.returncode}")
        if not output.exists():
            raise RuntimeError(f"campaign runner did not produce {output}")
        evidence = CampaignEvidence.from_dict(json.loads(output.read_text(encoding="utf-8")))
        if evidence.cell.key != cell.key:
            raise ValueError("campaign runner produced evidence for the wrong cell")
        return evidence


class DiscoveryTreeBuilder:
    """Convert completed empirical campaign cells into deterministic replay worlds.

    One replay world is built per ``(seed, split)``. Each ``(system, task)`` is a
    root branch and increasing transition milestones extend that branch. This
    preserves the meta-exploration choice between historical experiment branches
    instead of flattening every run into an isolated linear world.
    """

    def __init__(self, projection: MetricProjection | None = None):
        self.projection = projection or MetricProjection()

    def build(self, evidence: Iterable[CampaignEvidence]) -> list[DiscoveryTree]:
        rows = list(evidence)
        worlds: dict[tuple[int, str], dict[tuple[str, str], list[CampaignEvidence]]] = {}
        for row in rows:
            r = row.record
            worlds.setdefault((r.seed, r.split), {}).setdefault((r.system, r.task), []).append(row)
        trees: list[DiscoveryTree] = []
        for world_key, branches in sorted(worlds.items()):
            seed, split = world_key
            tree = DiscoveryTree(f"{seed}|{split}")
            for branch_index, (branch_key, group) in enumerate(sorted(branches.items())):
                policy_id, task = branch_key
                group.sort(key=lambda e: (e.record.transitions, e.sha256))
                transitions = [e.record.transitions for e in group]
                if len(transitions) != len(set(transitions)):
                    raise ValueError(f"duplicate transition milestone in history branch: {world_key}/{branch_key}")
                parent = DiscoveryTree.ROOT_ID
                for idx, e in enumerate(group):
                    metrics, compute, latency = self.projection.project(e.record)
                    node_id = f"{policy_id}-{task}-m{e.record.transitions}-{e.cell.cell_id}"
                    tree.add(DiscoveryNode(
                        node_id=node_id,
                        parent_id=parent,
                        iteration=idx + 1,
                        generation_index=branch_index if parent == DiscoveryTree.ROOT_ID else 0,
                        evidence=e.evidence,
                        metrics=metrics,
                        compute_cost=compute,
                        latency_seconds=latency,
                        artifact_ref=e.record.checkpoint_sha256 or None,
                        provenance_sha256=e.provenance_sha256,
                        metadata={
                            "system": policy_id,
                            "seed": seed,
                            "task": task,
                            "split": split,
                            "transitions": e.record.transitions,
                        },
                    ))
                    parent = node_id
            trees.append(tree)
        if not trees:
            raise ValueError("completed campaign evidence is required to build replay worlds")
        return trees


@dataclass(frozen=True)
class ValidationMatrix:
    seeds: tuple[int, ...]
    tasks: tuple[str, ...]
    splits: tuple[str, ...]
    milestones: tuple[int, ...]
    primary_metric: str = "success_rate"
    minimum_seeds: int = 5

    def plan(self, proposal: PolicyProposal) -> PairedCampaignPlan:
        return PairedCampaignPlan(
            active_policy_id=proposal.active_policy_id,
            candidate_policy_id=proposal.candidate.policy_id,
            seeds=self.seeds,
            tasks=self.tasks,
            splits=self.splits,
            milestones=self.milestones,
            primary_metric=self.primary_metric,
            minimum_seeds=self.minimum_seeds,
        )

    def bootstrap_cells(self, policy_id: str) -> tuple[CampaignCell, ...]:
        return tuple(
            CampaignCell(policy_id, seed, task, split, milestone)
            for seed in self.seeds
            for task in self.tasks
            for split in self.splits
            for milestone in self.milestones
        )


@dataclass(frozen=True)
class ClosedLoopReceipt:
    status: str
    protocol_status: str
    promotion_status: str
    active_policy_before: str
    active_policy_after: str
    candidate_policy_id: str
    evidence_cells: int
    protocol_receipt_sha256: str
    failures: tuple[str, ...]
    promotion: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EvolvingCampaignOrchestrator:
    """Operational v2.21 loop: history -> dream -> paired real runs -> qualify -> promote.

    This class never rewrites source code. Replay can only propose a bounded declarative policy.
    Promotion requires exact v2.19 protocol closure over grounded online evidence.
    """

    STATE_FORMAT = "awa-v2.24-evolving-campaign-state-v1"

    def __init__(
        self,
        root: str | Path,
        *,
        active_policy: DeclarativeExplorationPolicy,
        validation: ValidationMatrix,
        objective: ReplayObjective | None = None,
        projection: MetricProjection | None = None,
        mutation_step: float = 0.10,
        mutation_fields: tuple[str, ...] | None = None,
        minimum_replay_gain: float = 0.0,
        minimum_new_world_fraction: float = 0.10,
        minimum_adversarial_fraction: float = 0.05,
        maximum_seed_regression: float = 0.05,
        minimum_final_mean_gain: float = 0.0,
        minimum_curve_mean_gain: float = 0.0,
        maximum_split_mean_regression: float = 0.02,
        require_positive_final_ci: bool = False,
        promotion_bootstrap_resamples: int = 4000,
        candidate_equivalence_key: Callable[[DeclarativeExplorationPolicy], Hashable] | None = None,
        candidate_equivalence_id: str = "policy-sha256",
    ):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "evolving_campaign_state.json"
        self.evidence_store = CampaignEvidenceStore(self.root / "evidence")
        self.validation = validation
        self.objective = objective or ReplayObjective()
        self.projection = projection or MetricProjection()
        self.mutation_step = float(mutation_step)
        self.mutation_fields = tuple(mutation_fields) if mutation_fields is not None else None
        self.minimum_replay_gain = float(minimum_replay_gain)
        self.minimum_new_world_fraction = float(minimum_new_world_fraction)
        self.minimum_adversarial_fraction = float(minimum_adversarial_fraction)
        self.maximum_seed_regression = float(maximum_seed_regression)
        self.minimum_final_mean_gain = float(minimum_final_mean_gain)
        self.minimum_curve_mean_gain = float(minimum_curve_mean_gain)
        self.maximum_split_mean_regression = float(maximum_split_mean_regression)
        self.require_positive_final_ci = bool(require_positive_final_ci)
        self.promotion_bootstrap_resamples = int(promotion_bootstrap_resamples)
        self.candidate_equivalence_key = candidate_equivalence_key
        self.candidate_equivalence_id = str(candidate_equivalence_id)
        self.milestone_promotion = MilestonePromotionConfig(
            minimum_final_mean_gain=self.minimum_final_mean_gain,
            minimum_curve_mean_gain=self.minimum_curve_mean_gain,
            maximum_seed_final_regression=self.maximum_seed_regression,
            maximum_split_final_regression=self.maximum_split_mean_regression,
            require_positive_final_ci=self.require_positive_final_ci,
            bootstrap_resamples=self.promotion_bootstrap_resamples,
        )
        config_fingerprint = _sha({
            "initial_active_policy": active_policy.to_dict(),
            "validation": asdict(validation),
            "objective": asdict(self.objective),
            "projection": asdict(self.projection),
            "mutation_step": self.mutation_step,
            "mutation_fields": list(self.mutation_fields) if self.mutation_fields is not None else None,
            "minimum_replay_gain": self.minimum_replay_gain,
            "minimum_new_world_fraction": self.minimum_new_world_fraction,
            "minimum_adversarial_fraction": self.minimum_adversarial_fraction,
            "maximum_seed_regression": self.maximum_seed_regression,
            "minimum_final_mean_gain": self.minimum_final_mean_gain,
            "minimum_curve_mean_gain": self.minimum_curve_mean_gain,
            "maximum_split_mean_regression": self.maximum_split_mean_regression,
            "require_positive_final_ci": self.require_positive_final_ci,
            "promotion_bootstrap_resamples": self.promotion_bootstrap_resamples,
            "candidate_equivalence_id": self.candidate_equivalence_id,
        })
        if self.state_path.exists():
            raw = json.loads(self.state_path.read_text(encoding="utf-8"))
            if raw.get("format") != self.STATE_FORMAT:
                raise ValueError("unsupported evolving campaign state format")
            if raw.get("config_sha256") != config_fingerprint:
                raise ValueError("evolving campaign configuration changed; use a new output directory")
            self.active_policy = DeclarativeExplorationPolicy(**raw["active_policy"])
            self.iteration = int(raw.get("iteration", 0))
            self.history = list(raw.get("history", []))
        else:
            self.active_policy = active_policy
            self.iteration = 0
            self.history: list[dict[str, Any]] = []
            self._save_state(config_fingerprint)
        self.config_sha256 = config_fingerprint

    def _save_state(self, config_sha256: str | None = None) -> None:
        _atomic_json(self.state_path, {
            "format": self.STATE_FORMAT,
            "config_sha256": config_sha256 or self.config_sha256,
            "iteration": self.iteration,
            "active_policy": self.active_policy.to_dict(),
            "history": self.history,
        })

    def import_history(self, evidence: Iterable[CampaignEvidence]) -> int:
        count = 0
        for row in evidence:
            path = self.evidence_store.put(row)
            if path.exists():
                count += 1
        return count

    def bootstrap_plan(self) -> dict[str, Any]:
        cells = self.validation.bootstrap_cells(self.active_policy.policy_id)
        return {
            "format": "awa-v2.22-bootstrap-plan-v1",
            "policy_id": self.active_policy.policy_id,
            "cells": [c.to_dict() for c in cells],
            "cell_count": len(cells),
        }

    def bootstrap_active(self, runner: CampaignRunner, *, rerun_existing: bool = False) -> dict[str, Any]:
        """Collect grounded history for the current active policy without promotion.

        This is an explicit execution step. It only populates replay history and never
        changes the active policy. Existing identical cells are reused unless asked to
        rerun them.
        """
        cells = self.validation.bootstrap_cells(self.active_policy.policy_id)
        created = 0
        reused = 0
        root = self.root / "bootstrap" / self.active_policy.policy_id
        for cell in cells:
            existing = None if rerun_existing else self.evidence_store.get(cell)
            if existing is not None:
                reused += 1
                continue
            row = runner.run(cell, self.active_policy, root / cell.cell_id)
            if row.cell.key != cell.key:
                raise ValueError("bootstrap runner returned evidence for a different planned cell")
            self.evidence_store.put(row)
            created += 1
        receipt = {
            "format": "awa-v2.22-bootstrap-receipt-v1",
            "status": "PASS",
            "policy_id": self.active_policy.policy_id,
            "cells": len(cells),
            "created": created,
            "reused": reused,
        }
        _atomic_json(self.root / "bootstrap_receipt.json", receipt)
        return receipt

    def _controller(self, evidence: Iterable[CampaignEvidence]) -> EvolvingExplorationController:
        trees = DiscoveryTreeBuilder(self.projection).build(evidence)
        pool = ReplaySimulatorPool()
        for tree in trees:
            pool.add_tree(tree, self.objective)
        mutator = DeclarativePolicyMutator(
            self.mutation_step,
            min_new_world_fraction=self.minimum_new_world_fraction,
            min_adversarial_fraction=self.minimum_adversarial_fraction,
            fields=self.mutation_fields,
        )
        optimizer = MetaPolicyOptimizer(pool, mutator, min_replay_gain=self.minimum_replay_gain)
        optimizer.candidate_equivalence_key = self.candidate_equivalence_key
        gate = ExplorationPolicyPromotionGate(
            minimum_seeds=self.validation.minimum_seeds,
            minimum_mean_gain=0.0,
            max_seed_regression=self.maximum_seed_regression,
        )
        return EvolvingExplorationController(self.active_policy, pool, optimizer, gate)

    def plan_iteration(self, *, max_rounds: int = 64) -> dict[str, Any]:
        evidence = self.evidence_store.all()
        iteration_dir = self.root / "iterations" / f"{self.iteration:04d}"
        iteration_dir.mkdir(parents=True, exist_ok=True)
        if not evidence:
            report = {
                "format": "awa-v2.22-evolving-campaign-iteration-v1",
                "status": "NEEDS_BOOTSTRAP",
                "iteration": self.iteration,
                "evidence_history_cells": 0,
                "bootstrap_plan": self.bootstrap_plan(),
            }
            _atomic_json(iteration_dir / "iteration_plan.json", report)
            return report
        controller = self._controller(evidence)
        proposal = controller.dream(max_rounds=max_rounds)
        proposal_dict = {
            "active_policy_id": proposal.active_policy_id,
            "candidate": proposal.candidate.to_dict(),
            "replay_score": proposal.replay_score,
            "active_replay_score": proposal.active_replay_score,
            "replay_gain": proposal.replay_gain,
            "status": proposal.status,
            "candidates_considered": proposal.candidates_considered,
            "unique_execution_candidates": proposal.unique_execution_candidates,
        }
        _atomic_json(iteration_dir / "proposal.json", proposal_dict)
        if proposal.candidate.policy_id == proposal.active_policy_id:
            report = {
                "format": "awa-v2.22-evolving-campaign-iteration-v1",
                "status": "NO_CHANGE",
                "iteration": self.iteration,
                "proposal": proposal_dict,
                "evidence_history_cells": len(evidence),
            }
            _atomic_json(iteration_dir / "iteration_plan.json", report)
            return report
        plan = self.validation.plan(proposal)
        report = {
            "format": "awa-v2.22-evolving-campaign-iteration-v1",
            "status": "PLANNED",
            "iteration": self.iteration,
            "proposal": proposal_dict,
            "validation_plan": plan.to_dict(),
            "evidence_history_cells": len(evidence),
        }
        _atomic_json(iteration_dir / "iteration_plan.json", report)
        return report

    @staticmethod
    def _proposal_from_plan(plan_report: dict[str, Any]) -> PolicyProposal:
        p = plan_report["proposal"]
        return PolicyProposal(
            active_policy_id=p["active_policy_id"],
            candidate=DeclarativeExplorationPolicy(**p["candidate"]),
            replay_score=float(p["replay_score"]),
            active_replay_score=float(p["active_replay_score"]),
            replay_gain=float(p["replay_gain"]),
            status=str(p.get("status", "PROPOSED")),
            candidates_considered=int(p.get("candidates_considered", 0)),
            unique_execution_candidates=int(p.get("unique_execution_candidates", 0)),
        )

    @staticmethod
    def _plan_from_report(plan_report: dict[str, Any]) -> PairedCampaignPlan:
        p = plan_report["validation_plan"]
        return PairedCampaignPlan(
            active_policy_id=p["active_policy_id"],
            candidate_policy_id=p["candidate_policy_id"],
            seeds=tuple(int(x) for x in p["seeds"]),
            tasks=tuple(str(x) for x in p["tasks"]),
            splits=tuple(str(x) for x in p["splits"]),
            milestones=tuple(int(x) for x in p["milestones"]),
            primary_metric=str(p["primary_metric"]),
            minimum_seeds=int(p["minimum_seeds"]),
        )

    def execute_planned_iteration(
        self,
        runner: CampaignRunner,
        *,
        rerun_existing: bool = False,
    ) -> ClosedLoopReceipt:
        iteration_dir = self.root / "iterations" / f"{self.iteration:04d}"
        plan_path = iteration_dir / "iteration_plan.json"
        if not plan_path.exists():
            raise RuntimeError("plan_iteration() must run before execution")
        plan_report = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan_report.get("status") == "NO_CHANGE":
            return ClosedLoopReceipt(
                status="NO_CHANGE", protocol_status="NOT_RUN", promotion_status="NOT_RUN",
                active_policy_before=self.active_policy.policy_id, active_policy_after=self.active_policy.policy_id,
                candidate_policy_id=self.active_policy.policy_id, evidence_cells=0,
                protocol_receipt_sha256="", failures=(), promotion=None,
            )
        if plan_report.get("status") != "PLANNED":
            raise RuntimeError("iteration is not in PLANNED state")
        proposal = self._proposal_from_plan(plan_report)
        plan = self._plan_from_report(plan_report)
        policies = {
            proposal.active_policy_id: self.active_policy,
            proposal.candidate.policy_id: proposal.candidate,
        }
        online: list[CampaignEvidence] = []
        for cell in plan.cells:
            existing = None if rerun_existing else self.evidence_store.get(cell)
            if existing is not None:
                online.append(existing)
                continue
            work_dir = iteration_dir / "runs" / cell.cell_id
            row = runner.run(cell, policies[cell.policy_id], work_dir)
            if row.cell.key != cell.key:
                raise ValueError("runner returned evidence for a different planned cell")
            self.evidence_store.put(row)
            online.append(row)

        records = [x.record for x in online]
        provenance = {_run_id(x.record): x.provenance for x in online}
        protocol_receipt = build_protocol_receipt(plan.protocol, records, provenance, baseline=plan.active_policy_id)
        _atomic_json(iteration_dir / "protocol_receipt.json", protocol_receipt)
        failures: list[str] = []
        if protocol_receipt["status"] != "PASS":
            failures.extend(protocol_receipt["completeness"].get("failures", []))
            failures.extend(protocol_receipt["pairing"].get("failures", []))
            failures.extend(protocol_receipt["metric_semantics"].get("failures", []))
            failures.extend(protocol_receipt["integrity"].get("integrity", {}).get("failures", []))
            failures.extend(protocol_receipt["integrity"].get("split_leakage", {}).get("failures", []))
            receipt = ClosedLoopReceipt(
                status="FAIL",
                protocol_status="FAIL",
                promotion_status="NOT_RUN",
                active_policy_before=self.active_policy.policy_id,
                active_policy_after=self.active_policy.policy_id,
                candidate_policy_id=proposal.candidate.policy_id,
                evidence_cells=len(online),
                protocol_receipt_sha256=protocol_receipt["receipt_sha256"],
                failures=tuple(failures),
                promotion=None,
            )
            _atomic_json(iteration_dir / "closed_loop_receipt.json", receipt.to_dict())
            return receipt

        scorecard = build_milestone_scorecard(
            records,
            baseline=plan.active_policy_id,
            candidate=plan.candidate_policy_id,
            primary_metric=plan.primary_metric,
            minimum_seeds=plan.minimum_seeds,
            confidence=self.milestone_promotion.confidence,
            resamples=self.milestone_promotion.bootstrap_resamples,
        )
        _atomic_json(iteration_dir / "milestone_scorecard.json", scorecard)
        promotion = evaluate_milestone_promotion(scorecard, self.milestone_promotion)
        before = self.active_policy.policy_id
        if promotion.status == "PASS":
            self.active_policy = proposal.candidate
        after = self.active_policy.policy_id
        receipt = ClosedLoopReceipt(
            status="PASS" if promotion.status == "PASS" else "RETAIN",
            protocol_status="PASS",
            promotion_status=promotion.status,
            active_policy_before=before,
            active_policy_after=after,
            candidate_policy_id=proposal.candidate.policy_id,
            evidence_cells=len(online),
            protocol_receipt_sha256=protocol_receipt["receipt_sha256"],
            failures=promotion.failures,
            promotion=asdict(promotion),
        )
        _atomic_json(iteration_dir / "closed_loop_receipt.json", receipt.to_dict())
        self.history.append({
            "iteration": self.iteration,
            "proposal": plan_report["proposal"],
            "receipt": receipt.to_dict(),
        })
        self.iteration += 1
        self._save_state()
        return receipt


def load_history_evidence(records_path: str | Path, provenance_path: str | Path) -> list[CampaignEvidence]:
    records = load_jsonl(records_path)
    raw = json.loads(Path(provenance_path).read_text(encoding="utf-8"))
    if isinstance(raw, dict) and "provenance" in raw:
        raw = raw["provenance"]
    if not isinstance(raw, dict):
        raise ValueError("provenance file must be a mapping keyed by run_id")
    out = []
    for record in records:
        rid = _run_id(record)
        p = raw.get(rid)
        if p is None:
            raise ValueError(f"missing provenance for history record {rid}")
        row = dict(p)
        row["scenario_ids"] = tuple(row["scenario_ids"])
        out.append(CampaignEvidence(record, RunProvenance(**row), EvidenceClass.OBSERVED))
    return out


def runner_from_config(config: dict[str, Any]) -> CampaignRunner:
    runner = config.get("runner") or {}
    runner_type = str(runner.get("type", "subprocess"))
    if runner_type == "native_procedural":
        # Lazy import avoids a module cycle: native_campaign implements the
        # CampaignRunner protocol defined in this module.
        from .native_campaign import native_runner_from_config
        return native_runner_from_config(config)
    if runner_type != "subprocess":
        raise ValueError("runner.type must be subprocess or native_procedural")
    argv = runner.get("argv")
    if not isinstance(argv, list) or not argv:
        raise ValueError("runner.argv must be a non-empty list")
    env = runner.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("runner.env must be a mapping")
    return SubprocessCampaignRunner(argv, timeout_seconds=runner.get("timeout_seconds"), env={str(k): str(v) for k, v in env.items()})


def orchestrator_from_config(config: dict[str, Any], out_dir: str | Path) -> EvolvingCampaignOrchestrator:
    meta = config.get("meta_exploration") or {}
    replay = meta.get("replay") or {}
    objective = ReplayObjective(**(replay.get("objective") or {}))
    active_raw = dict(config.get("active_policy") or meta.get("policy") or {})
    active_raw.setdefault("policy_id", "active")
    active = DeclarativeExplorationPolicy(**active_raw)
    mutation = meta.get("mutation") or {}
    promotion = meta.get("promotion") or {}
    validation_raw = config.get("validation") or {}
    validation = ValidationMatrix(
        seeds=tuple(int(x) for x in validation_raw.get("seeds", (1701, 1702, 1703, 1704, 1705))),
        tasks=tuple(str(x) for x in validation_raw.get("tasks", ("vizdoom",))),
        splits=tuple(str(x) for x in validation_raw.get("splits", ("heldout", "transfer"))),
        milestones=tuple(int(x) for x in validation_raw.get("milestones", (25_000,))),
        primary_metric=str(validation_raw.get("primary_metric", "success_rate")),
        minimum_seeds=int(validation_raw.get("minimum_seeds", 5)),
    )
    projection = MetricProjection(**(config.get("metric_projection") or {}))
    candidate_equivalence_key = None
    candidate_equivalence_id = "policy-sha256"
    mutation_fields = tuple(str(x) for x in mutation.get("fields", ())) or None
    runner_cfg = config.get("runner") or {}
    if str(runner_cfg.get("type", "subprocess")) == "native_procedural":
        # Native execution collapses many continuous meta-policy perturbations into
        # the same integer stage allocation.  Deduplicate on the behavior that will
        # actually reach the training collector, not on policy JSON identity.
        from .native_campaign import ExplorationPolicyCurriculumAdapter, NativeProceduralRunnerConfig

        native_cfg = NativeProceduralRunnerConfig.from_dict(runner_cfg.get("config") or {})
        adapter = ExplorationPolicyCurriculumAdapter()
        stages = tuple(native_cfg.curriculum_stages)
        slots = int(native_cfg.tasks_per_batch)
        candidate_equivalence_key = lambda p: adapter.execution_signature(p, stages, slots)
        candidate_equivalence_id = canonical_sha256({
            "kind": "native-curriculum-stage-counts",
            "stages": list(stages),
            "slots": slots,
        })
        if mutation_fields is None:
            # Only mutate dimensions that change the native training allocation.
            # Replay traversal knobs remain fixed rather than pretending to be
            # curriculum improvements.
            mutation_fields = (
                "quality_weight",
                "information_weight",
                "novelty_weight",
                "transfer_weight",
            )
    return EvolvingCampaignOrchestrator(
        out_dir,
        active_policy=active,
        validation=validation,
        objective=objective,
        projection=projection,
        mutation_step=float(mutation.get("step", 0.10)),
        mutation_fields=mutation_fields,
        minimum_replay_gain=float(mutation.get("minimum_replay_gain", 0.0)),
        minimum_new_world_fraction=float(mutation.get("minimum_new_world_fraction", 0.10)),
        minimum_adversarial_fraction=float(mutation.get("minimum_adversarial_fraction", 0.05)),
        maximum_seed_regression=float(promotion.get("maximum_per_seed_regression", 0.05)),
        minimum_final_mean_gain=float(promotion.get("minimum_final_mean_gain", 0.0)),
        minimum_curve_mean_gain=float(promotion.get("minimum_curve_mean_gain", 0.0)),
        maximum_split_mean_regression=float(promotion.get("maximum_split_mean_regression", 0.02)),
        require_positive_final_ci=bool(promotion.get("require_positive_final_ci", False)),
        promotion_bootstrap_resamples=int(promotion.get("bootstrap_resamples", 4000)),
        candidate_equivalence_key=candidate_equivalence_key,
        candidate_equivalence_id=candidate_equivalence_id,
    )
