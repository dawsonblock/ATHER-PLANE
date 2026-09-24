"""Grounded DREAM-RSI meta-learning for Aether's training allocation.

This module deliberately optimizes *how Aether learns*, never action-time inference.  Replay
reveals only recorded grounded outcomes.  A replay winner remains a hypothesis until paired
real-online qualification passes under an unchanged agent/evaluator/interface contract.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib
import json
import math
import statistics
from typing import Iterable, Sequence

from awa.v2.learning_system.meta_allocation import (
    DeclarativeTrainingAllocationPolicy,
    DeclarativeTrainingPolicyMutator,
    TrainingAllocationDecision,
    TrainingAllocationOption,
)
from awa.v2.learning_system.dream_execution import ExactAllocationExecutionReceipt
from awa.v2.meta_exploration import EvidenceClass


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _valid_sha256(value: str) -> bool:
    value = value.lower()
    return len(value) == 64 and all(ch in "0123456789abcdef" for ch in value)


@dataclass(frozen=True)
class DreamRSIContract:
    """Freezes the underlying agent, evaluator, and decision interface.

    DREAM-RSI's controlled comparison changes the exploration/meta-policy while keeping these
    three surfaces fixed.  Aether binds that requirement cryptographically.
    """

    agent_sha256: str
    evaluator_sha256: str
    interface_version: str = "awa-training-allocation-v2"

    def __post_init__(self) -> None:
        if not _valid_sha256(self.agent_sha256) or not _valid_sha256(self.evaluator_sha256):
            raise ValueError("agent/evaluator fingerprints must be SHA-256 hex digests")
        if not self.interface_version:
            raise ValueError("interface_version required")

    @property
    def sha256(self) -> str:
        return _sha(asdict(self))


@dataclass(frozen=True)
class GroundedTrainingOutcome:
    """One recorded online training-allocation outcome.

    ``option`` contains only information available before execution.  ``metrics`` contains the
    realized result and is revealed only after replay selects the recorded branch.
    """

    outcome_id: str
    parent_id: str | None
    iteration: int
    seed: int
    option: TrainingAllocationOption
    metrics: dict[str, float]
    compute_cost: float
    latency_seconds: float
    provenance_sha256: str
    contract_sha256: str
    evidence: EvidenceClass = EvidenceClass.VALIDATED
    offered_options: tuple[TrainingAllocationOption, ...] = ()

    def __post_init__(self) -> None:
        if not self.outcome_id:
            raise ValueError("outcome_id required")
        if self.iteration < 0 or self.seed < 0:
            raise ValueError("iteration/seed must be >= 0")
        if not self.evidence.grounded:
            raise ValueError("replay history accepts grounded outcomes only")
        if not _valid_sha256(self.provenance_sha256) or not _valid_sha256(self.contract_sha256):
            raise ValueError("grounded outcome requires valid provenance and contract hashes")
        if not math.isfinite(float(self.compute_cost)) or self.compute_cost < 0:
            raise ValueError("compute_cost must be finite and >= 0")
        if not math.isfinite(float(self.latency_seconds)) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and >= 0")
        if not all(math.isfinite(float(value)) for value in self.metrics.values()):
            raise ValueError("outcome metrics must be finite")
        if self.offered_options:
            keys = [option.decision.execution_key for option in self.offered_options]
            if len(keys) != len(set(keys)):
                raise ValueError("offered decision menu contains duplicate execution keys")
            if self.option.decision.execution_key not in set(keys):
                raise ValueError("executed decision is absent from its recorded offered menu")

    @classmethod
    def from_dict(cls, payload: dict) -> "GroundedTrainingOutcome":
        row = dict(payload)
        row["option"] = TrainingAllocationOption.from_dict(row["option"])
        row["offered_options"] = tuple(
            TrainingAllocationOption.from_dict(item) for item in row.get("offered_options", ())
        )
        row["evidence"] = EvidenceClass(row.get("evidence", EvidenceClass.VALIDATED.value))
        return cls(**row)

    def to_dict(self) -> dict:
        return {
            "outcome_id": self.outcome_id,
            "parent_id": self.parent_id,
            "iteration": self.iteration,
            "seed": self.seed,
            "option": self.option.to_dict(),
            "metrics": {k: float(v) for k, v in sorted(self.metrics.items())},
            "compute_cost": float(self.compute_cost),
            "latency_seconds": float(self.latency_seconds),
            "provenance_sha256": self.provenance_sha256,
            "contract_sha256": self.contract_sha256,
            "evidence": self.evidence.value,
            "offered_options": [option.to_dict() for option in self.offered_options],
        }


@dataclass(frozen=True)
class ReplayObjective:
    """Replay scoring with an exact-paper mode and an Aether extension mode.

    ``mode="paper"`` follows DREAM-RSI's quality - beta1*attempts +
    beta2*attempts/rounds structure.  ``mode="aether"`` keeps the richer
    embodied-learning utility used by Aether.
    """

    mode: str = "aether"
    quality_weight: float = 1.0
    transfer_weight: float = 0.35
    adaptation_weight: float = 0.20
    failure_weight: float = 0.50
    transition_cost_weight: float = 0.0
    compute_cost_weight: float = 0.10
    latency_weight: float = 0.0
    parallelism_bonus_weight: float = 0.05
    paper_beta1: float = 0.01
    paper_beta2: float = 0.05

    def __post_init__(self) -> None:
        if self.mode not in {"aether", "paper"}:
            raise ValueError("replay objective mode must be aether or paper")
        values = [
            self.quality_weight, self.transfer_weight, self.adaptation_weight,
            self.failure_weight, self.transition_cost_weight,
            self.compute_cost_weight, self.latency_weight,
            self.parallelism_bonus_weight, self.paper_beta1, self.paper_beta2,
        ]
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("replay objective weights must be finite")
        if self.paper_beta1 < 0 or self.paper_beta2 < 0:
            raise ValueError("paper beta coefficients must be >= 0")

    @classmethod
    def from_dict(cls, payload: dict) -> "ReplayObjective":
        return cls(**dict(payload))

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SupportPolicy:
    min_worlds: int = 2
    min_selected_nodes: int = 2
    min_replay_coverage: float = 0.10
    min_exact_decision_support: int = 1

    def __post_init__(self) -> None:
        if self.min_worlds < 1 or self.min_selected_nodes < 1 or self.min_exact_decision_support < 1:
            raise ValueError("support count thresholds must be >= 1")
        if not 0 <= self.min_replay_coverage <= 1:
            raise ValueError("min_replay_coverage must lie in [0,1]")


class SupportStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    WEAK_SUPPORT = "WEAK_SUPPORT"
    UNSUPPORTED = "UNSUPPORTED"


@dataclass(frozen=True)
class ReplayTrace:
    world_id: str
    policy_id: str
    selected_outcome_ids: tuple[str, ...]
    selected_execution_keys: tuple[str, ...]
    unsupported_execution_keys: tuple[str, ...]
    rounds: int
    attempted_worlds: int
    score: float
    compute_cost: float
    latency_seconds: float
    replay_coverage: float


@dataclass(frozen=True)
class SupportAwareReplayReport:
    policy_id: str
    mean_score: float
    min_score: float
    world_count: int
    selected_nodes: int
    unique_decisions: int
    minimum_exact_decision_support: int
    mean_replay_coverage: float
    unsupported_decisions: int
    support_status: SupportStatus
    traces: tuple[ReplayTrace, ...]


class GroundedReplayWorld:
    """A completed historical discovery tree used as a zero-execution-cost simulator."""

    ROOT = "root"

    def __init__(
        self,
        world_id: str,
        outcomes: Iterable[GroundedTrainingOutcome],
        *,
        contract: DreamRSIContract,
        objective: ReplayObjective | None = None,
    ) -> None:
        self.world_id = str(world_id)
        self.contract = contract
        self.objective = objective or ReplayObjective()
        rows = list(outcomes)
        if not rows:
            raise ValueError("replay world requires grounded history")
        if any(row.contract_sha256 != contract.sha256 for row in rows):
            raise ValueError("history contract mismatch")
        ids = [row.outcome_id for row in rows]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate outcome_id")
        self._rows = {row.outcome_id: row for row in rows}
        self._children: dict[str, list[str]] = {self.ROOT: []}
        pending = rows[:]
        while pending:
            before = len(pending)
            for row in pending[:]:
                parent = row.parent_id or self.ROOT
                if parent == self.ROOT or parent in self._rows and parent not in {p.outcome_id for p in pending}:
                    self._children.setdefault(parent, []).append(row.outcome_id)
                    self._children.setdefault(row.outcome_id, [])
                    pending.remove(row)
            if len(pending) == before:
                # Validate by simpler parent existence before declaring cycle.
                missing = [r.parent_id for r in pending if r.parent_id and r.parent_id not in self._rows]
                if missing:
                    raise ValueError(f"history contains missing parents: {sorted(set(missing))}")
                raise ValueError("history contains a cycle")
        for children in self._children.values():
            children.sort(key=lambda outcome_id: (self._rows[outcome_id].iteration, outcome_id))

    @property
    def grounded_node_count(self) -> int:
        return len(self._rows)

    @property
    def outcomes(self) -> tuple[GroundedTrainingOutcome, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def realized_trace(self, policy_id: str = "executed-online") -> ReplayTrace:
        """Score the grounded outcomes that actually executed, without replay reselection."""
        selected = list(self.outcomes)
        objective = self.objective
        best_quality = max((row.metrics.get("quality", 0.0) for row in selected), default=0.0)
        best_transfer = max((row.metrics.get("transfer", 0.0) for row in selected), default=0.0)
        best_adaptation = max((row.metrics.get("adaptation", 0.0) for row in selected), default=0.0)
        failures = sum(row.metrics.get("failure_rate", 0.0) for row in selected)
        transitions = sum(row.option.decision.transition_budget for row in selected)
        compute = sum(row.compute_cost for row in selected)
        latency = sum(row.latency_seconds for row in selected)
        attempted_worlds = sum(row.option.decision.parallel_worlds for row in selected)
        rounds = 1 if selected else 0
        parallelism = attempted_worlds / max(1, rounds) if selected else 0.0
        if objective.mode == "paper":
            score = best_quality - objective.paper_beta1 * attempted_worlds + objective.paper_beta2 * parallelism
        else:
            score = (
                objective.quality_weight * best_quality
                + objective.transfer_weight * best_transfer
                + objective.adaptation_weight * best_adaptation
                - objective.failure_weight * failures
                - objective.transition_cost_weight * transitions
                - objective.compute_cost_weight * compute
                - objective.latency_weight * latency
                + objective.parallelism_bonus_weight * parallelism
            )
        return ReplayTrace(
            world_id=self.world_id,
            policy_id=str(policy_id),
            selected_outcome_ids=tuple(row.outcome_id for row in selected),
            selected_execution_keys=tuple(row.option.decision.execution_key for row in selected),
            unsupported_execution_keys=(),
            rounds=rounds,
            attempted_worlds=int(attempted_worlds),
            score=float(score),
            compute_cost=float(compute),
            latency_seconds=float(latency),
            replay_coverage=1.0 if selected else 0.0,
        )

    def to_dict(self) -> dict:
        return {
            "world_id": self.world_id,
            "contract_sha256": self.contract.sha256,
            "objective": self.objective.to_dict(),
            "outcomes": [row.to_dict() for row in self.outcomes],
        }

    @classmethod
    def from_dict(cls, payload: dict, *, contract: DreamRSIContract) -> "GroundedReplayWorld":
        if payload.get("contract_sha256") not in {None, contract.sha256}:
            raise ValueError("serialized replay world contract mismatch")
        objective = ReplayObjective.from_dict(payload.get("objective", {}))
        outcomes = [GroundedTrainingOutcome.from_dict(row) for row in payload.get("outcomes", ())]
        return cls(str(payload["world_id"]), outcomes, contract=contract, objective=objective)

    def _available(self, revealed: set[str]) -> list[GroundedTrainingOutcome]:
        options: list[GroundedTrainingOutcome] = []
        # Any recorded root branch remains a valid alternate exploration direction.
        options.extend(self._rows[c] for c in self._children[self.ROOT] if c not in revealed)
        # A revealed leaf can continue along recorded children.
        for node_id in sorted(revealed):
            hidden = [child for child in self._children.get(node_id, ()) if child not in revealed]
            options.extend(self._rows[child] for child in hidden)
        # Deduplicate if a root branch is also reachable in malformed external history.
        by_id = {row.outcome_id: row for row in options}
        return [by_id[key] for key in sorted(by_id)]

    def _offered_options(self, available: Sequence[GroundedTrainingOutcome]) -> list[TrainingAllocationOption]:
        """Return the full logged menu when present, otherwise legacy grounded options.

        The v2.33 canonical path records the complete decision menu that was visible
        before an online action.  Choosing an offered option without a grounded child
        is explicitly unsupported and triggers an online probe rather than fabricated
        replay feedback.
        """
        by_key: dict[str, TrainingAllocationOption] = {}
        for row in available:
            menu = row.offered_options or (row.option,)
            for option in menu:
                by_key.setdefault(option.decision.execution_key, option)
        return [by_key[key] for key in sorted(by_key)]

    def run(self, policy: DeclarativeTrainingAllocationPolicy, *, max_rounds: int = 64) -> ReplayTrace:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        revealed: set[str] = set()
        selected: list[GroundedTrainingOutcome] = []
        unsupported: list[str] = []
        rounds = 0
        while rounds < max_rounds:
            available = self._available(revealed)
            if not available:
                break
            offered = self._offered_options(available)
            decisions = policy.choose_batch(offered)
            if len(decisions) == 1 and decisions[0].stop:
                break
            grounded_by_key: dict[str, list[GroundedTrainingOutcome]] = {}
            for row in available:
                grounded_by_key.setdefault(row.option.decision.execution_key, []).append(row)
            requested = [decision for decision in decisions if not decision.stop]
            missing = [
                decision.execution_key
                for decision in requested
                if decision.execution_key not in grounded_by_key
            ]
            if missing:
                unsupported.extend(missing)
                break
            chosen: list[GroundedTrainingOutcome] = []
            for decision in requested:
                matches = grounded_by_key[decision.execution_key]
                chosen.append(matches[0])
            if not chosen:
                break
            for row in chosen:
                revealed.add(row.outcome_id)
                selected.append(row)
            rounds += 1

        objective = self.objective
        best_quality = max((row.metrics.get("quality", 0.0) for row in selected), default=0.0)
        best_transfer = max((row.metrics.get("transfer", 0.0) for row in selected), default=0.0)
        best_adaptation = max((row.metrics.get("adaptation", 0.0) for row in selected), default=0.0)
        failures = sum(row.metrics.get("failure_rate", 0.0) for row in selected)
        transitions = sum(row.option.decision.transition_budget for row in selected)
        compute = sum(row.compute_cost for row in selected)
        latency = sum(row.latency_seconds for row in selected)
        attempted_worlds = sum(row.option.decision.parallel_worlds for row in selected)
        parallelism = attempted_worlds / max(1, rounds) if selected else 0.0
        if objective.mode == "paper":
            score = (
                best_quality
                - objective.paper_beta1 * attempted_worlds
                + objective.paper_beta2 * parallelism
            )
        else:
            score = (
                objective.quality_weight * best_quality
                + objective.transfer_weight * best_transfer
                + objective.adaptation_weight * best_adaptation
                - objective.failure_weight * failures
                - objective.transition_cost_weight * transitions
                - objective.compute_cost_weight * compute
                - objective.latency_weight * latency
                + objective.parallelism_bonus_weight * parallelism
            )
        return ReplayTrace(
            world_id=self.world_id,
            policy_id=policy.policy_id,
            selected_outcome_ids=tuple(row.outcome_id for row in selected),
            selected_execution_keys=tuple(row.option.decision.execution_key for row in selected),
            unsupported_execution_keys=tuple(unsupported),
            rounds=rounds,
            attempted_worlds=int(attempted_worlds),
            score=float(score),
            compute_cost=float(compute),
            latency_seconds=float(latency),
            replay_coverage=float(len(revealed) / max(1, self.grounded_node_count)),
        )



class SupportAwareReplayPool:
    def __init__(self, worlds: Iterable[GroundedReplayWorld], support_policy: SupportPolicy | None = None):
        self.worlds = tuple(worlds)
        if not self.worlds:
            raise ValueError("replay pool cannot be empty")
        contract_hashes = {world.contract.sha256 for world in self.worlds}
        if len(contract_hashes) != 1:
            raise ValueError("all replay worlds must share one frozen contract")
        self.contract_sha256 = next(iter(contract_hashes))
        self.support_policy = support_policy or SupportPolicy()
        counts: dict[str, int] = {}
        for world in self.worlds:
            seen = {row.option.decision.execution_key for row in world._rows.values()}
            for key in seen:
                counts[key] = counts.get(key, 0) + 1
        self._decision_world_support = counts

    def evaluate(self, policy: DeclarativeTrainingAllocationPolicy, *, max_rounds: int = 64) -> SupportAwareReplayReport:
        traces = tuple(world.run(policy, max_rounds=max_rounds) for world in self.worlds)
        scores = [trace.score for trace in traces]
        selected_keys = [key for trace in traces for key in trace.selected_execution_keys]
        unsupported_keys = [key for trace in traces for key in trace.unsupported_execution_keys]
        supports = [self._decision_world_support.get(key, 0) for key in selected_keys]
        minimum_support = min(supports) if supports else 0
        mean_coverage = float(statistics.fmean(trace.replay_coverage for trace in traces))
        policy_gate = self.support_policy
        hard_fail = (
            len(self.worlds) < policy_gate.min_worlds
            or not selected_keys
            or bool(unsupported_keys)
            or minimum_support < policy_gate.min_exact_decision_support
        )
        weak = (
            len(selected_keys) < policy_gate.min_selected_nodes
            or mean_coverage < policy_gate.min_replay_coverage
        )
        status = SupportStatus.UNSUPPORTED if hard_fail else (SupportStatus.WEAK_SUPPORT if weak else SupportStatus.SUPPORTED)
        return SupportAwareReplayReport(
            policy_id=policy.policy_id,
            mean_score=float(statistics.fmean(scores)),
            min_score=float(min(scores)),
            world_count=len(self.worlds),
            selected_nodes=len(selected_keys),
            unique_decisions=len(set(selected_keys)),
            minimum_exact_decision_support=int(minimum_support),
            mean_replay_coverage=mean_coverage,
            unsupported_decisions=len(unsupported_keys),
            support_status=status,
            traces=traces,
        )


@dataclass(frozen=True)
class DreamRSIProposal:
    active_policy_id: str
    candidate_policy: DeclarativeTrainingAllocationPolicy
    active_replay_score: float
    candidate_replay_score: float
    replay_gain: float
    candidate_support: SupportStatus
    status: str
    candidates_considered: int
    revision_history: tuple["PolicyRevisionReceipt", ...] = ()


@dataclass(frozen=True)
class CandidateReplayEvaluation:
    policy_id: str
    mean_score: float
    support_status: SupportStatus
    unsupported_decisions: int


@dataclass(frozen=True)
class PolicyRevisionReceipt:
    revision_index: int
    incumbent_policy_id: str
    selected_policy_id: str
    incumbent_score: float
    selected_score: float
    evaluations: tuple[CandidateReplayEvaluation, ...]


class DreamRSIOptimizer:
    """Bounded replay optimization with incumbent protection and support gating."""

    def __init__(
        self,
        pool: SupportAwareReplayPool,
        mutator: DeclarativeTrainingPolicyMutator | None = None,
        *,
        min_replay_gain: float = 0.0,
    ) -> None:
        self.pool = pool
        self.mutator = mutator or DeclarativeTrainingPolicyMutator()
        self.min_replay_gain = float(min_replay_gain)

    def propose(
        self,
        active: DeclarativeTrainingAllocationPolicy,
        *,
        max_rounds: int = 64,
        revision_rounds: int = 4,
    ) -> DreamRSIProposal:
        """Iteratively revise the policy from replay feedback.

        Each development round evaluates a bounded declarative neighborhood around
        the current incumbent on the same replay pool, records the feedback, and uses
        the best supported improvement as the starting point for the next revision.
        This is the safe Aether analogue of DREAM-RSI's repeated policy-development
        agent revisions; no arbitrary Python is generated or executed.
        """
        if revision_rounds < 1:
            raise ValueError("revision_rounds must be >= 1")
        active_report = self.pool.evaluate(active, max_rounds=max_rounds)
        current_policy = active
        current_report = active_report
        receipts: list[PolicyRevisionReceipt] = []
        total_candidates = 1
        for revision_index in range(int(revision_rounds)):
            candidates = self.mutator.neighbors(current_policy)
            total_candidates += max(0, len(candidates) - 1)
            evaluations: list[tuple[DeclarativeTrainingAllocationPolicy, SupportAwareReplayReport]] = []
            for candidate in candidates:
                report = current_report if candidate.policy_id == current_policy.policy_id else self.pool.evaluate(candidate, max_rounds=max_rounds)
                evaluations.append((candidate, report))
            eligible = [
                (candidate, report)
                for candidate, report in evaluations
                if report.support_status is not SupportStatus.UNSUPPORTED
            ]
            selected_policy = current_policy
            selected_report = current_report
            if eligible:
                candidate, report = max(eligible, key=lambda item: (item[1].mean_score, item[0].sha256))
                if report.mean_score > current_report.mean_score + self.min_replay_gain:
                    selected_policy, selected_report = candidate, report
            receipts.append(PolicyRevisionReceipt(
                revision_index=revision_index,
                incumbent_policy_id=current_policy.policy_id,
                selected_policy_id=selected_policy.policy_id,
                incumbent_score=float(current_report.mean_score),
                selected_score=float(selected_report.mean_score),
                evaluations=tuple(
                    CandidateReplayEvaluation(
                        policy_id=candidate.policy_id,
                        mean_score=float(report.mean_score),
                        support_status=report.support_status,
                        unsupported_decisions=int(report.unsupported_decisions),
                    )
                    for candidate, report in evaluations
                ),
            ))
            if selected_policy.policy_id == current_policy.policy_id:
                break
            current_policy, current_report = selected_policy, selected_report

        gain = current_report.mean_score - active_report.mean_score
        if current_policy.policy_id == active.policy_id or gain <= self.min_replay_gain:
            return DreamRSIProposal(
                active_policy_id=active.policy_id,
                candidate_policy=active,
                active_replay_score=active_report.mean_score,
                candidate_replay_score=active_report.mean_score,
                replay_gain=0.0,
                candidate_support=active_report.support_status,
                status="INCUMBENT_RETAINED",
                candidates_considered=total_candidates,
                revision_history=tuple(receipts),
            )
        status = "PROPOSED_FOR_ONLINE_QUALIFICATION"
        if current_report.support_status is SupportStatus.WEAK_SUPPORT:
            status = "ONLINE_PROBE_REQUIRED"
        return DreamRSIProposal(
            active_policy_id=active.policy_id,
            candidate_policy=current_policy,
            active_replay_score=active_report.mean_score,
            candidate_replay_score=current_report.mean_score,
            replay_gain=float(gain),
            candidate_support=current_report.support_status,
            status=status,
            candidates_considered=total_candidates,
            revision_history=tuple(receipts),
        )



@dataclass(frozen=True)
class OnlineMetaPolicyResult:
    policy_id: str
    seed: int
    score: float
    provenance_sha256: str
    contract_sha256: str
    evidence: EvidenceClass = EvidenceClass.VALIDATED

    def __post_init__(self) -> None:
        if self.seed < 0 or not math.isfinite(float(self.score)):
            raise ValueError("seed/score invalid")
        if not self.evidence.grounded:
            raise ValueError("online qualification requires grounded evidence")
        if not _valid_sha256(self.provenance_sha256) or not _valid_sha256(self.contract_sha256):
            raise ValueError("online result requires provenance and contract SHA-256")


@dataclass(frozen=True)
class OnlinePromotionReceipt:
    status: str
    candidate_policy_id: str
    baseline_policy_id: str
    seeds: tuple[int, ...]
    mean_paired_gain: float
    worst_paired_gain: float
    failures: tuple[str, ...]


class OnlineQualificationGate:
    """Replay never promotes.  Paired grounded online evidence does."""

    def __init__(
        self,
        contract: DreamRSIContract,
        *,
        minimum_seeds: int = 5,
        minimum_mean_gain: float = 0.0,
        max_seed_regression: float = 0.05,
    ) -> None:
        if minimum_seeds < 2:
            raise ValueError("minimum_seeds must be >= 2")
        self.contract = contract
        self.minimum_seeds = int(minimum_seeds)
        self.minimum_mean_gain = float(minimum_mean_gain)
        self.max_seed_regression = float(max_seed_regression)

    def evaluate(self, proposal: DreamRSIProposal, results: Iterable[OnlineMetaPolicyResult]) -> OnlinePromotionReceipt:
        rows = list(results)
        failures: list[str] = []
        if proposal.candidate_policy.policy_id == proposal.active_policy_id:
            failures.append("proposal did not change incumbent")
        if proposal.status not in {"PROPOSED_FOR_ONLINE_QUALIFICATION", "ONLINE_PROBE_REQUIRED"}:
            failures.append("proposal is not eligible for online qualification")
        expected_contract = self.contract.sha256
        if any(row.contract_sha256 != expected_contract for row in rows):
            failures.append("agent/evaluator/interface contract mismatch")
        keys = [(row.policy_id, row.seed) for row in rows]
        if len(keys) != len(set(keys)):
            failures.append("duplicate policy/seed evidence")
        baseline = {row.seed: row for row in rows if row.policy_id == proposal.active_policy_id and row.contract_sha256 == expected_contract}
        candidate = {row.seed: row for row in rows if row.policy_id == proposal.candidate_policy.policy_id and row.contract_sha256 == expected_contract}
        seeds = tuple(sorted(set(baseline) & set(candidate)))
        if len(seeds) < self.minimum_seeds:
            failures.append(f"paired grounded seeds {len(seeds)} < required {self.minimum_seeds}")
        gains = [candidate[seed].score - baseline[seed].score for seed in seeds]
        mean_gain = float(statistics.fmean(gains)) if gains else float("-inf")
        worst_gain = float(min(gains)) if gains else float("-inf")
        if gains and mean_gain <= self.minimum_mean_gain:
            failures.append("candidate mean paired gain did not clear threshold")
        if gains and worst_gain < -abs(self.max_seed_regression):
            failures.append("candidate exceeded allowed per-seed regression")
        return OnlinePromotionReceipt(
            status="PASS" if not failures else "FAIL",
            candidate_policy_id=proposal.candidate_policy.policy_id,
            baseline_policy_id=proposal.active_policy_id,
            seeds=seeds,
            mean_paired_gain=mean_gain,
            worst_paired_gain=worst_gain,
            failures=tuple(failures),
        )


class DreamRSIMetaController:
    """Closed loop: dream cheaply, qualify expensively, then update incumbent."""

    def __init__(
        self,
        active_policy: DeclarativeTrainingAllocationPolicy,
        pool: SupportAwareReplayPool,
        contract: DreamRSIContract,
        optimizer: DreamRSIOptimizer | None = None,
        qualification_gate: OnlineQualificationGate | None = None,
    ) -> None:
        if pool.contract_sha256 != contract.sha256:
            raise ValueError("replay pool contract does not match controller contract")
        self.active_policy = active_policy
        self.pool = pool
        self.contract = contract
        self.optimizer = optimizer or DreamRSIOptimizer(pool)
        self.qualification_gate = qualification_gate or OnlineQualificationGate(contract)
        self.last_proposal: DreamRSIProposal | None = None
        self.promotion_history: list[OnlinePromotionReceipt] = []

    def dream(self, *, max_rounds: int = 64, revision_rounds: int = 4) -> DreamRSIProposal:
        self.last_proposal = self.optimizer.propose(
            self.active_policy,
            max_rounds=max_rounds,
            revision_rounds=revision_rounds,
        )
        return self.last_proposal

    def ingest_online_world(self, world: GroundedReplayWorld) -> None:
        """Expand the replay simulator pool after a completed real online round."""
        if world.contract.sha256 != self.contract.sha256:
            raise ValueError("online world contract mismatch")
        support_policy = self.pool.support_policy
        worlds = (*self.pool.worlds, world)
        self.pool = SupportAwareReplayPool(worlds, support_policy)
        # Rebind the optimizer to the expanded history while preserving its mutation rules.
        self.optimizer = DreamRSIOptimizer(
            self.pool,
            self.optimizer.mutator,
            min_replay_gain=self.optimizer.min_replay_gain,
        )

    def validate_and_promote(self, results: Iterable[OnlineMetaPolicyResult]) -> OnlinePromotionReceipt:
        if self.last_proposal is None:
            raise RuntimeError("dream() must be called before online qualification")
        receipt = self.qualification_gate.evaluate(self.last_proposal, results)
        self.promotion_history.append(receipt)
        if receipt.status == "PASS":
            self.active_policy = self.last_proposal.candidate_policy
        return receipt


def ground_execution_receipt(
    receipt: ExactAllocationExecutionReceipt,
    *,
    outcome_id: str,
    parent_id: str | None,
    option: TrainingAllocationOption,
    contract: DreamRSIContract,
    evidence: EvidenceClass = EvidenceClass.VALIDATED,
) -> GroundedTrainingOutcome:
    """Bind a real learning-system execution receipt into grounded replay evidence."""
    if option.decision.execution_key != receipt.decision.execution_key:
        raise ValueError("receipt decision does not match grounded option")
    provenance = _sha(receipt.to_dict())
    return GroundedTrainingOutcome(
        outcome_id=str(outcome_id),
        parent_id=parent_id,
        iteration=int(receipt.iteration),
        seed=int(receipt.seed),
        option=option,
        metrics={
            "quality": float(
                receipt.heldout_success_rate if receipt.generalization_evaluated else receipt.actor_success_rate
            ),
            "transfer": float(receipt.transfer_success_rate if receipt.generalization_evaluated else 0.0),
            "adaptation": 0.0,
            "failure_rate": float(
                max(
                    0.0,
                    1.0 - (receipt.heldout_success_rate if receipt.generalization_evaluated else receipt.actor_success_rate),
                )
            ),
            "collection_success_rate": float(receipt.collection_success_rate),
            "mean_return": float(receipt.actor_mean_return),
            "heldout_mean_return": float(receipt.heldout_mean_return),
            "transfer_mean_return": float(receipt.transfer_mean_return),
            "world_loss": float(receipt.world_loss),
        },
        compute_cost=float(receipt.wall_seconds),
        latency_seconds=float(receipt.wall_seconds),
        provenance_sha256=provenance,
        contract_sha256=contract.sha256,
        evidence=evidence,
        offered_options=tuple(receipt.offered_options),
    )
