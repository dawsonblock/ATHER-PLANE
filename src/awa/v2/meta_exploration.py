from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Hashable, Iterable
import hashlib
import json
import math
import statistics


class EvidenceClass(str, Enum):
    """Origin of a discovery outcome.

    Only OBSERVED and VALIDATED outcomes are allowed to support empirical
    promotion. MODEL_SIMULATED and COUNTERFACTUAL outcomes are useful for search
    but remain hypotheses until checked in the real environment.
    """

    OBSERVED = "observed"
    MODEL_SIMULATED = "model_simulated"
    COUNTERFACTUAL = "counterfactual"
    VALIDATED = "validated"

    @property
    def grounded(self) -> bool:
        return self in {EvidenceClass.OBSERVED, EvidenceClass.VALIDATED}


@dataclass(frozen=True)
class DiscoveryNode:
    node_id: str
    parent_id: str | None
    iteration: int
    evidence: EvidenceClass
    metrics: dict[str, float] = field(default_factory=dict)
    compute_cost: float = 0.0
    latency_seconds: float = 0.0
    artifact_ref: str | None = None
    provenance_sha256: str | None = None
    generation_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.node_id:
            raise ValueError("node_id required")
        if self.iteration < 0 or self.generation_index < 0:
            raise ValueError("iteration/generation_index must be >= 0")
        if not math.isfinite(float(self.compute_cost)) or self.compute_cost < 0:
            raise ValueError("compute_cost must be finite and >= 0")
        if not math.isfinite(float(self.latency_seconds)) or self.latency_seconds < 0:
            raise ValueError("latency_seconds must be finite and >= 0")
        for name, value in self.metrics.items():
            if not math.isfinite(float(value)):
                raise ValueError(f"metric {name!r} must be finite")
        if self.evidence.grounded and self.provenance_sha256 is None:
            raise ValueError("grounded discovery nodes require provenance_sha256")
        if self.provenance_sha256 is not None:
            p = self.provenance_sha256.lower()
            if len(p) != 64 or any(c not in "0123456789abcdef" for c in p):
                raise ValueError("provenance_sha256 must be a 64-character hex digest")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["evidence"] = self.evidence.value
        return d

    @property
    def sha256(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), default=str).encode()
        return hashlib.sha256(raw).hexdigest()


class DiscoveryTree:
    """Append-only discovery tree with one primary parent per node.

    The root is always present. Historical replay may reveal grounded children
    but never upgrades simulated outcomes into empirical evidence.
    """

    ROOT_ID = "root"

    def __init__(self, tree_id: str):
        if not tree_id:
            raise ValueError("tree_id required")
        self.tree_id = str(tree_id)
        self._nodes: dict[str, DiscoveryNode] = {
            self.ROOT_ID: DiscoveryNode(
                node_id=self.ROOT_ID,
                parent_id=None,
                iteration=0,
                evidence=EvidenceClass.OBSERVED,
                metrics={},
                provenance_sha256="0" * 64,
            )
        }
        self._children: dict[str, list[str]] = {self.ROOT_ID: []}

    @property
    def nodes(self) -> tuple[DiscoveryNode, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes))

    def get(self, node_id: str) -> DiscoveryNode:
        return self._nodes[node_id]

    def add(self, node: DiscoveryNode) -> None:
        if node.node_id == self.ROOT_ID or node.node_id in self._nodes:
            raise ValueError(f"duplicate/reserved node_id: {node.node_id}")
        if node.parent_id is None or node.parent_id not in self._nodes:
            raise ValueError("parent_id must reference an existing node")
        siblings = self._children.setdefault(node.parent_id, [])
        if any(self._nodes[s].generation_index == node.generation_index for s in siblings):
            raise ValueError("generation_index must be unique among siblings")
        self._nodes[node.node_id] = node
        siblings.append(node.node_id)
        siblings.sort(key=lambda n: (self._nodes[n].generation_index, n))
        self._children.setdefault(node.node_id, [])

    def children(self, node_id: str, *, grounded_only: bool = False) -> tuple[DiscoveryNode, ...]:
        out = [self._nodes[n] for n in self._children.get(node_id, [])]
        if grounded_only:
            out = [n for n in out if n.evidence.grounded]
        return tuple(out)

    def grounded_nodes(self) -> tuple[DiscoveryNode, ...]:
        return tuple(n for n in self.nodes if n.evidence.grounded)

    def to_dict(self) -> dict[str, Any]:
        return {"format": "awa-v2.20-discovery-tree-v1", "tree_id": self.tree_id, "nodes": [n.to_dict() for n in self.nodes]}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DiscoveryTree":
        if payload.get("format") != "awa-v2.20-discovery-tree-v1":
            raise ValueError("unsupported discovery tree format")
        tree = cls(payload["tree_id"])
        rows = [dict(x) for x in payload.get("nodes", []) if x.get("node_id") != cls.ROOT_ID]
        # Parent-first reconstruction. Fail if the payload is cyclic or incomplete.
        pending = rows[:]
        while pending:
            before = len(pending)
            for raw in pending[:]:
                if raw.get("parent_id") in tree._nodes:
                    raw["evidence"] = EvidenceClass(raw["evidence"])
                    tree.add(DiscoveryNode(**raw))
                    pending.remove(raw)
            if len(pending) == before:
                raise ValueError("discovery tree contains missing parents or a cycle")
        return tree


@dataclass(frozen=True)
class DeclarativeExplorationPolicy:
    """Bounded meta-exploration policy; no arbitrary code execution is involved."""

    policy_id: str
    quality_weight: float = 1.0
    novelty_weight: float = 0.25
    uncertainty_weight: float = 0.20
    transfer_weight: float = 0.30
    information_weight: float = 0.20
    cost_weight: float = 0.10
    new_world_bias: float = 0.35
    stop_threshold: float = -0.05
    max_parallel: int = 4
    new_world_fraction: float = 0.20
    adversarial_fraction: float = 0.10

    def __post_init__(self):
        if not self.policy_id:
            raise ValueError("policy_id required")
        vals = (
            self.quality_weight, self.novelty_weight, self.uncertainty_weight,
            self.transfer_weight, self.information_weight, self.cost_weight,
            self.new_world_bias, self.stop_threshold, self.new_world_fraction,
            self.adversarial_fraction,
        )
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError("policy parameters must be finite")
        if self.max_parallel < 1:
            raise ValueError("max_parallel must be >= 1")
        if not 0 <= self.new_world_fraction <= 1 or not 0 <= self.adversarial_fraction <= 1:
            raise ValueError("reserve fractions must lie in [0,1]")
        if self.new_world_fraction + self.adversarial_fraction > 1:
            raise ValueError("reserve fractions must sum to <= 1")
        if self.cost_weight < 0:
            raise ValueError("cost_weight must be >= 0")

    def score_parent(self, node: DiscoveryNode, *, is_root: bool = False) -> float:
        if is_root:
            return float(self.new_world_bias)
        m = node.metrics
        return float(
            self.quality_weight * m.get("quality", 0.0)
            + self.novelty_weight * m.get("novelty", 0.0)
            + self.uncertainty_weight * m.get("uncertainty_reduction", 0.0)
            + self.transfer_weight * m.get("transfer", 0.0)
            + self.information_weight * m.get("information_gain", 0.0)
            - self.cost_weight * node.compute_cost
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        raw=json.dumps(self.to_dict(),sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ReplayObjective:
    quality_weight: float = 1.0
    transfer_weight: float = 0.25
    information_weight: float = 0.10
    uncertainty_weight: float = 0.10
    compute_cost_weight: float = 0.10
    latency_weight: float = 0.0
    constraint_violation_weight: float = 1.0


@dataclass(frozen=True)
class ReplayResult:
    tree_id: str
    policy_id: str
    score: float
    revealed: tuple[str, ...]
    rounds: int
    stopped: bool
    compute_cost: float
    latency_seconds: float


class ReplayWorld:
    """Historical discovery tree used as a zero-execution-cost replay world.

    Replay exposes only grounded historical outcomes. Model-simulated and
    counterfactual nodes are deliberately invisible to this empirical simulator.
    """

    def __init__(self, tree: DiscoveryTree, objective: ReplayObjective | None = None):
        self.tree = tree
        self.objective = objective or ReplayObjective()

    def run(self, policy: DeclarativeExplorationPolicy, *, max_rounds: int = 64) -> ReplayResult:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        revealed = {DiscoveryTree.ROOT_ID}
        revealed_order: list[str] = []
        root_cursor = 0
        rounds = 0
        stopped = False

        def visible_children(parent: str) -> list[str]:
            return [n.node_id for n in self.tree.children(parent, grounded_only=True) if n.node_id in revealed]

        def eligible() -> list[str]:
            out: list[str] = []
            # Root can be selected repeatedly to open additional historical branches.
            root_children = list(self.tree.children(DiscoveryTree.ROOT_ID, grounded_only=True))
            if root_cursor < len(root_children):
                out.append(DiscoveryTree.ROOT_ID)
            for node_id in sorted(revealed - {DiscoveryTree.ROOT_ID}):
                # Following DREAM-RSI's root-or-leaf decision interface, only a
                # currently visible leaf may continue its branch.
                if visible_children(node_id):
                    continue
                if self.tree.children(node_id, grounded_only=True):
                    out.append(node_id)
            return out

        while rounds < max_rounds:
            candidates = eligible()
            if not candidates:
                break
            scored = sorted(
                ((policy.score_parent(self.tree.get(n), is_root=n == DiscoveryTree.ROOT_ID), n) for n in candidates),
                key=lambda x: (-x[0], x[1]),
            )
            selected = [n for score, n in scored[: policy.max_parallel] if score > policy.stop_threshold]
            if not selected:
                stopped = True
                break
            for parent in selected:
                children = list(self.tree.children(parent, grounded_only=True))
                if parent == DiscoveryTree.ROOT_ID:
                    if root_cursor >= len(children):
                        continue
                    child = children[root_cursor]
                    root_cursor += 1
                else:
                    # Non-root visible leaves can have multiple historical children
                    # in malformed/external data; use canonical first child only.
                    hidden = [c for c in children if c.node_id not in revealed]
                    if not hidden:
                        continue
                    child = hidden[0]
                revealed.add(child.node_id)
                revealed_order.append(child.node_id)
            rounds += 1

        nodes = [self.tree.get(n) for n in revealed_order]
        o = self.objective
        best_quality = max((n.metrics.get("quality", 0.0) for n in nodes), default=0.0)
        best_transfer = max((n.metrics.get("transfer", 0.0) for n in nodes), default=0.0)
        info = sum(n.metrics.get("information_gain", 0.0) for n in nodes)
        uncertainty = sum(n.metrics.get("uncertainty_reduction", 0.0) for n in nodes)
        violations = sum(n.metrics.get("constraint_violations", 0.0) for n in nodes)
        compute = sum(n.compute_cost for n in nodes)
        latency = sum(n.latency_seconds for n in nodes)
        score = (
            o.quality_weight * best_quality
            + o.transfer_weight * best_transfer
            + o.information_weight * info
            + o.uncertainty_weight * uncertainty
            - o.compute_cost_weight * compute
            - o.latency_weight * latency
            - o.constraint_violation_weight * violations
        )
        return ReplayResult(
            self.tree.tree_id, policy.policy_id, float(score), tuple(revealed_order), rounds,
            stopped, float(compute), float(latency)
        )


class ReplaySimulatorPool:
    def __init__(self, worlds: Iterable[ReplayWorld] = ()):
        self.worlds = list(worlds)

    def add_tree(self, tree: DiscoveryTree, objective: ReplayObjective | None = None) -> None:
        self.worlds.append(ReplayWorld(tree, objective))

    def evaluate(self, policy: DeclarativeExplorationPolicy, *, max_rounds: int = 64) -> dict[str, Any]:
        if not self.worlds:
            raise ValueError("replay simulator pool is empty")
        results = [w.run(policy, max_rounds=max_rounds) for w in self.worlds]
        scores = [r.score for r in results]
        return {
            "policy_id": policy.policy_id,
            "mean_score": float(statistics.fmean(scores)),
            "min_score": float(min(scores)),
            "max_score": float(max(scores)),
            "worlds": [asdict(r) for r in results],
        }


class DeclarativePolicyMutator:
    """Deterministic bounded neighborhood over policy parameters."""

    _FIELDS = (
        "quality_weight", "novelty_weight", "uncertainty_weight", "transfer_weight",
        "information_weight", "cost_weight", "new_world_bias", "stop_threshold",
    )

    def __init__(
        self,
        step: float = 0.10,
        *,
        min_new_world_fraction: float = 0.10,
        min_adversarial_fraction: float = 0.05,
        fields: Iterable[str] | None = None,
    ):
        if step <= 0 or not math.isfinite(step):
            raise ValueError("step must be finite and > 0")
        if not 0 <= min_new_world_fraction <= 1 or not 0 <= min_adversarial_fraction <= 1:
            raise ValueError("reserve floors must lie in [0,1]")
        if min_new_world_fraction + min_adversarial_fraction > 1:
            raise ValueError("reserve floors must sum to <= 1")
        self.step = float(step)
        self.min_new_world_fraction = float(min_new_world_fraction)
        self.min_adversarial_fraction = float(min_adversarial_fraction)
        self.fields = tuple(fields or self._FIELDS)
        unknown = set(self.fields) - set(self._FIELDS)
        if unknown:
            raise ValueError(f"unknown mutation fields: {sorted(unknown)}")
        if len(self.fields) != len(set(self.fields)):
            raise ValueError("mutation fields must be unique")

    def neighbors(self, base: DeclarativeExplorationPolicy) -> list[DeclarativeExplorationPolicy]:
        out = [base]
        for name in self.fields:
            for sign in (-1.0, 1.0):
                value = float(getattr(base, name)) + sign * self.step
                if name == "cost_weight":
                    value = max(0.0, value)
                p = replace(
                    base,
                    policy_id=f"{base.policy_id}:{name}:{'plus' if sign > 0 else 'minus'}",
                    **{name: value},
                    new_world_fraction=max(base.new_world_fraction, self.min_new_world_fraction),
                    adversarial_fraction=max(base.adversarial_fraction, self.min_adversarial_fraction),
                )
                out.append(p)
        return out


@dataclass(frozen=True)
class PolicyProposal:
    active_policy_id: str
    candidate: DeclarativeExplorationPolicy
    replay_score: float
    active_replay_score: float
    replay_gain: float
    status: str = "PROPOSED"
    candidates_considered: int = 0
    unique_execution_candidates: int = 0


class MetaPolicyOptimizer:
    def __init__(
        self,
        simulator_pool: ReplaySimulatorPool,
        mutator: DeclarativePolicyMutator | None = None,
        *,
        min_replay_gain: float = 0.0,
        candidate_equivalence_key: Callable[[DeclarativeExplorationPolicy], Hashable] | None = None,
    ):
        self.pool = simulator_pool
        self.mutator = mutator or DeclarativePolicyMutator()
        self.min_replay_gain = float(min_replay_gain)
        self.candidate_equivalence_key = candidate_equivalence_key

    def propose(self, active: DeclarativeExplorationPolicy, *, max_rounds: int = 64) -> PolicyProposal:
        active_eval = self.pool.evaluate(active, max_rounds=max_rounds)
        best_policy = active
        best_score = float(active_eval["mean_score"])
        neighbors = self.mutator.neighbors(active)
        seen: set[Hashable] = set()
        unique: list[DeclarativeExplorationPolicy] = []
        for candidate in neighbors:
            key: Hashable = (
                self.candidate_equivalence_key(candidate)
                if self.candidate_equivalence_key is not None
                else candidate.sha256
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append(candidate)
        for candidate in unique:
            score = float(self.pool.evaluate(candidate, max_rounds=max_rounds)["mean_score"])
            if score > best_score:
                best_policy, best_score = candidate, score
        gain = best_score - float(active_eval["mean_score"])
        if gain <= self.min_replay_gain:
            best_policy = active
            best_score = float(active_eval["mean_score"])
            gain = 0.0
        return PolicyProposal(
            active.policy_id,
            best_policy,
            best_score,
            float(active_eval["mean_score"]),
            float(gain),
            "PROPOSED",
            len(neighbors),
            len(unique),
        )


@dataclass(frozen=True)
class OnlinePolicyResult:
    policy_id: str
    seed: int
    score: float
    evidence: EvidenceClass
    provenance_sha256: str | None = None

    def __post_init__(self):
        if not math.isfinite(float(self.score)):
            raise ValueError("online score must be finite")
        if self.evidence.grounded and self.provenance_sha256 is None:
            raise ValueError("grounded online results require provenance_sha256")
        if self.provenance_sha256 is not None:
            p=self.provenance_sha256.lower()
            if len(p)!=64 or any(c not in "0123456789abcdef" for c in p):
                raise ValueError("provenance_sha256 must be a 64-character hex digest")


@dataclass(frozen=True)
class PromotionReceipt:
    status: str
    candidate_policy_id: str
    baseline_policy_id: str
    seeds: tuple[int, ...]
    mean_paired_gain: float
    worst_paired_gain: float
    failures: tuple[str, ...]


class ExplorationPolicyPromotionGate:
    """Fail-closed online gate. Replay alone can never promote a policy."""

    def __init__(self, *, minimum_seeds: int = 5, minimum_mean_gain: float = 0.0, max_seed_regression: float = 0.05):
        if minimum_seeds < 2:
            raise ValueError("minimum_seeds must be >= 2")
        self.minimum_seeds = int(minimum_seeds)
        self.minimum_mean_gain = float(minimum_mean_gain)
        self.max_seed_regression = float(max_seed_regression)

    def evaluate(self, proposal: PolicyProposal, results: Iterable[OnlinePolicyResult]) -> PromotionReceipt:
        rows = list(results)
        failures: list[str] = []
        keys=[(r.policy_id,r.seed) for r in rows]
        if len(keys)!=len(set(keys)):
            failures.append("duplicate policy/seed online evidence")
        if proposal.candidate.policy_id == proposal.active_policy_id:
            failures.append("proposal does not change the active policy")
        if any(not r.evidence.grounded for r in rows):
            failures.append("non-grounded online evidence is not promotable")
        baseline = {r.seed: r for r in rows if r.policy_id == proposal.active_policy_id and r.evidence.grounded}
        candidate = {r.seed: r for r in rows if r.policy_id == proposal.candidate.policy_id and r.evidence.grounded}
        seeds = tuple(sorted(set(baseline) & set(candidate)))
        if len(seeds) < self.minimum_seeds:
            failures.append(f"paired grounded seeds {len(seeds)} < required {self.minimum_seeds}")
        gains = [candidate[s].score - baseline[s].score for s in seeds]
        mean_gain = float(statistics.fmean(gains)) if gains else float("-inf")
        worst_gain = float(min(gains)) if gains else float("-inf")
        if gains and mean_gain <= self.minimum_mean_gain:
            failures.append("candidate mean paired gain did not clear promotion threshold")
        if gains and worst_gain < -abs(self.max_seed_regression):
            failures.append("candidate exceeded allowed per-seed regression")
        return PromotionReceipt(
            "PASS" if not failures else "FAIL",
            proposal.candidate.policy_id,
            proposal.active_policy_id,
            seeds,
            mean_gain,
            worst_gain,
            tuple(failures),
        )


class EvolvingExplorationController:
    """DREAM-RSI-style outer loop with explicit empirical promotion boundary."""

    def __init__(
        self,
        active_policy: DeclarativeExplorationPolicy,
        simulator_pool: ReplaySimulatorPool | None = None,
        optimizer: MetaPolicyOptimizer | None = None,
        promotion_gate: ExplorationPolicyPromotionGate | None = None,
    ):
        self.active_policy = active_policy
        self.pool = simulator_pool or ReplaySimulatorPool()
        self.optimizer = optimizer or MetaPolicyOptimizer(self.pool)
        self.promotion_gate = promotion_gate or ExplorationPolicyPromotionGate()
        self.last_proposal: PolicyProposal | None = None
        self.promotion_history: list[PromotionReceipt] = []

    def add_completed_tree(self, tree: DiscoveryTree, objective: ReplayObjective | None = None) -> None:
        self.pool.add_tree(tree, objective)

    def dream(self, *, max_rounds: int = 64) -> PolicyProposal:
        self.last_proposal = self.optimizer.propose(self.active_policy, max_rounds=max_rounds)
        return self.last_proposal

    def validate_and_promote(self, results: Iterable[OnlinePolicyResult]) -> PromotionReceipt:
        if self.last_proposal is None:
            raise RuntimeError("dream() must create a proposal before promotion")
        receipt = self.promotion_gate.evaluate(self.last_proposal, results)
        self.promotion_history.append(receipt)
        if receipt.status == "PASS":
            self.active_policy = self.last_proposal.candidate
            self.last_proposal = None
        return receipt
