from __future__ import annotations

import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.learning_system import (
    DeclarativeTrainingAllocationPolicy,
    TrainingAllocationDecision,
    TrainingAllocationOption,
    build_collector_from_decision,
    materialize_training_allocation,
)
from awa.v2.meta_exploration import EvidenceClass
from awa.v2.research_os import (
    DreamRSIContract,
    DreamRSIMetaController,
    GroundedReplayWorld,
    GroundedTrainingOutcome,
    OnlineMetaPolicyResult,
    OnlineQualificationGate,
    SupportAwareReplayPool,
    SupportPolicy,
    SupportStatus,
)
from awa.v2.research_os.dream_rsi import DreamRSIOptimizer
from awa.v2.learning_system.meta_allocation import DeclarativeTrainingPolicyMutator
from awa.v2.system_split import validate_split_packages


def contract() -> DreamRSIContract:
    return DreamRSIContract("1" * 64, "2" * 64)


def outcome(
    oid: str,
    *,
    seed: int,
    factors=("navigation",),
    deficit=0.1,
    quality=0.2,
    parent=None,
    evidence=EvidenceClass.VALIDATED,
    contract_sha=None,
):
    c = contract()
    decision = TrainingAllocationDecision(
        factor_signature=tuple(sorted(factors)),
        collector="aether_actor",
        transition_budget=1000,
        planner_budget=0,
        parallel_worlds=2,
    )
    option = TrainingAllocationOption(
        decision=decision,
        capability_deficit=deficit,
        uncertainty=0.2,
        transfer_gap=deficit,
        novelty=0.1,
        expected_cost=0.1,
        historical_support=2,
    )
    return GroundedTrainingOutcome(
        outcome_id=oid,
        parent_id=parent,
        iteration=1,
        seed=seed,
        option=option,
        metrics={
            "quality": quality,
            "transfer": quality * 0.8,
            "adaptation": quality * 0.5,
            "failure_rate": 0.0,
        },
        compute_cost=0.1,
        latency_seconds=0.01,
        provenance_sha256=f"{seed + len(oid):064x}"[-64:],
        contract_sha256=contract_sha or c.sha256,
        evidence=evidence,
    )


def world(seed: int) -> GroundedReplayWorld:
    c = contract()
    return GroundedReplayWorld(
        f"world-{seed}",
        [
            outcome(f"nav-{seed}", seed=seed, deficit=0.1, quality=0.2),
            outcome(
                f"hard-{seed}",
                seed=seed,
                factors=("memory", "combat"),
                deficit=0.9,
                quality=0.9,
            ),
        ],
        contract=c,
    )


def test_version_bumped_for_dream_rsi_release():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_training_allocation_contract_is_declarative_and_exactly_hashable():
    decision = TrainingAllocationDecision(
        factor_signature=("combat", "memory"),
        collector="aether_planner",
        transition_budget=25000,
        planner_budget=32,
        parallel_worlds=4,
    )
    assert len(decision.execution_key) == 64
    assert decision.execution_key == TrainingAllocationDecision(**{
        **decision.__dict__,
    }).execution_key
    with pytest.raises(ValueError):
        TrainingAllocationDecision(
            factor_signature=("memory", "combat"),
            collector="coverage",
            transition_budget=1,
        )



def test_shared_decision_materializes_into_real_factorized_tasks_and_collector():
    decision = TrainingAllocationDecision(
        factor_signature=("combat", "memory"),
        collector="coverage",
        transition_budget=5000,
        planner_budget=0,
        parallel_worlds=2,
    )
    plan = materialize_training_allocation(
        decision,
        base_seed=32,
        difficulty=0.4,
        split="train",
        validate_tasks=True,
    )
    assert len(plan.tasks) == 2
    assert all(set(task.concepts) == {"combat", "memory"} for task in plan.tasks)
    collector = build_collector_from_decision(decision, seed=9)
    assert collector.receipt()["type"] == "coverage"


def test_replay_rejects_nongrounded_history_and_contract_drift():
    c = contract()
    with pytest.raises(ValueError):
        outcome("sim", seed=1, evidence=EvidenceClass.MODEL_SIMULATED)
    with pytest.raises(ValueError):
        GroundedReplayWorld(
            "bad-contract",
            [outcome("x", seed=1, contract_sha="3" * 64)],
            contract=c,
        )


def test_same_policy_decision_interface_works_in_grounded_replay():
    policy = DeclarativeTrainingAllocationPolicy(
        "prefer-deficit",
        deficit_weight=1.0,
        uncertainty_weight=0.0,
        transfer_weight=0.0,
        novelty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=1,
    )
    trace = world(1).run(policy, max_rounds=1)
    assert len(trace.selected_outcome_ids) == 1
    assert trace.selected_outcome_ids[0].startswith("hard-")
    assert trace.score > 0


def test_support_gate_distinguishes_supported_and_undercovered_replay():
    policy = DeclarativeTrainingAllocationPolicy(
        "p",
        deficit_weight=1.0,
        stop_threshold=-1.0,
        max_parallel=1,
    )
    supported = SupportAwareReplayPool(
        [world(1), world(2)],
        SupportPolicy(min_worlds=2, min_selected_nodes=2, min_replay_coverage=0.1),
    ).evaluate(policy, max_rounds=1)
    assert supported.support_status is SupportStatus.SUPPORTED

    weak = SupportAwareReplayPool(
        [world(1), world(2)],
        SupportPolicy(min_worlds=2, min_selected_nodes=10, min_replay_coverage=0.9),
    ).evaluate(policy, max_rounds=1)
    assert weak.support_status is SupportStatus.WEAK_SUPPORT


def test_optimizer_keeps_incumbent_unless_supported_candidate_beats_it():
    pool = SupportAwareReplayPool(
        [world(1), world(2)],
        SupportPolicy(min_worlds=2, min_selected_nodes=2, min_replay_coverage=0.1),
    )
    active = DeclarativeTrainingAllocationPolicy(
        "active",
        deficit_weight=-0.05,
        uncertainty_weight=0.0,
        transfer_weight=0.0,
        novelty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=1,
    )
    optimizer = DreamRSIOptimizer(
        pool,
        DeclarativeTrainingPolicyMutator(step=0.1, fields=("deficit_weight",)),
    )
    proposal = optimizer.propose(active, max_rounds=1)
    assert proposal.candidate_policy.policy_id != active.policy_id
    assert proposal.replay_gain > 0
    assert proposal.status == "PROPOSED_FOR_ONLINE_QUALIFICATION"


def test_replay_winner_cannot_promote_without_paired_grounded_online_evidence():
    c = contract()
    pool = SupportAwareReplayPool(
        [world(1), world(2)],
        SupportPolicy(min_worlds=2, min_selected_nodes=2, min_replay_coverage=0.1),
    )
    active = DeclarativeTrainingAllocationPolicy(
        "active",
        deficit_weight=-0.05,
        uncertainty_weight=0.0,
        transfer_weight=0.0,
        novelty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=1,
    )
    controller = DreamRSIMetaController(
        active,
        pool,
        c,
        optimizer=DreamRSIOptimizer(
            pool,
            DeclarativeTrainingPolicyMutator(step=0.1, fields=("deficit_weight",)),
        ),
        qualification_gate=OnlineQualificationGate(c, minimum_seeds=2),
    )
    proposal = controller.dream(max_rounds=1)
    assert controller.active_policy.policy_id == active.policy_id

    with pytest.raises(RuntimeError):
        DreamRSIMetaController(active, pool, c).validate_and_promote([])

    rows = []
    for seed in (11, 12):
        rows.append(OnlineMetaPolicyResult(active.policy_id, seed, 0.5, f"{seed:064x}", c.sha256))
        rows.append(OnlineMetaPolicyResult(proposal.candidate_policy.policy_id, seed, 0.6, f"{seed + 100:064x}", c.sha256))
    receipt = controller.validate_and_promote(rows)
    assert receipt.status == "PASS"
    assert controller.active_policy.policy_id == proposal.candidate_policy.policy_id


def test_online_gate_fails_closed_on_agent_or_evaluator_contract_mismatch():
    c = contract()
    pool = SupportAwareReplayPool([world(1), world(2)])
    active = DeclarativeTrainingAllocationPolicy(
        "active",
        deficit_weight=-0.05,
        stop_threshold=-1.0,
        max_parallel=1,
    )
    proposal = DreamRSIOptimizer(
        pool,
        DeclarativeTrainingPolicyMutator(step=0.1, fields=("deficit_weight",)),
    ).propose(active, max_rounds=1)
    gate = OnlineQualificationGate(c, minimum_seeds=2)
    rows = []
    for seed in (1, 2):
        rows.append(OnlineMetaPolicyResult(active.policy_id, seed, 0.5, f"{seed:064x}", c.sha256))
        rows.append(OnlineMetaPolicyResult(proposal.candidate_policy.policy_id, seed, 0.6, f"{seed + 10:064x}", "9" * 64))
    receipt = gate.evaluate(proposal, rows)
    assert receipt.status == "FAIL"
    assert "agent/evaluator/interface contract mismatch" in receipt.failures


def test_v232_canonical_namespaces_still_obey_system_split():
    assert validate_split_packages("src") == []
