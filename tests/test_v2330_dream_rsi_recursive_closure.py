from __future__ import annotations

from pathlib import Path

import pytest

from awa.v2.learning_system import (
    DeclarativeTrainingAllocationPolicy,
    TrainingAllocationDecision,
    TrainingAllocationOption,
    execute_training_allocation,
)
from awa.v2.meta_exploration import EvidenceClass
from awa.v2.research_os import (
    DreamRSIContract,
    DreamRSIMetaController,
    FixedVsDreamComparisonPlan,
    GroundedReplayWorld,
    GroundedTrainingOutcome,
    OnlineQualificationGate,
    ReplayObjective,
    SupportAwareReplayPool,
    SupportPolicy,
    SupportStatus,
    ground_execution_receipt,
)
from awa.v2.research_os.dream_rsi import DreamRSIOptimizer
from awa.v2.learning_system.meta_allocation import DeclarativeTrainingPolicyMutator
from awa.v2.system_split import validate_split_packages


def _contract() -> DreamRSIContract:
    return DreamRSIContract("a" * 64, "b" * 64)


def _option(
    factors: tuple[str, ...],
    *,
    deficit: float,
    quality_hint: float = 0.0,
    worlds: int = 1,
    budget: int = 100,
) -> TrainingAllocationOption:
    decision = TrainingAllocationDecision(
        factor_signature=tuple(sorted(factors)),
        collector="coverage",
        transition_budget=budget,
        parallel_worlds=worlds,
    )
    return TrainingAllocationOption(
        decision=decision,
        capability_deficit=deficit,
        uncertainty=0.0,
        transfer_gap=0.0,
        novelty=quality_hint,
        expected_cost=0.0,
    )


def _outcome(
    oid: str,
    option: TrainingAllocationOption,
    menu: tuple[TrainingAllocationOption, ...],
    quality: float,
    *,
    contract: DreamRSIContract,
) -> GroundedTrainingOutcome:
    return GroundedTrainingOutcome(
        outcome_id=oid,
        parent_id=None,
        iteration=0,
        seed=1,
        option=option,
        metrics={"quality": quality, "transfer": 0.0, "adaptation": 0.0, "failure_rate": 0.0},
        compute_cost=0.0,
        latency_seconds=0.0,
        provenance_sha256=(oid.encode().hex() + "0" * 64)[:64],
        contract_sha256=contract.sha256,
        evidence=EvidenceClass.VALIDATED,
        offered_options=menu,
    )


def test_v233_worker_budget_counts_parallel_worlds_not_meta_decisions():
    a = _option(("memory",), deficit=1.0, worlds=3)
    b = _option(("combat",), deficit=0.9, worlds=2)
    c = _option(("navigation",), deficit=0.8, worlds=1)
    policy = DeclarativeTrainingAllocationPolicy(
        "workers",
        deficit_weight=1.0,
        uncertainty_weight=0.0,
        transfer_weight=0.0,
        novelty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=3,
        worker_budget=4,
    )
    chosen = policy.choose_batch((a, b, c))
    assert [row.parallel_worlds for row in chosen] == [3, 1]
    assert sum(row.parallel_worlds for row in chosen) == 4


def test_v233_complete_menu_marks_unobserved_choice_unsupported():
    contract = _contract()
    observed = _option(("navigation",), deficit=0.1)
    unobserved = _option(("memory", "combat"), deficit=0.9)
    menu = (observed, unobserved)
    world = GroundedReplayWorld(
        "menu-world",
        [_outcome("observed", observed, menu, 0.2, contract=contract)],
        contract=contract,
    )
    policy = DeclarativeTrainingAllocationPolicy(
        "prefer-hard",
        deficit_weight=1.0,
        uncertainty_weight=0.0,
        transfer_weight=0.0,
        novelty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=1,
        worker_budget=1,
    )
    trace = world.run(policy, max_rounds=1)
    assert trace.selected_outcome_ids == ()
    assert trace.unsupported_execution_keys == (unobserved.decision.execution_key,)
    report = SupportAwareReplayPool([world], SupportPolicy(min_worlds=1, min_selected_nodes=1)).evaluate(policy, max_rounds=1)
    assert report.support_status is SupportStatus.UNSUPPORTED
    assert report.unsupported_decisions == 1


def test_v233_exact_paper_objective_uses_world_attempts_and_parallelism():
    contract = _contract()
    option = _option(("navigation",), deficit=1.0, worlds=3)
    world = GroundedReplayWorld(
        "paper-world",
        [_outcome("x", option, (option,), 2.0, contract=contract)],
        contract=contract,
        objective=ReplayObjective(mode="paper", paper_beta1=0.1, paper_beta2=0.2),
    )
    policy = DeclarativeTrainingAllocationPolicy("p", stop_threshold=-1.0, max_parallel=1, worker_budget=3)
    trace = world.run(policy, max_rounds=1)
    assert trace.attempted_worlds == 3
    assert trace.score == pytest.approx(2.0 - 0.1 * 3 + 0.2 * 3)


def test_v233_optimizer_records_iterative_policy_development_feedback():
    contract = _contract()
    easy = _option(("navigation",), deficit=0.1)
    hard = _option(("memory", "combat"), deficit=0.9)
    menu = (easy, hard)
    worlds = [
        GroundedReplayWorld(
            f"w{seed}",
            [
                _outcome(f"easy-{seed}", easy, menu, 0.1, contract=contract),
                _outcome(f"hard-{seed}", hard, menu, 0.9, contract=contract),
            ],
            contract=contract,
        )
        for seed in (1, 2)
    ]
    pool = SupportAwareReplayPool(worlds, SupportPolicy(min_worlds=2, min_selected_nodes=2, min_replay_coverage=0.1))
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
        worker_budget=1,
    )
    optimizer = DreamRSIOptimizer(pool, DeclarativeTrainingPolicyMutator(step=0.1, fields=("deficit_weight",)))
    proposal = optimizer.propose(active, max_rounds=1, revision_rounds=4)
    assert proposal.revision_history
    assert proposal.revision_history[0].evaluations
    assert proposal.candidates_considered >= 3
    assert proposal.candidate_policy.policy_id != active.policy_id


def test_v233_real_exact_transition_execution_is_groundable(tmp_path: Path):
    contract = _contract()
    decision = TrainingAllocationDecision(
        factor_signature=("navigation",),
        collector="coverage",
        transition_budget=24,
        parallel_worlds=1,
    )
    option = TrainingAllocationOption(decision, capability_deficit=0.5, expected_cost=0.1)
    receipt = execute_training_allocation(
        decision,
        (option,),
        tmp_path / "real",
        seed=3,
        iteration=0,
        base_seed=3,
        difficulty=0.2,
        horizon=12,
        device="cpu",
        sequence_length=2,
        hidden=16,
        world_epochs=1,
        actor_epochs=1,
        calibration_epochs=0,
        batch_size=12,
        one_step_aux_epochs=0,
    )
    assert receipt.exact_transitions == 24
    grounded = ground_execution_receipt(
        receipt,
        outcome_id="real-0",
        parent_id=None,
        option=option,
        contract=contract,
    )
    assert grounded.offered_options == (option,)
    assert grounded.evidence.grounded
    assert (tmp_path / "real" / "allocation_execution_receipt.json").exists()


def test_v233_recursive_controller_expands_replay_pool_after_real_history():
    contract = _contract()
    option = _option(("navigation",), deficit=0.5)
    first = GroundedReplayWorld(
        "first",
        [_outcome("first-out", option, (option,), 0.5, contract=contract)],
        contract=contract,
    )
    second = GroundedReplayWorld(
        "second",
        [_outcome("second-out", option, (option,), 0.6, contract=contract)],
        contract=contract,
    )
    active = DeclarativeTrainingAllocationPolicy("active", stop_threshold=-1.0, max_parallel=1, worker_budget=1)
    controller = DreamRSIMetaController(
        active,
        SupportAwareReplayPool([first], SupportPolicy(min_worlds=1, min_selected_nodes=1)),
        contract,
        qualification_gate=OnlineQualificationGate(contract, minimum_seeds=2),
    )
    controller.ingest_online_world(second)
    assert len(controller.pool.worlds) == 2
    assert controller.optimizer.pool is controller.pool


def test_v233_fixed_vs_dream_plan_requires_identical_starting_policy_and_budget_contract():
    policy = DeclarativeTrainingAllocationPolicy("same")
    plan = FixedVsDreamComparisonPlan(
        fixed_policy_sha256=policy.sha256,
        adaptive_initial_policy_sha256=policy.sha256,
        contract_sha256=_contract().sha256,
        seeds=(1, 2, 3, 4, 5),
        rounds=5,
        transition_budget_per_round=25000,
    )
    assert len(plan.sha256) == 64
    with pytest.raises(ValueError):
        FixedVsDreamComparisonPlan(
            fixed_policy_sha256="1" * 64,
            adaptive_initial_policy_sha256="2" * 64,
            contract_sha256=_contract().sha256,
            seeds=(1, 2),
            rounds=1,
            transition_budget_per_round=1,
        )


def test_v233_system_split_still_has_no_illegal_dependency_edges():
    assert validate_split_packages("src") == []
