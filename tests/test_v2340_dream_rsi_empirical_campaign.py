from __future__ import annotations

from pathlib import Path

import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.learning_system import (
    DeclarativeTrainingAllocationPolicy,
    TrainingAllocationDecision,
    TrainingAllocationOption,
)
from awa.v2.research_os import DreamRSIContract, ReplayObjective, SupportPolicy
from awa.v2.research_os.dream_rsi_campaign import (
    DreamRSICampaignConfig,
    MenuOptionSpec,
    SeedComparisonSummary,
    build_campaign_report,
)
from awa.v2.research_os.dream_rsi_loop import RealOnlineRoundConfig, execute_policy_online_world, load_online_world
from awa.v2.system_split import validate_split_packages


def _policy(cap: int = 20) -> DeclarativeTrainingAllocationPolicy:
    return DeclarativeTrainingAllocationPolicy(
        "p",
        deficit_weight=1.0,
        uncertainty_weight=0.0,
        transfer_weight=0.0,
        novelty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=4,
        worker_budget=4,
        transition_budget_cap=cap,
    )


def test_v234_version_and_split():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"
    assert validate_split_packages("src") == []


def test_v234_policy_enforces_total_transition_ceiling_independent_of_worker_ceiling():
    options = tuple(
        TrainingAllocationOption(
            TrainingAllocationDecision((factor,), "coverage", 8, parallel_worlds=1),
            capability_deficit=1.0 - index * 0.01,
        )
        for index, factor in enumerate(("navigation", "memory", "combat", "scarcity"))
    )
    chosen = _policy(cap=20).choose_batch(options)
    assert len(chosen) == 2
    assert sum(row.transition_budget for row in chosen) == 16
    assert sum(row.parallel_worlds for row in chosen) == 2


def test_v234_campaign_config_requires_budget_bound_initial_policy():
    with pytest.raises(ValueError):
        DreamRSICampaignConfig(
            seeds=(1, 2), rounds=2,
            round_transition_budget=20, option_transition_budget=10,
            worker_budget=4, max_parallel=4, revision_rounds=2, replay_max_rounds=2,
            mutation_step=0.1,
            menu=(MenuOptionSpec(("navigation",), 0.5),),
            initial_policy=DeclarativeTrainingAllocationPolicy("bad", max_parallel=4, worker_budget=4),
            support_policy=SupportPolicy(min_worlds=1, min_selected_nodes=1),
            objective=ReplayObjective(mode="paper"),
            runtime=RealOnlineRoundConfig(),
        )


def test_v234_grounded_world_roundtrip_binds_artifact_hashes(tmp_path: Path):
    contract = DreamRSIContract("a" * 64, "b" * 64, interface_version="awa-training-allocation-v2.34")
    decision = TrainingAllocationDecision(("navigation",), "coverage", 16, parallel_worlds=1)
    option = TrainingAllocationOption(decision, capability_deficit=0.5)
    policy = DeclarativeTrainingAllocationPolicy(
        "roundtrip", stop_threshold=-1.0, max_parallel=1, worker_budget=1, transition_budget_cap=16
    )
    world, receipt = execute_policy_online_world(
        policy,
        (option,),
        contract,
        tmp_path / "round",
        iteration=0,
        seed=7,
        config=RealOnlineRoundConfig(
            horizon=8,
            device="cpu",
            sequence_length=2,
            hidden=16,
            world_epochs=1,
            actor_epochs=1,
            calibration_epochs=0,
            batch_size=8,
            one_step_aux_epochs=0,
            evaluate_generalization=True,
            evaluation_replicates=1,
        ),
        objective=ReplayObjective(mode="paper", paper_beta1=0.0, paper_beta2=0.0),
    )
    assert receipt.artifacts and receipt.best_artifact is not None
    assert receipt.exact_transitions == 16
    assert world.outcomes[0].metrics["transfer"] >= 0.0
    loaded_world, loaded_receipt = load_online_world(tmp_path / "round", contract)
    assert loaded_receipt.world_sha256 == receipt.world_sha256
    assert loaded_receipt.best_artifact.world_checkpoint_sha256 == receipt.best_artifact.world_checkpoint_sha256
    assert loaded_world.to_dict() == world.to_dict()


def test_v234_report_uses_independent_final_generalization_gain_and_fails_closed():
    config = DreamRSICampaignConfig(
        seeds=(1, 2), rounds=2,
        round_transition_budget=20, option_transition_budget=10,
        worker_budget=2, max_parallel=2, revision_rounds=2, replay_max_rounds=2,
        mutation_step=0.1,
        menu=(MenuOptionSpec(("navigation",), 0.5),),
        initial_policy=DeclarativeTrainingAllocationPolicy(
            "same", max_parallel=2, worker_budget=2, transition_budget_cap=20
        ),
        support_policy=SupportPolicy(min_worlds=1, min_selected_nodes=1),
        objective=ReplayObjective(mode="paper"),
        runtime=RealOnlineRoundConfig(),
        bootstrap_samples=200,
        minimum_final_mean_gain=0.0,
        minimum_curve_mean_gain=0.0,
        max_seed_regression=0.2,
    )
    contract = DreamRSIContract("1" * 64, "2" * 64)
    rows = (
        SeedComparisonSummary(1, 0.50, 0.60, 0.10, 0.5, 0.6, 0.5, 0.6, 0.4, 0.5, 0.1, 40, 40, 1.0, 1.0, 1),
        SeedComparisonSummary(2, 0.50, 0.55, 0.05, 0.5, 0.6, 0.5, 0.5, 0.4, 0.45, 0.05, 40, 40, 1.0, 1.0, 1),
    )
    report = build_campaign_report(config, contract, rows)
    assert report.status == "PASS"
    assert report.mean_final_gain == pytest.approx(0.075)
    incomplete = build_campaign_report(config, contract, rows[:1])
    assert incomplete.status == "INSUFFICIENT_EVIDENCE"
