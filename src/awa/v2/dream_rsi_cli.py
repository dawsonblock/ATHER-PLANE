from __future__ import annotations

import json

from awa.v2.learning_system.meta_allocation import (
    DeclarativeTrainingAllocationPolicy,
    TrainingAllocationDecision,
    TrainingAllocationOption,
)
from awa.v2.meta_exploration import EvidenceClass
from awa.v2.research_os.dream_rsi import (
    DreamRSIContract,
    DreamRSIMetaController,
    GroundedReplayWorld,
    GroundedTrainingOutcome,
    OnlineMetaPolicyResult,
    OnlineQualificationGate,
    SupportAwareReplayPool,
    SupportPolicy,
)


def _history(contract: DreamRSIContract, seed: int, prefix: str):
    def option(factors, deficit, quality, idx):
        decision = TrainingAllocationDecision(
            factor_signature=tuple(sorted(factors)),
            collector="aether_actor",
            transition_budget=1000,
            parallel_worlds=2,
        )
        return GroundedTrainingOutcome(
            outcome_id=f"{prefix}-{idx}",
            parent_id=None,
            iteration=idx,
            seed=seed,
            option=TrainingAllocationOption(
                decision=decision,
                capability_deficit=deficit,
                uncertainty=0.2,
                transfer_gap=deficit,
                novelty=0.1,
                expected_cost=0.1,
                historical_support=2,
            ),
            metrics={"quality": quality, "transfer": quality * 0.8, "adaptation": quality * 0.5, "failure_rate": 0.0},
            compute_cost=0.1,
            latency_seconds=0.01,
            provenance_sha256=f"{seed + idx + 1:064x}"[-64:],
            contract_sha256=contract.sha256,
            evidence=EvidenceClass.VALIDATED,
        )

    return [
        option(("navigation",), 0.1, 0.2, 1),
        option(("memory", "combat"), 0.9, 0.9, 2),
    ]


def dream_rsi_smoke_main() -> None:
    contract = DreamRSIContract("1" * 64, "2" * 64)
    worlds = [
        GroundedReplayWorld(f"world-{seed}", _history(contract, seed, f"w{seed}"), contract=contract)
        for seed in (1, 2)
    ]
    pool = SupportAwareReplayPool(
        worlds,
        SupportPolicy(min_worlds=2, min_selected_nodes=2, min_replay_coverage=0.1),
    )
    active = DeclarativeTrainingAllocationPolicy(
        "active",
        deficit_weight=-0.05,
        transfer_weight=0.0,
        novelty_weight=0.0,
        uncertainty_weight=0.0,
        cost_weight=0.0,
        support_weight=0.0,
        staleness_weight=0.0,
        stop_threshold=-1.0,
        max_parallel=1,
    )
    controller = DreamRSIMetaController(
        active,
        pool,
        contract,
        qualification_gate=OnlineQualificationGate(contract, minimum_seeds=2),
    )
    proposal = controller.dream(max_rounds=1)
    if proposal.candidate_policy.policy_id == active.policy_id:
        raise SystemExit("smoke failed: replay did not find a candidate")
    results = []
    for seed in (11, 12):
        results.append(OnlineMetaPolicyResult(active.policy_id, seed, 0.5, f"{seed:064x}"[-64:], contract.sha256))
        results.append(OnlineMetaPolicyResult(proposal.candidate_policy.policy_id, seed, 0.6, f"{seed + 100:064x}"[-64:], contract.sha256))
    receipt = controller.validate_and_promote(results)
    if receipt.status != "PASS":
        raise SystemExit(f"smoke failed: {receipt.failures}")
    print(json.dumps({
        "status": "PASS",
        "proposal": proposal.status,
        "candidate": proposal.candidate_policy.policy_id,
        "replay_gain": proposal.replay_gain,
        "online_mean_gain": receipt.mean_paired_gain,
        "active_policy": controller.active_policy.policy_id,
    }, indent=2, sort_keys=True))


def dream_rsi_loop_smoke_main() -> None:
    """Execute one tiny real online allocation, append it to replay, then dream."""
    import tempfile
    from pathlib import Path

    from awa.v2.research_os.dream_rsi import ReplayObjective
    from awa.v2.research_os.dream_rsi_loop import (
        DreamRSIRecursiveExperiment,
        RealOnlineRoundConfig,
    )

    contract = DreamRSIContract("3" * 64, "4" * 64)
    bootstrap_decision = TrainingAllocationDecision(
        factor_signature=("navigation",),
        collector="coverage",
        transition_budget=16,
        parallel_worlds=1,
    )
    bootstrap_option = TrainingAllocationOption(
        bootstrap_decision,
        capability_deficit=0.5,
        expected_cost=0.1,
    )
    bootstrap_outcome = GroundedTrainingOutcome(
        outcome_id="bootstrap",
        parent_id=None,
        iteration=0,
        seed=0,
        option=bootstrap_option,
        metrics={"quality": 0.1, "transfer": 0.0, "adaptation": 0.0, "failure_rate": 0.0},
        compute_cost=0.0,
        latency_seconds=0.0,
        provenance_sha256="5" * 64,
        contract_sha256=contract.sha256,
        evidence=EvidenceClass.VALIDATED,
        offered_options=(bootstrap_option,),
    )
    pool = SupportAwareReplayPool(
        [GroundedReplayWorld("bootstrap-world", [bootstrap_outcome], contract=contract)],
        SupportPolicy(min_worlds=1, min_selected_nodes=1, min_replay_coverage=0.1),
    )
    active = DeclarativeTrainingAllocationPolicy(
        "active-loop",
        stop_threshold=-1.0,
        max_parallel=1,
        worker_budget=1,
    )
    controller = DreamRSIMetaController(
        active,
        pool,
        contract,
        qualification_gate=OnlineQualificationGate(contract, minimum_seeds=2),
    )
    experiment = DreamRSIRecursiveExperiment(controller)
    with tempfile.TemporaryDirectory(prefix="awa-v233-dream-loop-") as tmp:
        receipt = experiment.run_online_and_dream(
            (bootstrap_option,),
            Path(tmp),
            seed=7,
            config=RealOnlineRoundConfig(
                horizon=8,
                sequence_length=2,
                hidden=16,
                batch_size=8,
                world_epochs=1,
                actor_epochs=1,
            ),
            objective=ReplayObjective(mode="paper"),
            max_replay_rounds=1,
            revision_rounds=2,
        )
    print(json.dumps({
        "status": "PASS",
        "real_exact_transitions": receipt.online.exact_transitions,
        "replay_world_count": receipt.replay_world_count,
        "proposal_status": receipt.proposal.status,
        "revision_rounds": len(receipt.proposal.revision_history),
        "active_policy": controller.active_policy.policy_id,
    }, indent=2, sort_keys=True))


def _tree_hash(paths) -> str:
    import hashlib
    from pathlib import Path

    digest = hashlib.sha256()
    for base in sorted(Path(path) for path in paths):
        if base.is_file():
            files = [base]
            root = base.parent
        else:
            files = sorted(p for p in base.rglob("*.py") if p.is_file())
            root = base
        for path in files:
            rel = str(path.relative_to(root)).replace("\\", "/")
            digest.update(rel.encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _campaign_contract() -> DreamRSIContract:
    from pathlib import Path
    import awa.v2 as v2pkg

    root = Path(v2pkg.__file__).resolve().parent
    agent_paths = [
        root / "agent_runtime",
        root / "agent.py",
        root / "world.py",
        root / "temporal.py",
        root / "representation.py",
        root / "video_representation.py",
        root / "planners.py",
        root / "voc.py",
        root / "safety.py",
        root / "reasoning",
        root / "game" / "action_codec.py",
        root / "game" / "belief.py",
        root / "game" / "controller.py",
        root / "game" / "hybrid_control.py",
    ]
    evaluator_paths = [
        root / "research_os" / "dream_rsi.py",
        root / "research_os" / "dream_rsi_loop.py",
        root / "research_os" / "dream_rsi_campaign.py",
        root / "learning_system" / "meta_allocation.py",
        root / "learning_system" / "dream_execution.py",
    ]
    return DreamRSIContract(
        agent_sha256=_tree_hash([path for path in agent_paths if path.exists()]),
        evaluator_sha256=_tree_hash([path for path in evaluator_paths if path.exists()]),
        interface_version="awa-training-allocation-v2.34",
    )


def dream_rsi_campaign_main() -> None:
    import argparse
    from pathlib import Path

    from awa.v2.research_os.dream_rsi_campaign import (
        DreamRSICampaignConfig,
        FixedVsDreamCampaignRunner,
    )

    parser = argparse.ArgumentParser(description="Run the paired fixed-vs-DREAM-RSI empirical campaign.")
    parser.add_argument("--config", default="configs/v2_34_dream_rsi_empirical.yaml")
    parser.add_argument("--out-dir", default="runs/v2_34_dream_rsi")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    config = DreamRSICampaignConfig.from_yaml(args.config)
    contract = _campaign_contract()
    runner = FixedVsDreamCampaignRunner(config, contract, Path(args.out_dir))
    plan = runner.freeze_plan()
    if not args.execute:
        print(json.dumps({
            "status": "PLANNED",
            "config_sha256": config.sha256,
            "contract_sha256": contract.sha256,
            "plan_sha256": plan["sha256"],
            "paired_seeds": list(config.seeds),
            "rounds": config.rounds,
            "max_transition_budget_per_round": config.round_transition_budget,
        }, indent=2, sort_keys=True))
        return
    report = runner.run()
    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


def dream_rsi_campaign_smoke_main() -> None:
    import tempfile
    from pathlib import Path

    from awa.v2.research_os.dream_rsi_campaign import (
        DreamRSICampaignConfig,
        FixedVsDreamCampaignRunner,
        MenuOptionSpec,
    )
    from awa.v2.research_os.dream_rsi import ReplayObjective, SupportPolicy
    from awa.v2.research_os.dream_rsi_loop import RealOnlineRoundConfig

    initial = DeclarativeTrainingAllocationPolicy(
        "smoke-initial",
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
        transition_budget_cap=12,
    )
    config = DreamRSICampaignConfig(
        seeds=(31, 32),
        rounds=2,
        round_transition_budget=12,
        option_transition_budget=12,
        worker_budget=1,
        max_parallel=1,
        revision_rounds=2,
        replay_max_rounds=1,
        mutation_step=0.1,
        menu=(
            MenuOptionSpec(("navigation",), capability_deficit=0.2),
            MenuOptionSpec(("memory",), capability_deficit=0.8),
        ),
        initial_policy=initial,
        support_policy=SupportPolicy(min_worlds=1, min_selected_nodes=1, min_replay_coverage=0.1),
        objective=ReplayObjective(mode="paper", paper_beta1=0.0, paper_beta2=0.0),
        runtime=RealOnlineRoundConfig(
            horizon=6,
            device="cpu",
            sequence_length=2,
            hidden=16,
            world_epochs=1,
            actor_epochs=1,
            calibration_epochs=0,
            batch_size=8,
            one_step_aux_epochs=0,
        ),
        minimum_final_mean_gain=-1.0,
        minimum_curve_mean_gain=-1.0,
        max_seed_regression=1.0,
        bootstrap_samples=200,
    )
    with tempfile.TemporaryDirectory(prefix="awa-v234-dream-campaign-") as tmp:
        runner = FixedVsDreamCampaignRunner(config, _campaign_contract(), Path(tmp))
        report = runner.run()
        # A second run must reuse all committed rounds and reproduce the same report.
        report2 = runner.run()
        if report.to_dict() != report2.to_dict():
            raise SystemExit("campaign resume smoke failed: repeated report changed")
        print(json.dumps({
            "status": "PASS",
            "campaign_status": report.status,
            "paired_seeds": list(report.seeds),
            "completed_rounds": report.completed_rounds,
            "resume_reproducible": True,
        }, indent=2, sort_keys=True))
