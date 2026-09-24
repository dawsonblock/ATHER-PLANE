"""Aether v2 policy-first world-model stack.

v2.31 introduces three canonical lazy surfaces: ``agent_runtime``,
``learning_system``, and ``research_os``. Historical flat exports remain lazy
compatibility shims while the empirical baseline is frozen.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "2.38.6"

_LAYER_PACKAGES = {
    "agent_runtime": "awa.v2.agent_runtime",
    "learning_system": "awa.v2.learning_system",
    "research_os": "awa.v2.research_os",
}

_EXPORTS: dict[str, tuple[str, str]] = {
    "AetherV2Agent": ("awa.v2.agent", "AetherV2Agent"),
    "FactorizedEnvironmentFactory": ("awa.v2.curriculum.environment_factory", "FactorizedEnvironmentFactory"),
    "CompositionalBenchmark": ("awa.v2.curriculum.environment_factory", "CompositionalBenchmark"),
    "build_empirical_status": ("awa.v2.empirical_status", "build_empirical_status"),
    "RealViZDoomQualificationConfig": ("awa.v2.game.vizdoom_qualification", "RealViZDoomQualificationConfig"),
    "run_real_vizdoom_qualification": ("awa.v2.game.vizdoom_qualification", "run_real_vizdoom_qualification"),
    "PlannerHardwareBenchmarkConfig": ("awa.v2.planner_hardware_benchmark", "PlannerHardwareBenchmarkConfig"),
    "benchmark_planner_schedules": ("awa.v2.planner_hardware_benchmark", "benchmark_planner_schedules"),
    "PreflightRequirement": ("awa.v2.research_os.execution_preflight", "PreflightRequirement"),
    "run_execution_preflight": ("awa.v2.research_os.execution_preflight", "run_execution_preflight"),
    "build_real_vizdoom_campaign_plan": ("awa.v2.research_os.real_vizdoom_campaign", "build_real_vizdoom_campaign_plan"),
    "run_real_vizdoom_training_campaign": ("awa.v2.research_os.real_vizdoom_campaign", "run_real_vizdoom_training_campaign"),
    "MultimodalWorldModel": ("awa.v2.world", "MultimodalWorldModel"),
    "WorldModelEnsemble": ("awa.v2.ensemble", "WorldModelEnsemble"),
    "ModularWorldModel": ("awa.v2.modular_world", "ModularWorldModel"),
    "SparseMoETrunk": ("awa.v2.modular_world", "SparseMoETrunk"),
    "ComputeProfile": ("awa.v2.modular_world", "ComputeProfile"),
    "make_compute_matched_pair": ("awa.v2.modular_world", "make_compute_matched_pair"),
    "ControlledScalingRunner": ("awa.v2.scaling", "ControlledScalingRunner"),
    "ScalingPoint": ("awa.v2.scaling", "ScalingPoint"),
    "ScalingReport": ("awa.v2.scaling", "ScalingReport"),
    "run_controlled_scaling_experiment": ("awa.v2.scaling", "run_controlled_scaling_experiment"),
    "ExperimentLedger": ("awa.v2.experiment_ledger", "ExperimentLedger"),
    "GameTrainingCampaign": ("awa.v2.campaign", "GameTrainingCampaign"),
    "CampaignStage": ("awa.v2.campaign", "CampaignStage"),
    "ResourceProfile": ("awa.v2.campaign", "ResourceProfile"),
    "PromotionPolicy": ("awa.v2.campaign", "PromotionPolicy"),
    "campaign_plan": ("awa.v2.campaign", "campaign_plan"),
    "detect_resource_profile": ("awa.v2.campaign", "detect_resource_profile"),
    "estimate_campaign_compute": ("awa.v2.campaign", "estimate_campaign_compute"),
    "TelemetryRecorder": ("awa.v2.telemetry", "TelemetryRecorder"),
    "TelemetrySummary": ("awa.v2.telemetry", "TelemetrySummary"),
    "FailureEvidence": ("awa.v2.failure_triage", "FailureEvidence"),
    "FailureClass": ("awa.v2.failure_triage", "FailureClass"),
    "TriageResult": ("awa.v2.failure_triage", "TriageResult"),
    "triage_failure": ("awa.v2.failure_triage", "triage_failure"),
    "BenchmarkTrack": ("awa.v2.benchmark_tracks", "BenchmarkTrack"),
    "RepresentationContract": ("awa.v2.benchmark_tracks", "RepresentationContract"),
    "BenchmarkManifest": ("awa.v2.benchmark_tracks", "BenchmarkManifest"),
    "assert_representation_compatible": ("awa.v2.benchmark_tracks", "assert_representation_compatible"),
    "AblationSpec": ("awa.v2.ablation", "AblationSpec"),
    "DEFAULT_ABLATIONS": ("awa.v2.ablation", "DEFAULT_ABLATIONS"),
    "run_ablation_suite": ("awa.v2.ablation", "run_ablation_suite"),
    "EmpiricalAblationSpec": ("awa.v2.ablation", "EmpiricalAblationSpec"),
    "EMPIRICAL_ABLATIONS": ("awa.v2.ablation", "EMPIRICAL_ABLATIONS"),
    "build_empirical_ablation_protocol": ("awa.v2.ablation", "build_empirical_ablation_protocol"),
    "SystemVariant": ("awa.v2.game.variant_runtime", "SystemVariant"),
    "SYSTEM_VARIANTS": ("awa.v2.game.variant_runtime", "SYSTEM_VARIANTS"),
    "train_and_evaluate_variant": ("awa.v2.game.variant_runtime", "train_and_evaluate_variant"),
    "MilestoneRow": ("awa.v2.qualification_report", "MilestoneRow"),
    "build_milestone_report": ("awa.v2.qualification_report", "build_milestone_report"),
    "write_milestone_report": ("awa.v2.qualification_report", "write_milestone_report"),
    "ViZDoomScenario": ("awa.v2.game.vizdoom_env", "ViZDoomScenario"),
    "ViZDoomConfig": ("awa.v2.game.vizdoom_env", "ViZDoomConfig"),
    "ViZDoomAetherEnv": ("awa.v2.game.vizdoom_env", "ViZDoomAetherEnv"),
    "ViZDoomExplorerPolicy": ("awa.v2.game.vizdoom_dataset", "ViZDoomExplorerPolicy"),
    "collect_vizdoom_dataset": ("awa.v2.game.vizdoom_dataset", "collect_vizdoom_dataset"),
    "DEFAULT_VJEPA2_MODEL": ("awa.v2.game.vizdoom_vjepa", "DEFAULT_VJEPA2_MODEL"),
    "ViZDoomVJEPAConfig": ("awa.v2.game.vizdoom_vjepa", "ViZDoomVJEPAConfig"),
    "RunRecord": ("awa.v2.empirical_closure", "RunRecord"),
    "summarize_runs": ("awa.v2.empirical_closure", "summarize_runs"),
    "qualification_gates": ("awa.v2.empirical_closure", "qualification_gates"),
    "planner_dependence": ("awa.v2.empirical_closure", "planner_dependence"),
    "build_evidence_bundle": ("awa.v2.empirical_closure", "build_evidence_bundle"),
    "EvidenceClass": ("awa.v2.meta_exploration", "EvidenceClass"),
    "DiscoveryNode": ("awa.v2.meta_exploration", "DiscoveryNode"),
    "DiscoveryTree": ("awa.v2.meta_exploration", "DiscoveryTree"),
    "DeclarativeExplorationPolicy": ("awa.v2.meta_exploration", "DeclarativeExplorationPolicy"),
    "ReplayObjective": ("awa.v2.meta_exploration", "ReplayObjective"),
    "ReplayResult": ("awa.v2.meta_exploration", "ReplayResult"),
    "ReplayWorld": ("awa.v2.meta_exploration", "ReplayWorld"),
    "ReplaySimulatorPool": ("awa.v2.meta_exploration", "ReplaySimulatorPool"),
    "DeclarativePolicyMutator": ("awa.v2.meta_exploration", "DeclarativePolicyMutator"),
    "PolicyProposal": ("awa.v2.meta_exploration", "PolicyProposal"),
    "MetaPolicyOptimizer": ("awa.v2.meta_exploration", "MetaPolicyOptimizer"),
    "OnlinePolicyResult": ("awa.v2.meta_exploration", "OnlinePolicyResult"),
    "PromotionReceipt": ("awa.v2.meta_exploration", "PromotionReceipt"),
    "ExplorationPolicyPromotionGate": ("awa.v2.meta_exploration", "ExplorationPolicyPromotionGate"),
    "EvolvingExplorationController": ("awa.v2.meta_exploration", "EvolvingExplorationController"),
    "MetricProjection": ("awa.v2.evolving_campaign", "MetricProjection"),
    "CampaignCell": ("awa.v2.evolving_campaign", "CampaignCell"),
    "PairedCampaignPlan": ("awa.v2.evolving_campaign", "PairedCampaignPlan"),
    "CampaignEvidence": ("awa.v2.evolving_campaign", "CampaignEvidence"),
    "CampaignEvidenceStore": ("awa.v2.evolving_campaign", "CampaignEvidenceStore"),
    "CampaignRunner": ("awa.v2.evolving_campaign", "CampaignRunner"),
    "SubprocessCampaignRunner": ("awa.v2.evolving_campaign", "SubprocessCampaignRunner"),
    "DiscoveryTreeBuilder": ("awa.v2.evolving_campaign", "DiscoveryTreeBuilder"),
    "ValidationMatrix": ("awa.v2.evolving_campaign", "ValidationMatrix"),
    "ClosedLoopReceipt": ("awa.v2.evolving_campaign", "ClosedLoopReceipt"),
    "EvolvingCampaignOrchestrator": ("awa.v2.evolving_campaign", "EvolvingCampaignOrchestrator"),
    "load_history_evidence": ("awa.v2.evolving_campaign", "load_history_evidence"),
    "runner_from_config": ("awa.v2.evolving_campaign", "runner_from_config"),
    "orchestrator_from_config": ("awa.v2.evolving_campaign", "orchestrator_from_config"),
    "NativeProceduralRunnerConfig": ("awa.v2.native_campaign", "NativeProceduralRunnerConfig"),
    "ExplorationPolicyCurriculumAdapter": ("awa.v2.native_campaign", "ExplorationPolicyCurriculumAdapter"),
    "NativeProceduralCampaignRunner": ("awa.v2.native_campaign", "NativeProceduralCampaignRunner"),
    "native_runner_from_config": ("awa.v2.native_campaign", "native_runner_from_config"),
    "MilestonePromotionConfig": ("awa.v2.milestone_analysis", "MilestonePromotionConfig"),
    "MilestonePromotionReceipt": ("awa.v2.milestone_analysis", "MilestonePromotionReceipt"),
    "build_milestone_scorecard": ("awa.v2.milestone_analysis", "build_milestone_scorecard"),
    "evaluate_milestone_promotion": ("awa.v2.milestone_analysis", "evaluate_milestone_promotion"),
}

__all__ = list(_LAYER_PACKAGES) + list(_EXPORTS)


def __getattr__(name: str) -> Any:
    layer_module = _LAYER_PACKAGES.get(name)
    if layer_module is not None:
        value = import_module(layer_module)
        globals()[name] = value
        return value
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
