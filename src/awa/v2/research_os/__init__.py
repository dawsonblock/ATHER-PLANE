"""Aether research operating-system surface.

This layer owns preregistration, provenance, resumable experiment execution,
compute accounting, empirical qualification, and keep/remove decisions.  It is
not imported by the action-time agent runtime.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AblationCampaignConfig": ("awa.v2.ablation_campaign", "AblationCampaignConfig"),
    "AblationCampaignRunner": ("awa.v2.ablation_campaign", "AblationCampaignRunner"),
    "ExperimentProtocol": ("awa.v2.experiment_protocol", "ExperimentProtocol"),
    "PhysicalComputeLedger": ("awa.v2.compute_ledger", "PhysicalComputeLedger"),
    "PlannerExecutionFingerprint": ("awa.v2.compute_ledger", "PlannerExecutionFingerprint"),
    "build_empirical_status": ("awa.v2.empirical_status", "build_empirical_status"),
    "MilestonePromotionConfig": ("awa.v2.milestone_analysis", "MilestonePromotionConfig"),
    "MilestonePromotionReceipt": ("awa.v2.milestone_analysis", "MilestonePromotionReceipt"),
    "build_milestone_scorecard": ("awa.v2.milestone_analysis", "build_milestone_scorecard"),
    "evaluate_milestone_promotion": ("awa.v2.milestone_analysis", "evaluate_milestone_promotion"),
    "DreamRSIContract": ("awa.v2.research_os.dream_rsi", "DreamRSIContract"),
    "GroundedTrainingOutcome": ("awa.v2.research_os.dream_rsi", "GroundedTrainingOutcome"),
    "GroundedReplayWorld": ("awa.v2.research_os.dream_rsi", "GroundedReplayWorld"),
    "SupportPolicy": ("awa.v2.research_os.dream_rsi", "SupportPolicy"),
    "SupportStatus": ("awa.v2.research_os.dream_rsi", "SupportStatus"),
    "SupportAwareReplayPool": ("awa.v2.research_os.dream_rsi", "SupportAwareReplayPool"),
    "DreamRSIOptimizer": ("awa.v2.research_os.dream_rsi", "DreamRSIOptimizer"),
    "DreamRSIMetaController": ("awa.v2.research_os.dream_rsi", "DreamRSIMetaController"),
    "OnlineMetaPolicyResult": ("awa.v2.research_os.dream_rsi", "OnlineMetaPolicyResult"),
    "OnlineQualificationGate": ("awa.v2.research_os.dream_rsi", "OnlineQualificationGate"),
    "ReplayObjective": ("awa.v2.research_os.dream_rsi", "ReplayObjective"),
    "CandidateReplayEvaluation": ("awa.v2.research_os.dream_rsi", "CandidateReplayEvaluation"),
    "PolicyRevisionReceipt": ("awa.v2.research_os.dream_rsi", "PolicyRevisionReceipt"),
    "ground_execution_receipt": ("awa.v2.research_os.dream_rsi", "ground_execution_receipt"),
    "RealOnlineRoundConfig": ("awa.v2.research_os.dream_rsi_loop", "RealOnlineRoundConfig"),
    "RealOnlineRoundReceipt": ("awa.v2.research_os.dream_rsi_loop", "RealOnlineRoundReceipt"),
    "RecursiveRoundReceipt": ("awa.v2.research_os.dream_rsi_loop", "RecursiveRoundReceipt"),
    "FixedVsDreamComparisonPlan": ("awa.v2.research_os.dream_rsi_loop", "FixedVsDreamComparisonPlan"),
    "execute_policy_online_world": ("awa.v2.research_os.dream_rsi_loop", "execute_policy_online_world"),
    "DreamRSIRecursiveExperiment": ("awa.v2.research_os.dream_rsi_loop", "DreamRSIRecursiveExperiment"),
    "OnlineDecisionArtifact": ("awa.v2.research_os.dream_rsi_loop", "OnlineDecisionArtifact"),
    "load_online_world": ("awa.v2.research_os.dream_rsi_loop", "load_online_world"),
    "MenuOptionSpec": ("awa.v2.research_os.dream_rsi_campaign", "MenuOptionSpec"),
    "DreamRSICampaignConfig": ("awa.v2.research_os.dream_rsi_campaign", "DreamRSICampaignConfig"),
    "ArmRoundRecord": ("awa.v2.research_os.dream_rsi_campaign", "ArmRoundRecord"),
    "SeedComparisonSummary": ("awa.v2.research_os.dream_rsi_campaign", "SeedComparisonSummary"),
    "FixedVsDreamCampaignReport": ("awa.v2.research_os.dream_rsi_campaign", "FixedVsDreamCampaignReport"),
    "FixedVsDreamCampaignRunner": ("awa.v2.research_os.dream_rsi_campaign", "FixedVsDreamCampaignRunner"),
    "build_campaign_report": ("awa.v2.research_os.dream_rsi_campaign", "build_campaign_report"),
    "PreflightRequirement": ("awa.v2.research_os.execution_preflight", "PreflightRequirement"),
    "PreflightCheck": ("awa.v2.research_os.execution_preflight", "PreflightCheck"),
    "run_execution_preflight": ("awa.v2.research_os.execution_preflight", "run_execution_preflight"),
    "write_execution_preflight": ("awa.v2.research_os.execution_preflight", "write_execution_preflight"),
    "build_real_vizdoom_campaign_plan": ("awa.v2.research_os.real_vizdoom_campaign", "build_real_vizdoom_campaign_plan"),
    "run_real_vizdoom_training_campaign": ("awa.v2.research_os.real_vizdoom_campaign", "run_real_vizdoom_training_campaign"),
    "EvidenceBundleConfig": ("awa.v2.research_os.real_campaign_evidence", "EvidenceBundleConfig"),
    "build_real_execution_evidence_bundle": ("awa.v2.research_os.real_campaign_evidence", "build_real_execution_evidence_bundle"),
    "write_real_execution_evidence_bundle": ("awa.v2.research_os.real_campaign_evidence", "write_real_execution_evidence_bundle"),
    "BaselineResultRecord": ("awa.v2.research_os.external_baseline", "BaselineResultRecord"),
    "MatchedBaselineProtocol": ("awa.v2.research_os.external_baseline", "MatchedBaselineProtocol"),
    "compare_external_baseline": ("awa.v2.research_os.external_baseline", "compare_external_baseline"),
    "HorizonPoint": ("awa.v2.research_os.progressive_capability", "HorizonPoint"),
    "HorizonGateConfig": ("awa.v2.research_os.progressive_capability", "HorizonGateConfig"),
    "qualify_world_model_horizon": ("awa.v2.research_os.progressive_capability", "qualify_world_model_horizon"),
    "load_progressive_protocol": ("awa.v2.research_os.progressive_capability", "load_progressive_protocol"),
    "build_progressive_capability_report": ("awa.v2.research_os.progressive_capability", "build_progressive_capability_report"),
    "write_progressive_capability_report": ("awa.v2.research_os.progressive_capability", "write_progressive_capability_report"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attr_name = target
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
