"""Game-lab public surface with lazy feature-family loading."""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS: dict[str, tuple[str, str]] = {
    "ProceduralArenaEnv": ("awa.v2.game.procedural_arena", "ProceduralArenaEnv"),
    "ArenaTransitionInfo": ("awa.v2.game.procedural_arena", "ArenaTransitionInfo"),
    "OBS_DIM": ("awa.v2.game.procedural_arena", "OBS_DIM"),
    "ACTION_DIM": ("awa.v2.game.procedural_arena", "ACTION_DIM"),
    "GOAL_DIM": ("awa.v2.game.procedural_arena", "GOAL_DIM"),
    "GoalProgram": ("awa.v2.game.goal_program", "GoalProgram"),
    "ObjectiveCode": ("awa.v2.game.goal_program", "ObjectiveCode"),
    "normalize_objectives": ("awa.v2.game.goal_program", "normalize_objectives"),
    "LogicalArenaTeacher": ("awa.v2.game.teacher", "LogicalArenaTeacher"),
    "RandomArenaPolicy": ("awa.v2.game.teacher", "RandomArenaPolicy"),
    "CoverageArenaPolicy": ("awa.v2.game.collectors", "CoverageArenaPolicy"),
    "FrozenAetherArenaPolicy": ("awa.v2.game.collectors", "FrozenAetherArenaPolicy"),
    "collector_receipt": ("awa.v2.game.collectors", "collector_receipt"),
    "GameCollectionReport": ("awa.v2.game.dataset", "GameCollectionReport"),
    "collect_game_dataset": ("awa.v2.game.dataset", "collect_game_dataset"),
    "GameBeliefEncoder": ("awa.v2.game.belief", "GameBeliefEncoder"),
    "GameBeliefSequenceDataset": ("awa.v2.game.belief", "GameBeliefSequenceDataset"),
    "GameBeliefWorldTrainer": ("awa.v2.game.belief", "GameBeliefWorldTrainer"),
    "encode_game_transitions": ("awa.v2.game.belief", "encode_game_transitions"),
    "GameStackReport": ("awa.v2.game.training", "GameStackReport"),
    "train_game_stack": ("awa.v2.game.training", "train_game_stack"),
    "LoadedProceduralGameStack": ("awa.v2.game.procedural_runtime", "LoadedProceduralGameStack"),
    "ProceduralActorEvaluation": ("awa.v2.game.procedural_runtime", "ProceduralActorEvaluation"),
    "load_procedural_game_stack": ("awa.v2.game.procedural_runtime", "load_procedural_game_stack"),
    "evaluate_procedural_actor": ("awa.v2.game.procedural_runtime", "evaluate_procedural_actor"),
    "evaluate_game_policy_transfer": ("awa.v2.game.transfer", "evaluate_game_policy_transfer"),
    "AdaptationMode": ("awa.v2.game.transfer", "AdaptationMode"),
    "GameLoopReport": ("awa.v2.game.loop", "GameLoopReport"),
    "ReusableGameLoop": ("awa.v2.game.loop", "ReusableGameLoop"),
    "AdaptiveGamePolicy": ("awa.v2.game.controller", "AdaptiveGamePolicy"),
    "EffortRuntimeTrace": ("awa.v2.game.controller", "EffortRuntimeTrace"),
    "facts_from_env": ("awa.v2.game.controller", "facts_from_env"),
    "GameVOCReport": ("awa.v2.game.voc_training", "GameVOCReport"),
    "fit_game_voc": ("awa.v2.game.voc_training", "fit_game_voc"),
    "PrioritizedReplayExportReport": ("awa.v2.game.continual", "PrioritizedReplayExportReport"),
    "export_prioritized_game_replay": ("awa.v2.game.continual", "export_prioritized_game_replay"),
    "HybridGameAction": ("awa.v2.game.action_codec", "HybridGameAction"),
    "HybridGameActionCodec": ("awa.v2.game.action_codec", "HybridGameActionCodec"),
    "AggregationRound": ("awa.v2.game.aggregation", "AggregationRound"),
    "PlannerAssistedMixturePolicy": ("awa.v2.game.aggregation", "PlannerAssistedMixturePolicy"),
    "IterativeDataAggregator": ("awa.v2.game.aggregation", "IterativeDataAggregator"),
    "SeedMetric": ("awa.v2.game.qualification", "SeedMetric"),
    "MultiSeedGameReport": ("awa.v2.game.qualification", "MultiSeedGameReport"),
    "summarize_seed_metrics": ("awa.v2.game.qualification", "summarize_seed_metrics"),
    "train_bootstrap_world_ensemble": ("awa.v2.game.ensemble_training", "train_bootstrap_world_ensemble"),
    "BridgeCapabilities": ("awa.v2.game.bridge_protocol", "BridgeCapabilities"),
    "BridgeTransition": ("awa.v2.game.bridge_protocol", "BridgeTransition"),
    "JsonLineGameBridgeClient": ("awa.v2.game.bridge_protocol", "JsonLineGameBridgeClient"),
    "BridgeEnvironmentAdapter": ("awa.v2.game.bridge_protocol", "BridgeEnvironmentAdapter"),
    "BridgeSessionServer": ("awa.v2.game.bridge_protocol", "BridgeSessionServer"),
    "serve_tcp": ("awa.v2.game.bridge_protocol", "serve_tcp"),
    "ViZDoomScenario": ("awa.v2.game.vizdoom_env", "ViZDoomScenario"),
    "ViZDoomConfig": ("awa.v2.game.vizdoom_env", "ViZDoomConfig"),
    "ViZDoomAetherEnv": ("awa.v2.game.vizdoom_env", "ViZDoomAetherEnv"),
    "ViZDoomActionMapper": ("awa.v2.game.vizdoom_env", "ViZDoomActionMapper"),
    "VIZDOOM_ACTION_DIM": ("awa.v2.game.vizdoom_env", "VIZDOOM_ACTION_DIM"),
    "VIZDOOM_GOAL_DIM": ("awa.v2.game.vizdoom_env", "VIZDOOM_GOAL_DIM"),
    "VIZDOOM_TELEMETRY_DIM": ("awa.v2.game.vizdoom_env", "VIZDOOM_TELEMETRY_DIM"),
    "make_vizdoom_env": ("awa.v2.game.vizdoom_env", "make_vizdoom_env"),
    "ViZDoomCollectionReport": ("awa.v2.game.vizdoom_dataset", "ViZDoomCollectionReport"),
    "ViZDoomExplorerPolicy": ("awa.v2.game.vizdoom_dataset", "ViZDoomExplorerPolicy"),
    "collect_vizdoom_dataset": ("awa.v2.game.vizdoom_dataset", "collect_vizdoom_dataset"),
    "DEFAULT_VJEPA2_MODEL": ("awa.v2.game.vizdoom_vjepa", "DEFAULT_VJEPA2_MODEL"),
    "ViZDoomVJEPAConfig": ("awa.v2.game.vizdoom_vjepa", "ViZDoomVJEPAConfig"),
    "validate_vjepa_clip_contract": ("awa.v2.game.vizdoom_vjepa", "validate_vjepa_clip_contract"),
    "vizdoom_pixel_representation_contract": ("awa.v2.game.vizdoom_vjepa", "vizdoom_pixel_representation_contract"),
    "write_vizdoom_vjepa_manifest": ("awa.v2.game.vizdoom_vjepa", "write_vizdoom_vjepa_manifest"),
    "SystemVariant": ("awa.v2.game.variant_runtime", "SystemVariant"),
    "SYSTEM_VARIANTS": ("awa.v2.game.variant_runtime", "SYSTEM_VARIANTS"),
    "VariantTrainingReport": ("awa.v2.game.variant_runtime", "VariantTrainingReport"),
    "VariantEvaluationReport": ("awa.v2.game.variant_runtime", "VariantEvaluationReport"),
    "train_and_evaluate_variant": ("awa.v2.game.variant_runtime", "train_and_evaluate_variant"),
}

__all__ = list(_EXPORTS)


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
