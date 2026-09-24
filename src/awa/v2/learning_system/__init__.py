"""Stable Aether learning-system surface.

The learning system owns experience collection, datasets, training, grounded
replay, environment/task generation, and curriculum allocation.  It may consume
agent-runtime components, but action-time code must not depend on this package.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "CoverageArenaPolicy": ("awa.v2.game.collectors", "CoverageArenaPolicy"),
    "RandomArenaPolicy": ("awa.v2.game.teacher", "RandomArenaPolicy"),
    "FrozenAetherArenaPolicy": ("awa.v2.game.collectors", "FrozenAetherArenaPolicy"),
    "collect_game_dataset": ("awa.v2.game.dataset", "collect_game_dataset"),
    "train_game_stack": ("awa.v2.game.training", "train_game_stack"),
    "augment_game_hindsight": ("awa.v2.game.her", "augment_game_hindsight"),
    "FactorizedEnvironmentFactory": ("awa.v2.curriculum.environment_factory", "FactorizedEnvironmentFactory"),
    "CompositionalBenchmark": ("awa.v2.curriculum.environment_factory", "CompositionalBenchmark"),
    "ProceduralTaskFactory": ("awa.v2.curriculum.task_factory", "ProceduralTaskFactory"),
    "PrioritizedExperienceBuffer": ("awa.v2.replay.structural_priority", "PrioritizedExperienceBuffer"),
    "TrainingAllocationDecision": ("awa.v2.learning_system.meta_allocation", "TrainingAllocationDecision"),
    "TrainingAllocationOption": ("awa.v2.learning_system.meta_allocation", "TrainingAllocationOption"),
    "DeclarativeTrainingAllocationPolicy": ("awa.v2.learning_system.meta_allocation", "DeclarativeTrainingAllocationPolicy"),
    "DeclarativeTrainingPolicyMutator": ("awa.v2.learning_system.meta_allocation", "DeclarativeTrainingPolicyMutator"),
    "MaterializedTrainingAllocation": ("awa.v2.learning_system.meta_allocation", "MaterializedTrainingAllocation"),
    "materialize_training_allocation": ("awa.v2.learning_system.meta_allocation", "materialize_training_allocation"),
    "build_collector_from_decision": ("awa.v2.learning_system.meta_allocation", "build_collector_from_decision"),
    "ExactAllocationExecutionReceipt": ("awa.v2.learning_system.dream_execution", "ExactAllocationExecutionReceipt"),
    "collect_exact_transition_dataset": ("awa.v2.learning_system.dream_execution", "collect_exact_transition_dataset"),
    "execute_training_allocation": ("awa.v2.learning_system.dream_execution", "execute_training_allocation"),
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
