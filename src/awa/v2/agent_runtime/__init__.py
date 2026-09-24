"""Stable Aether agent runtime surface.

This package contains only components that may participate in action selection at
inference time.  It intentionally excludes dataset collection, training,
curriculum, ablation, provenance, and release-governance code.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AetherV2Agent": ("awa.v2.agent", "AetherV2Agent"),
    "GameBeliefEncoder": ("awa.v2.game.belief", "GameBeliefEncoder"),
    "AdaptiveGamePolicy": ("awa.v2.game.controller", "AdaptiveGamePolicy"),
    "EffortRuntimeTrace": ("awa.v2.game.controller", "EffortRuntimeTrace"),
    "HybridGameAction": ("awa.v2.game.action_codec", "HybridGameAction"),
    "HybridGameActionCodec": ("awa.v2.game.action_codec", "HybridGameActionCodec"),
    "MultimodalWorldModel": ("awa.v2.world", "MultimodalWorldModel"),
    "RiskConstraintModel": ("awa.v2.world", "RiskConstraintModel"),
    "PolicySeededMPPI": ("awa.v2.planners", "PolicySeededMPPI"),
    "PolicySeededICEM": ("awa.v2.planners", "PolicySeededICEM"),
    "RiskAwarePolicySeededMPPI": ("awa.v2.planners", "RiskAwarePolicySeededMPPI"),
    "HybridRiskAwarePolicySeededMPPI": ("awa.v2.planners", "HybridRiskAwarePolicySeededMPPI"),
    "ValueOfComputation": ("awa.v2.voc", "ValueOfComputation"),
    "AdaptiveReasoningEffortController": ("awa.v2.reasoning.effort", "AdaptiveReasoningEffortController"),
    "EffortSignals": ("awa.v2.reasoning.effort", "EffortSignals"),
    "SafeActionGuard": ("awa.v2.safety", "SafeActionGuard"),
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
