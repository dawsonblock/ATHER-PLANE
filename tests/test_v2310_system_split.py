from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import yaml

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.system_split import SystemLayer, classify_module, split_manifest, validate_split_packages

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_v231_versions_and_canonical_split_config():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"
    cfg = yaml.safe_load((ROOT / "configs" / "v2_31_system_split.yaml").read_text())
    assert cfg["release"] == "2.31.0"
    split = cfg["system_split"]
    assert split["canonical_entrypoints"] == {
        "agent_runtime": "awa.v2.agent_runtime",
        "learning_system": "awa.v2.learning_system",
        "research_os": "awa.v2.research_os",
    }
    assert cfg["agent_runtime"]["forbidden_responsibilities"]
    assert cfg["learning_system"]["forbidden_responsibilities"]
    assert cfg["research_os"]["may_not_change_agent_behavior_during_evaluation"] is True


def test_split_dependency_direction_is_fail_closed():
    assert classify_module("awa.v2.game.controller") is SystemLayer.AGENT_RUNTIME
    assert classify_module("awa.v2.game.training") is SystemLayer.LEARNING_SYSTEM
    assert classify_module("awa.v2.ablation_campaign") is SystemLayer.RESEARCH_OS
    assert classify_module("awa.v2.skills.registry") is SystemLayer.EXPERIMENTAL
    assert validate_split_packages(SRC) == []
    manifest = split_manifest()
    assert manifest["format"] == "awa-v2.31-system-split-v1"
    assert "research_os" not in manifest["layers"]["agent_runtime"]["may_depend_on"]
    assert "learning_system" not in manifest["layers"]["agent_runtime"]["may_depend_on"]


def test_agent_runtime_public_surface_does_not_expose_training_or_governance():
    ar = importlib.import_module("awa.v2.agent_runtime")
    forbidden = {
        "AblationCampaignRunner",
        "ExperimentProtocol",
        "PhysicalComputeLedger",
        "train_game_stack",
        "collect_game_dataset",
        "FactorizedEnvironmentFactory",
    }
    assert forbidden.isdisjoint(set(ar.__all__))
    assert "AdaptiveGamePolicy" in ar.__all__
    assert "MultimodalWorldModel" in ar.__all__
    assert "ValueOfComputation" in ar.__all__
    assert "SafeActionGuard" in ar.__all__


def test_learning_and_research_surfaces_are_separate():
    learning = importlib.import_module("awa.v2.learning_system")
    research = importlib.import_module("awa.v2.research_os")
    assert "train_game_stack" in learning.__all__
    assert "collect_game_dataset" in learning.__all__
    assert "AblationCampaignRunner" not in learning.__all__
    assert "AblationCampaignRunner" in research.__all__
    assert "ExperimentProtocol" in research.__all__
    assert "AdaptiveGamePolicy" not in research.__all__


def test_split_check_cli_reports_zero_violations():
    proc = subprocess.run(
        [sys.executable, "-m", "awa.v2.split_cli_runner"],
        cwd=ROOT,
        env={**__import__("os").environ, "PYTHONPATH": str(SRC)},
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["status"] == "ok"
    assert payload["violations"] == []
