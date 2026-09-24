from __future__ import annotations

import json
from pathlib import Path

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.research_os.progressive_capability import (
    HorizonGateConfig,
    build_progressive_capability_report,
    load_progressive_protocol,
    qualify_world_model_horizon,
)
from awa.v2.system_split import validate_split_packages

ROOT = Path(__file__).resolve().parents[1]


def test_v237_version_and_split():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"
    assert validate_split_packages(ROOT / "src") == []


def test_horizon_gate_selects_longest_contiguous_reliable_prefix():
    points = [
        {"horizon": 1, "normalized_prediction_error": 0.05, "calibration_error": 0.03, "samples": 1000},
        {"horizon": 2, "normalized_prediction_error": 0.10, "calibration_error": 0.04, "samples": 1000},
        {"horizon": 4, "normalized_prediction_error": 0.18, "calibration_error": 0.08, "samples": 1000},
        {"horizon": 8, "normalized_prediction_error": 0.25, "calibration_error": 0.09, "samples": 1000},
        {"horizon": 16, "normalized_prediction_error": 0.10, "calibration_error": 0.05, "samples": 1000},
    ]
    report = qualify_world_model_horizon(points)
    assert report["qualified"] is True
    assert report["selected_horizon"] == 4
    # H16 individually meets thresholds but cannot bypass failed H8.
    by_h = {x["horizon"]: x for x in report["points"]}
    assert by_h[16]["meets_thresholds"] is True
    assert by_h[16]["accepted"] is False


def test_horizon_gate_fails_on_too_few_samples():
    cfg = HorizonGateConfig(minimum_samples=256, minimum_qualified_horizon=4)
    points = [
        {"horizon": 1, "normalized_prediction_error": 0.01, "calibration_error": 0.01, "samples": 20},
        {"horizon": 4, "normalized_prediction_error": 0.01, "calibration_error": 0.01, "samples": 20},
    ]
    report = qualify_world_model_horizon(points, cfg)
    assert report["qualified"] is False
    assert report["selected_horizon"] == 0


def test_empty_evidence_stops_at_preflight(tmp_path: Path):
    report = build_progressive_capability_report(tmp_path, protocol=load_progressive_protocol(ROOT / "configs/v2_37_progressive_capability.yaml"))
    assert report["completed_prefix"] == 0
    assert report["first_blocked_phase"] == "P0"
    assert report["next_action"]["name"] == "cuda_vizdoom_preflight"


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_progression_advances_through_real_structured_receipt(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(
        tmp_path / "real_vizdoom_training_receipt.json",
        {
            "track": "structured",
            "backend": {"real_vizdoom": True},
            "transitions": 25000,
            "checkpoint_registry_verified": True,
            "vjepa": {"real_backbone": False},
        },
    )
    report = build_progressive_capability_report(tmp_path, protocol=load_progressive_protocol(ROOT / "configs/v2_37_progressive_capability.yaml"))
    assert report["completed_prefix"] == 4
    assert report["first_blocked_phase"] == "P4"
    assert report["next_action"]["name"] == "temporal_memory_value"


def test_ablation_keep_unlocks_memory_but_horizon_is_separate(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 25000,
        "checkpoint_registry_verified": True,
    })
    _write(tmp_path / "ablation_report.json", {
        "status": "QUALIFIED",
        "marginal_components": [
            {"mechanism": "temporal_memory", "decision": "KEEP"},
            {"mechanism": "fixed_planner", "decision": "KEEP"},
            {"mechanism": "adaptive_compute", "decision": "KEEP"},
        ],
    })
    report = build_progressive_capability_report(tmp_path, protocol=load_progressive_protocol(ROOT / "configs/v2_37_progressive_capability.yaml"))
    assert report["completed_prefix"] == 5
    assert report["first_blocked_phase"] == "P5"
    assert report["next_action"]["name"] == "world_model_reliable_horizon"


def test_qualified_horizon_report_unlocks_next_planner_gate(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 25000,
        "checkpoint_registry_verified": True,
    })
    _write(tmp_path / "ablation_report.json", {
        "status": "QUALIFIED",
        "marginal_components": [{"mechanism": "temporal_memory", "decision": "KEEP"}],
    })
    _write(tmp_path / "world_model_horizon_gate.json", {"qualified": True, "selected_horizon": 4})
    report = build_progressive_capability_report(tmp_path, protocol=load_progressive_protocol(ROOT / "configs/v2_37_progressive_capability.yaml"))
    assert report["completed_prefix"] == 6
    assert report["first_blocked_phase"] == "P6"


def test_real_pixel_gate_requires_real_vjepa(tmp_path: Path):
    # Directly use a protocol with only the pixel gate so the test isolates it.
    protocol = {
        "format": "test", "release": "2.38.0", "gating": {}, "claim_boundary": [],
        "phases": [{"id": "PX", "name": "pixel", "kind": "real_training", "track": "pixel", "min_transitions": 5000, "require_vjepa": True}],
    }
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "pixel", "backend": {"real_vizdoom": True}, "transitions": 5000,
        "checkpoint_registry_verified": True, "vjepa": {"real_backbone": False},
    })
    report = build_progressive_capability_report(tmp_path, protocol=protocol)
    assert report["completed_prefix"] == 0
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "pixel", "backend": {"real_vizdoom": True}, "transitions": 5000,
        "checkpoint_registry_verified": True, "vjepa": {"real_backbone": True},
    })
    report = build_progressive_capability_report(tmp_path, protocol=protocol)
    assert report["completed_prefix"] == 1


def test_config_and_cli_entries_are_shipped():
    protocol = load_progressive_protocol(ROOT / "configs/v2_37_progressive_capability.yaml")
    assert len(protocol["phases"]) == 15
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "awa-v2-progressive-capability" in pyproject
    assert "awa-v2-world-model-horizon-gate" in pyproject


def test_manual_scale_gate_requires_explicit_authorization(tmp_path: Path):
    protocol = {
        "format": "test", "release": "2.38.0", "gating": {}, "claim_boundary": [],
        "phases": [{"id": "S", "name": "scale", "kind": "manual_scale_gate"}],
    }
    report = build_progressive_capability_report(tmp_path, protocol=protocol)
    assert report["completed_prefix"] == 0
    _write(tmp_path / "scale_authorization.json", {"authorized": True, "status": "AUTHORIZED"})
    report = build_progressive_capability_report(tmp_path, protocol=protocol)
    assert report["completed_prefix"] == 1
