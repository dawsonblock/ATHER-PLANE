from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
import yaml

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.game.vizdoom_qualification import inspect_vjepa_backbone
from awa.v2.planner_hardware_benchmark import (
    PlannerHardwareBenchmarkConfig,
    benchmark_planner_schedules,
)
from awa.v2.research_os.execution_preflight import PreflightRequirement, run_execution_preflight
from awa.v2.research_os.real_vizdoom_campaign import build_real_vizdoom_campaign_plan
from awa.v2.system_split import SystemLayer, classify_module, validate_split_packages
from awa.v2.video_representation import VJEPA2HFBackbone

ROOT = Path(__file__).resolve().parents[1]


class _Actor:
    def deterministic_action(self, belief):
        return torch.zeros(belief.shape[0], 1, device=belief.device)


class _World(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def imagine_step(self, belief, action, deterministic=False):
        nxt = belief + action
        reward = -(nxt - 0.25).square()
        return {
            "belief": nxt,
            "reward": reward,
            "value": reward,
            "continuation": torch.ones_like(reward) * 0.99,
            "risk": torch.zeros(belief.shape[0], device=belief.device),
        }


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=4, frames_per_clip=2, _commit_hash="fake")


class _FakeProcessor:
    pass


def test_v235_version_and_system_split():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"
    assert validate_split_packages(ROOT / "src") == []
    assert classify_module("awa.v2.research_os.execution_preflight") is SystemLayer.RESEARCH_OS
    assert classify_module("awa.v2.game.vizdoom_campaign") is SystemLayer.LEARNING_SYSTEM
    assert classify_module("awa.v2.game.vizdoom_qualification") is SystemLayer.RESEARCH_OS


def test_preflight_can_pass_without_optional_hardware_requirements(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_execution_preflight(
        tmp_path,
        PreflightRequirement(
            require_cuda=False,
            min_cuda_vram_gib=0.0,
            min_disk_free_gib=0.0,
            min_system_ram_gib=0.0,
            require_vizdoom=False,
            require_transformers=False,
            require_vjepa_cache=False,
        ),
    )
    assert report["ready"] is True
    assert report["required_failures"] == []
    assert len(report["environment_fingerprint_sha256"]) == 64


def test_preflight_blocks_installed_vizdoom_when_local_ipc_is_denied(tmp_path: Path, monkeypatch):
    import awa.v2.research_os.execution_preflight as preflight_module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(preflight_module, "_package_version", lambda name: "1.3.1" if name == "vizdoom" else None)
    monkeypatch.setattr(preflight_module, "_vizdoom_ipc_probe", lambda: (False, "operation not permitted"))
    report = run_execution_preflight(
        tmp_path,
        PreflightRequirement(require_cuda=False, min_cuda_vram_gib=0, min_disk_free_gib=0,
                             require_vizdoom=True, require_transformers=False),
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["vizdoom"]["status"] == "FAIL"
    assert "vizdoom" in report["required_failures"]
    assert checks["vizdoom"]["details"]["local_ipc_socket_error"] == "operation not permitted"


def test_runpod_preflight_uses_provisioned_volume_quota_not_backing_disk_free(tmp_path: Path, monkeypatch):
    import awa.v2.research_os.execution_preflight as preflight_module

    volume = tmp_path / "workspace-volume"
    volume.mkdir()
    monkeypatch.setenv("AWA_TEST_VOLUME_SIZE_GB", "100")
    monkeypatch.setenv("AWA_TEST_VOLUME_ID", "network-volume-test")
    monkeypatch.setattr(preflight_module.os.path, "ismount", lambda path: Path(path) == volume.resolve())
    monkeypatch.setattr(preflight_module.shutil, "disk_usage", lambda _: (_ for _ in ()).throw(AssertionError("statvfs must not be used")))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_execution_preflight(
        volume / "runs",
        PreflightRequirement(require_cuda=False, min_cuda_vram_gib=0.0, min_disk_free_gib=80.0,
                             require_vizdoom=False, require_transformers=False,
                             persistent_volume_path=str(volume),
                             persistent_volume_size_env="AWA_TEST_VOLUME_SIZE_GB",
                             persistent_volume_id_env="AWA_TEST_VOLUME_ID",
                             require_persistent_volume_quota=True),
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["disk"]["status"] == "PASS"
    assert checks["disk"]["details"]["provisioned_quota_gib"] == 100.0


def test_runpod_preflight_fails_closed_without_volume_quota_provenance(tmp_path: Path, monkeypatch):
    import awa.v2.research_os.execution_preflight as preflight_module

    volume = tmp_path / "workspace-volume"
    volume.mkdir()
    monkeypatch.delenv("AWA_TEST_VOLUME_SIZE_GB", raising=False)
    monkeypatch.setenv("AWA_TEST_VOLUME_ID", "network-volume-test")
    monkeypatch.setattr(preflight_module.os.path, "ismount", lambda path: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_execution_preflight(
        volume / "runs",
        PreflightRequirement(require_cuda=False, min_cuda_vram_gib=0.0, min_disk_free_gib=0.0,
                             require_vizdoom=False, require_transformers=False,
                             persistent_volume_path=str(volume),
                             persistent_volume_size_env="AWA_TEST_VOLUME_SIZE_GB",
                             persistent_volume_id_env="AWA_TEST_VOLUME_ID",
                             require_persistent_volume_quota=True),
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["disk"]["status"] == "FAIL"
    assert "disk" in report["required_failures"]


def test_preflight_fails_closed_when_required_cuda_is_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_execution_preflight(
        tmp_path,
        PreflightRequirement(
            require_cuda=True,
            min_cuda_vram_gib=1.0,
            min_disk_free_gib=0.0,
            require_vizdoom=False,
        ),
    )
    assert report["ready"] is False
    assert "cuda" in report["required_failures"]


def test_preflight_fails_closed_when_host_ram_is_below_campaign_floor(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("awa.v2.research_os.execution_preflight._system_ram_bytes", lambda: 8 * 1024**3)
    report = run_execution_preflight(
        tmp_path,
        PreflightRequirement(
            require_cuda=False, min_cuda_vram_gib=0.0, min_disk_free_gib=0.0,
            min_system_ram_gib=32.0, require_vizdoom=False, require_transformers=False,
        ),
    )
    assert "system_ram" in report["required_failures"]


def test_preflight_detects_local_vjepa_cache(tmp_path: Path, monkeypatch):
    hf_home = tmp_path / "hf"
    model_dir = hf_home / "hub" / "models--facebook--vjepa2-vitl-fpc64-256"
    model_dir.mkdir(parents=True)
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    report = run_execution_preflight(
        tmp_path / "out",
        PreflightRequirement(
            require_cuda=False,
            min_cuda_vram_gib=0.0,
            min_disk_free_gib=0.0,
            require_vizdoom=False,
            require_transformers=False,
            require_vjepa_cache=True,
        ),
    )
    checks = {row["name"]: row for row in report["checks"]}
    assert checks["vjepa_cache"]["status"] == "PASS"
    assert checks["vjepa_cache"]["details"]["cached_paths"]


def test_real_vizdoom_campaign_plans_are_bound_and_nonempty(tmp_path: Path):
    structured = build_real_vizdoom_campaign_plan(
        ROOT / "configs" / "v2_35_vizdoom_structured.yaml",
        tmp_path / "structured",
    )
    pixel = build_real_vizdoom_campaign_plan(
        ROOT / "configs" / "v2_35_vizdoom_pixel.yaml",
        tmp_path / "pixel",
    )
    assert structured["track"] == "structured"
    assert structured["requires_real_vizdoom"] is True
    assert structured["requires_real_vjepa"] is False
    assert pixel["track"] == "pixel"
    assert pixel["requires_real_vjepa"] is True
    assert structured["stages"][-1]["target_transitions"] == 25000
    assert pixel["stages"][-1]["target_transitions"] == 5000
    assert len(structured["config_sha256"]) == 64


def test_vjepa_inspector_rejects_injected_components():
    backbone = VJEPA2HFBackbone("fake", model=_FakeModel(), processor=_FakeProcessor())
    provenance = inspect_vjepa_backbone(backbone)
    assert provenance["injected_components"] is True
    assert provenance["real_backbone"] is False


def test_hardware_benchmark_reports_numerical_schedule_equivalence_and_throughput():
    cfg = PlannerHardwareBenchmarkConfig(
        candidates=8,
        horizon=3,
        batch_limits=(0, 2, 1),
        warmup=0,
        repeats=2,
        seed=2350,
    )
    report = benchmark_planner_schedules(
        _World(),
        _Actor(),
        torch.zeros(1, 1),
        action_low=[-1.0],
        action_high=[1.0],
        config=cfg,
        require_cuda=False,
    )
    assert report["logical_work_matched"] is True
    assert report["action_schedule_equivalent"] is True
    assert report["max_action_abs_deviation"] <= 1e-6
    for row in report["rows"]:
        assert row["logical_transitions_per_second"] > 0.0
        assert row["physical_forwards_per_second"] > 0.0
        assert row["action_numerically_equivalent"] is True


def test_execution_config_references_existing_qualification_configs():
    cfg = yaml.safe_load((ROOT / "configs" / "v2_35_empirical_execution.yaml").read_text())
    targets = cfg["qualification_targets"]
    assert (ROOT / targets["structured_vizdoom_config"]).exists()
    assert (ROOT / targets["pixel_vjepa_config"]).exists()
    assert (ROOT / targets["dream_rsi_config"]).exists()
    assert targets["planner_batch_limits"] == [0, 32, 8, 4, 1]


def test_empirical_status_reads_real_training_receipt(tmp_path: Path):
    from awa.v2.empirical_status import build_empirical_status

    receipt_dir = tmp_path / "real"
    receipt_dir.mkdir()
    (receipt_dir / "real_vizdoom_training_receipt.json").write_text(
        __import__("json").dumps(
            {
                "format": "awa-v2.35-real-vizdoom-training-campaign-v1",
                "track": "pixel",
                "backend": {"real_vizdoom": True},
                "vjepa": {"real_backbone": True},
                "transitions": 1000,
            }
        ),
        encoding="utf-8",
    )
    status = build_empirical_status(tmp_path)
    rows = {row["tier"]: row for row in status["tiers"]}
    assert rows["T3"]["status"] == "HARDWARE_VALIDATED"
    assert rows["T4"]["status"] == "HARDWARE_VALIDATED"
