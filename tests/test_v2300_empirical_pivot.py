from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.curriculum.environment_factory import FactorizedEnvironmentFactory
from awa.v2.empirical_status import build_empirical_status, empirical_status_markdown
from awa.v2.game.vizdoom_env import ViZDoomAetherEnv, ViZDoomConfig
from awa.v2.game.vizdoom_qualification import assert_real_backend, inspect_backend
from awa.v2.planner_hardware_benchmark import (
    PlannerHardwareBenchmarkConfig,
    benchmark_planner_schedules,
)
from awa.v2.video_representation import VJEPA2HFBackbone


class _ZeroActor:
    def deterministic_action(self, belief):
        return torch.zeros(belief.shape[0], 1, device=belief.device)


class _World(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def imagine_step(self, belief, action, deterministic=False):
        nxt = belief + action
        reward = -(nxt - 0.5).square()
        return {
            "belief": nxt,
            "reward": reward,
            "value": reward,
            "continuation": torch.ones_like(reward) * 0.99,
            "risk": torch.zeros(belief.shape[0], device=belief.device),
        }


class _FakeDoomGame:
    variable_enums = {}

    def get_available_buttons(self):
        return []

    def close(self):
        pass


class _FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.config = SimpleNamespace(hidden_size=4, frames_per_clip=2, _commit_hash="fake")

    def get_vision_features(self, pixel_values_videos=None, **kwargs):
        b = int(pixel_values_videos.shape[0])
        return torch.ones(b, 2, 4)


class _FakeProcessor:
    def __call__(self, videos=None, return_tensors="pt", **kwargs):
        return {"pixel_values_videos": torch.as_tensor(videos).float()}


def test_v230_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_factorized_benchmark_is_valid_and_has_no_exact_signature_leakage():
    benchmark = FactorizedEnvironmentFactory(230).build_canonical_benchmark(
        train_replicates=1,
        eval_replicates=1,
    )
    payload = benchmark.to_dict()
    assert payload["all_tasks_valid"]
    assert not payload["exact_signature_leakage"]
    assert len(benchmark.train) == 10
    assert len(benchmark.heldout) == 3
    assert len(benchmark.transfer) == 3
    assert len(benchmark.sha256) == 64


def test_factorized_benchmark_is_deterministic():
    a = FactorizedEnvironmentFactory(231).build_canonical_benchmark( train_replicates=1, eval_replicates=1)
    b = FactorizedEnvironmentFactory(231).build_canonical_benchmark( train_replicates=1, eval_replicates=1)
    assert a.sha256 == b.sha256


def test_empirical_status_does_not_claim_unexecuted_real_tiers():
    payload = build_empirical_status()
    by_tier = {row["tier"]: row for row in payload["tiers"]}
    assert by_tier["T1"]["status"] == "SOFTWARE_VALIDATED"
    assert by_tier["T2"]["status"] == "UNEXECUTED"
    assert by_tier["T3"]["status"] == "UNEXECUTED"
    assert by_tier["T4"]["status"] == "UNEXECUTED"
    assert "unit test" in empirical_status_markdown(payload).lower()


def test_empirical_status_promotes_only_from_grounded_receipt(tmp_path: Path):
    q = tmp_path / "q"
    q.mkdir()
    receipt = {
        "format": "awa-v2.30-real-vizdoom-qualification-v1",
        "track": "pixel",
        "backend": {"real_vizdoom": True},
        "collection": {"transitions": 12},
        "vjepa": {"real_backbone": True},
    }
    (q / "real_vizdoom_qualification.json").write_text(json.dumps(receipt))
    payload = build_empirical_status(tmp_path)
    by_tier = {row["tier"]: row for row in payload["tiers"]}
    assert by_tier["T3"]["status"] == "HARDWARE_VALIDATED"
    assert by_tier["T4"]["status"] == "HARDWARE_VALIDATED"


def test_real_vizdoom_qualification_rejects_injected_fake():
    env = ViZDoomAetherEnv(ViZDoomConfig(track="structured"), game=_FakeDoomGame())
    try:
        provenance = inspect_backend(env)
        assert not provenance["real_vizdoom"]
        try:
            assert_real_backend(env)
        except RuntimeError as exc:
            assert "refuses injected/fake" in str(exc)
        else:
            raise AssertionError("fake backend should be rejected")
    finally:
        env.close()


def test_vjepa_backbone_marks_injected_test_components():
    backbone = VJEPA2HFBackbone("fake", model=_FakeModel(), processor=_FakeProcessor())
    assert backbone._awa_injected_components is True


def test_hardware_schedule_benchmark_matches_logical_work_on_cpu():
    world = _World()
    belief = torch.zeros(1, 1)
    cfg = PlannerHardwareBenchmarkConfig(
        candidates=8,
        horizon=3,
        batch_limits=(0, 2, 1),
        warmup=0,
        repeats=2,
        seed=11,
    )
    report = benchmark_planner_schedules(
        world,
        _ZeroActor(),
        belief,
        action_low=[-1.0],
        action_high=[1.0],
        config=cfg,
        require_cuda=False,
    )
    assert report["logical_work_matched"]
    by = {row["planner_world_batch_size"]: row for row in report["rows"]}
    assert by[0]["mean_physical_world_model_forwards"] < by[2]["mean_physical_world_model_forwards"]
    assert by[2]["mean_physical_world_model_forwards"] < by[1]["mean_physical_world_model_forwards"]
    assert report["accelerator_energy_joules"] is None
