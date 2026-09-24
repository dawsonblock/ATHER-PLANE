from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.ablation_campaign import AblationCampaignConfig, build_campaign_plan
from awa.v2.research_os.progressive_executor import (
    build_next_phase_execution_plan,
    execute_phase_plan,
    load_executor_config,
)
from awa.v2.system_split import validate_split_packages
from awa.v2.research_os.planner_diagnostic import build_planner_diagnostic_report
from test_v2386_planner_diagnostic import _p1p_raw, _p1p_replay_fixture

ROOT = Path(__file__).resolve().parents[1]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    world = path.parent / "p1-world.pt"
    actor = path.parent / "p1-actor.pt"
    world_hash = hashlib.sha256(b"p1-world").hexdigest()
    actor_hash = hashlib.sha256(b"p1-actor").hexdigest()
    if path.name == "real_vizdoom_training_receipt.json" and payload.get("track") == "structured":
        stages = ["real-wiring-2k", "real-navigation-10k", "real-control-25k"]
        scenarios = ["basic", "my_way_home", "health_gathering"]
        count = 3 if payload["transitions"] >= 25000 else 2 if payload["transitions"] >= 10000 else 1
        rows = []
        for i in range(count):
            rows.append({"stage": stages[i], "promoted": True,
                         "stable": {"stage": stages[i], "world": str(world), "actor": str(actor),
                                    "world_sha256": world_hash, "actor_sha256": actor_hash},
                         "evaluation": {"actor_success_rate": 0.2, "planner_success_rate": 0.2},
                         "incumbent_evaluation": {"actor_success_rate": 0.1, "planner_success_rate": 0.1} if i else None,
                         "training": {"resumed": i > 0, "resume_world_sha256": world_hash if i else None,
                                      "resume_actor_sha256": actor_hash if i else None,
                                      "world_global_updates": i + 1},
                         "comparison_contract": {"scenario": scenarios[i], "track": "structured",
                                                 "incumbent_stage": stages[i-1] if i else None,
                                                 "incumbent_world_sha256": world_hash,
                                                 "incumbent_actor_sha256": actor_hash,
                                                 "seed_start": 90000, "eval_episodes": 4}})
        payload = {**payload, "stable_checkpoint": {"stage": stages[count-1]},
                   "result": {"results": rows}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    if path.name == "execution_preflight.json":
        world.write_bytes(b"p1-world")
        actor.write_bytes(b"p1-actor")
        # Compact synthetic gate receipts keep executor tests focused on ordering.
        from awa.v2.research_os.planner_diagnostic import build_reward_diagnostic_report
        training_seeds = [7101, 7102, 7103]
        evaluation_seeds = [9101, 9102, 9103, 9104, 9105, 9106]
        arms = {}
        for arm, transform in (("raw_reward", {"kind": "identity"}),
                               ("clipped_reward", {"kind": "clip", "min": -1.0, "max": 1.0})):
            seed_rows = []
            for seed in training_seeds:
                episodes = [{"seed": eval_seed, "success": arm == "clipped_reward" and i < 2,
                             "return": float(i), "action_entropy_nats": .5,
                             "action_distribution": {"forward": .5}}
                            for i, eval_seed in enumerate(evaluation_seeds)]
                seed_rows.append({"seed": seed, "model_initialization_seed": seed,
                                  "source_dataset_sha256": "d" * 64,
                                  "shared_nonreward_data_sha256": "e" * 64,
                                  "checkpoint_sha256": {"world": "f" * 64, "actor": "a" * 64},
                                  "training_report": {"world_global_updates": 20, "actor_steps": 20},
                                  "episodes": episodes})
            arms[arm] = {"reward_transform": transform, "per_training_seed": seed_rows}
        p1d_raw = {"format": "awa-v2.38.6-reward-diagnostic-raw-v1", "scenario": "my_way_home",
                   "backend": {"real_vizdoom": True}, "training_seeds": training_seeds,
                   "evaluation_seed_ids": evaluation_seeds,
                   "fixed_contract": {"scenario": "my_way_home", "track": "structured",
                                      "source_dataset_sha256": "d" * 64,
                                      "shared_nonreward_data_sha256": "e" * 64,
                                      "optimizer": "fixed", "checkpoint_selection": "final",
                                      "paired_model_initialization_seeds": training_seeds,
                                      "paired_evaluation_seed_ids": evaluation_seeds},
                   "reward_component_statistics": {key: {} for key in
                       ("terminal_success", "collision_damage", "living_step_cost", "progress_shaping", "timeout")},
                   "arms": arms}
        p1d = build_reward_diagnostic_report(p1d_raw)
        (path.parent / "reward_diagnostic.json").write_text(json.dumps(p1d), encoding="utf-8")
        p1p_raw = _p1p_raw()
        p1p_raw["checkpoint_sha256"] = {"world": world_hash, "actor": actor_hash}
        replay = _p1p_replay_fixture(p1p_raw["seed_ids"])
        replay_text = json.dumps(replay)
        (path.parent / "vizdoom_branch_replay.json").write_text(replay_text, encoding="utf-8")
        p1p_raw["branch_replay_report_sha256"] = hashlib.sha256(replay_text.encode()).hexdigest()
        (path.parent / "planner_diagnostic_raw.json").write_text(json.dumps(p1p_raw), encoding="utf-8")
        (path.parent / "planner_diagnostic.json").write_text(
            json.dumps(build_planner_diagnostic_report(p1p_raw)), encoding="utf-8")


def _agent_runtime_hash() -> str:
    root = ROOT / "src" / "awa" / "v2" / "agent_runtime"
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(str(path.relative_to(root)).replace("\\", "/").encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def test_v238_version_split_and_frozen_agent_runtime():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"
    assert validate_split_packages(ROOT / "src") == []
    assert _agent_runtime_hash() == "f1d9942bad5d9b45bc1e6f177465eec1914ab540823a5c97e7cfa3c18f78582b"


def test_executor_config_and_runpod_bootstrap_are_shipped():
    cfg = load_executor_config(ROOT / "configs/v2_38_progressive_executor.yaml")
    assert cfg["release"] == "2.38.6"
    assert Path(cfg["progressive_config"]).name == "v2_38_progressive_capability.yaml"
    assert (ROOT / "deploy/runpod/bootstrap.sh").exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "awa-v2-progressive-executor" in pyproject
    assert "awa-v2-runpod-bootstrap" in pyproject


def test_fresh_executor_plans_preflight_only(tmp_path: Path):
    plan = build_next_phase_execution_plan(
        executor_config=ROOT / "configs/v2_38_progressive_executor.yaml",
        evidence_root=tmp_path,
        repo_root=ROOT,
    )
    assert plan["status"] == "READY"
    assert plan["phase_id"] == "P0"
    assert len(plan["commands"]) == 1
    assert plan["commands"][0]["argv"][0] == "awa-v2-execution-preflight"
    assert plan["commands"][0]["argv"][2].endswith("v2_38_6_empirical_execution.yaml")


def test_initial_checkpoint_is_registered_as_baseline_not_promoted(tmp_path: Path):
    from awa.v2.research_os.progressive_capability import build_progressive_capability_report, load_progressive_protocol

    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    (tmp_path / "real_vizdoom_training_receipt.json").write_text(json.dumps({
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 2000,
        "checkpoint_registry_verified": True,
        "result": {"results": [{"stage": "real-wiring-2k", "promoted": False,
                                  "baseline_registered": True, "promotion_status": "baseline_registration",
                                  "stable": {"stage": "real-wiring-2k", "world": str(tmp_path / "p1-world.pt"),
                                             "actor": str(tmp_path / "p1-actor.pt"),
                                             "world_sha256": hashlib.sha256(b"p1-world").hexdigest(),
                                             "actor_sha256": hashlib.sha256(b"p1-actor").hexdigest()}}]},
    }), encoding="utf-8")
    report = build_progressive_capability_report(
        tmp_path, protocol=load_progressive_protocol(ROOT / "configs/v2_38_progressive_capability.yaml"))
    assert report["phases"][1]["status"] == "PASS"
    # The shared helper also emits synthetic passing P1D/P1P receipts; P2 is
    # therefore the first unresolved phase in this ordering regression.
    assert report["first_blocked_phase"] == "P2"


def test_budget_guard_blocks_large_next_stage(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 10000,
        "checkpoint_registry_verified": True,
    })
    plan = build_next_phase_execution_plan(
        executor_config=ROOT / "configs/v2_38_progressive_executor.yaml",
        evidence_root=tmp_path,
        repo_root=ROOT,
        max_transition_target=10000,
    )
    assert plan["phase_id"] == "P3"
    assert plan["status"] == "BUDGET_BLOCKED"
    assert plan["transition_target"] == 25000


def test_focused_ablation_configs_are_small_complete_pairwise_protocols():
    expected = {
        "v2_38_ablation_temporal_memory.yaml": ("actor_only", "belief_actor"),
        "v2_38_ablation_fixed_planner.yaml": ("world_actor", "world_planner"),
        "v2_38_ablation_adaptive_compute.yaml": ("world_planner", "adaptive_compute"),
    }
    for name, systems in expected.items():
        cfg = AblationCampaignConfig.from_yaml(ROOT / "configs" / name)
        protocol, plan = build_campaign_plan(cfg)
        assert cfg.systems == systems
        assert len(plan.jobs) == 10  # 2 systems x 5 seeds x one 25K milestone
        assert plan.expected_result_cells == 20  # two evaluation splits
        assert tuple(protocol.systems) == systems


def test_after_25k_next_plan_is_focused_temporal_memory_campaign(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 25000,
        "checkpoint_registry_verified": True,
    })
    plan = build_next_phase_execution_plan(
        executor_config=ROOT / "configs/v2_38_progressive_executor.yaml",
        evidence_root=tmp_path,
        repo_root=ROOT,
    )
    assert plan["phase_id"] == "P4"
    assert plan["status"] == "READY"
    assert [x["argv"][0] for x in plan["commands"]] == ["awa-v2-ablation-campaign", "awa-v2-ablation-report"]
    assert any(str(x).endswith("v2_38_ablation_temporal_memory.yaml") for x in plan["commands"][0]["argv"])


def test_horizon_gate_refuses_to_invent_measurement_curve(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 25000,
        "checkpoint_registry_verified": True,
    })
    _write(tmp_path / "ablation_report.json", {
        "status": "QUALIFIED",
        "marginal_components": [{"mechanism": "temporal_memory", "decision": "KEEP"}],
    })
    plan = build_next_phase_execution_plan(
        executor_config=ROOT / "configs/v2_38_progressive_executor.yaml",
        evidence_root=tmp_path,
        repo_root=ROOT,
    )
    assert plan["phase_id"] == "P5"
    assert plan["status"] == "MANUAL_REQUIRED"
    assert "world_model_horizon_curve.json" in plan["manual_requirement"]
    assert plan["commands"] == []


def test_horizon_curve_unlocks_gate_command(tmp_path: Path):
    _write(tmp_path / "execution_preflight.json", {"ready": True, "required_failures": []})
    _write(tmp_path / "real_vizdoom_training_receipt.json", {
        "track": "structured", "backend": {"real_vizdoom": True}, "transitions": 25000,
        "checkpoint_registry_verified": True,
    })
    _write(tmp_path / "ablation_report.json", {
        "status": "QUALIFIED",
        "marginal_components": [{"mechanism": "temporal_memory", "decision": "KEEP"}],
    })
    _write(tmp_path / "world_model_horizon_curve.json", {"points": []})
    plan = build_next_phase_execution_plan(
        executor_config=ROOT / "configs/v2_38_progressive_executor.yaml",
        evidence_root=tmp_path,
        repo_root=ROOT,
    )
    assert plan["status"] == "READY"
    assert plan["commands"][0]["argv"][0] == "awa-v2-world-model-horizon-gate"


def test_executor_never_uses_shell(monkeypatch, tmp_path: Path):
    seen = {}
    monkeypatch.setattr("awa.v2.research_os.progressive_executor.shutil.which", lambda _: "/usr/bin/fake")

    def fake_runner(argv, **kwargs):
        seen["argv"] = argv
        seen.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    plan = {
        "status": "READY", "phase_id": "P0", "plan_sha256": "abc",
        "requires_expensive_ack": False,
        "commands": [{"argv": ["awa-v2-execution-preflight", "--out-dir", str(tmp_path)], "description": "test"}],
    }
    result = execute_phase_plan(plan, runner=fake_runner)
    assert result["status"] == "PASS"
    assert seen["shell"] is False
    assert seen["argv"][0] == "/usr/bin/fake"


def test_expensive_campaign_requires_separate_ack(monkeypatch):
    monkeypatch.setattr("awa.v2.research_os.progressive_executor.shutil.which", lambda _: "/usr/bin/fake")
    plan = {
        "status": "READY", "phase_id": "P11", "plan_sha256": "abc",
        "requires_expensive_ack": True,
        "commands": [{"argv": ["awa-v2-dream-rsi-campaign", "--execute"], "description": "expensive"}],
    }
    with pytest.raises(PermissionError):
        execute_phase_plan(plan, runner=lambda *a, **k: SimpleNamespace(returncode=0, stdout="", stderr=""))


def test_v238_protocol_requires_full_matrix_before_scale():
    from awa.v2.research_os.progressive_capability import load_progressive_protocol
    protocol = load_progressive_protocol(ROOT / "configs/v2_38_progressive_capability.yaml")
    assert len(protocol["phases"]) == 18
    phase_ids = [row["id"] for row in protocol["phases"]]
    assert phase_ids.index("P1D") < phase_ids.index("P1P") < phase_ids.index("P2")
    assert protocol["phases"][-2]["id"] == "P14"
    assert protocol["phases"][-2]["kind"] == "full_ablation"
    assert protocol["phases"][-1]["id"] == "P15"
    cfg = load_executor_config(ROOT / "configs/v2_38_progressive_executor.yaml")
    assert set(cfg["limits"].require_expensive_ack_for) == {"P11", "P14"}
