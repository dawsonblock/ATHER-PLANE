"""Regressions for the actual v2.38.0 P2 receipt and the corrected qualification path."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from awa.v2.campaign import PromotionPolicy
from awa.v2.game.vizdoom_campaign import ViZDoomCampaignStage, ViZDoomTrainingCampaign
from awa.v2.research_os.planner_diagnostic import build_planner_diagnostic_report, build_reward_diagnostic_report
from awa.v2.research_os.progressive_capability import build_progressive_capability_report, load_progressive_protocol
from awa.v2.research_os.progressive_executor import (
    _planner_checkpoints, build_next_phase_execution_plan, execute_next_progressive_phase,
)
from test_v2386_planner_diagnostic import _p1p_raw, _p1p_replay_fixture


ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = load_progressive_protocol(ROOT / "configs/v2_38_progressive_capability.yaml")
EXECUTOR = ROOT / "configs/v2_38_progressive_executor.yaml"


def _receipt(root, *, stage="real-navigation-10k", promoted=True, success=0.2, paired=True):
    root.mkdir(parents=True, exist_ok=True)
    world = root / "p1-world.pt"; actor = root / "p1-actor.pt"
    world.write_bytes(b"p1-world"); actor.write_bytes(b"p1-actor")
    world_hash = hashlib.sha256(b"p1-world").hexdigest()
    actor_hash = hashlib.sha256(b"p1-actor").hexdigest()
    rows = [{"stage": "real-wiring-2k", "promoted": True,
             "stable": {"stage": "real-wiring-2k", "world": str(world), "actor": str(actor),
                        "world_sha256": world_hash, "actor_sha256": actor_hash},
             "training": {"world_global_updates": 1}}]
    if stage != "real-wiring-2k":
        row = {"stage": stage, "promoted": promoted,
               "stable": {"stage": stage} if promoted else None,
               "training": {"resumed": True, "resume_world_sha256": world_hash,
                            "resume_actor_sha256": actor_hash, "world_global_updates": 2},
               "evaluation": {"actor_success_rate": success, "planner_success_rate": 0}}
        if paired:
            row["comparison_contract"] = {"scenario": "health_gathering" if stage == "real-control-25k" else "my_way_home",
                                          "track": "structured", "incumbent_stage": "real-wiring-2k" if stage != "real-control-25k" else "real-navigation-10k",
                                          "incumbent_world_sha256": world_hash, "incumbent_actor_sha256": actor_hash,
                                          "seed_start": 90216, "eval_episodes": 6}
            row["incumbent_evaluation"] = {"actor_success_rate": 0, "planner_success_rate": 0}
        rows.append(row)
    payload = {"track": "structured", "backend": {"real_vizdoom": True},
               "transitions": 25000 if stage == "real-control-25k" else 10000,
               "checkpoint_registry_verified": True,
               "stable_checkpoint": {"stage": stage if promoted else "real-wiring-2k"},
               "result": {"results": rows}}
    path = root / "real_vizdoom_training_receipt.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload


def _qualify_before_p2(root):
    """Write synthetic prerequisite receipts so tests exercise P2, not earlier gates."""
    (root / "execution_preflight.json").write_text(
        '{"ready":true,"required_failures":[]}', encoding="utf-8")
    training_seeds = [7101, 7102, 7103]
    evaluation_seeds = [9101, 9102, 9103, 9104, 9105, 9106]
    arms = {}
    for arm, transform in (("raw_reward", {"kind": "identity"}),
                           ("clipped_reward", {"kind": "clip", "min": -1.0, "max": 1.0})):
        seed_rows = []
        for seed in training_seeds:
            episodes = [{"seed": eval_seed,
                         "success": arm == "clipped_reward" and index < 2,
                         "return": float(index), "action_entropy_nats": 0.5,
                         "action_distribution": {"forward": 0.5}}
                        for index, eval_seed in enumerate(evaluation_seeds)]
            seed_rows.append({"seed": seed, "model_initialization_seed": seed,
                              "source_dataset_sha256": "d" * 64,
                              "shared_nonreward_data_sha256": "e" * 64,
                              "checkpoint_sha256": {"world": "f" * 64, "actor": "a" * 64},
                              "training_report": {"world_global_updates": 20, "actor_steps": 20},
                              "episodes": episodes})
        arms[arm] = {"reward_transform": transform, "per_training_seed": seed_rows}
    raw = {"format": "awa-v2.38.6-reward-diagnostic-raw-v1", "scenario": "my_way_home",
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
    (root / "reward_diagnostic.json").write_text(
        json.dumps(build_reward_diagnostic_report(raw)), encoding="utf-8")
    p1p_raw = _p1p_raw()
    p1p_raw["checkpoint_sha256"] = {"world": hashlib.sha256(b"p1-world").hexdigest(),
                                     "actor": hashlib.sha256(b"p1-actor").hexdigest()}
    replay = _p1p_replay_fixture(p1p_raw["seed_ids"])
    replay_text = json.dumps(replay)
    (root / "vizdoom_branch_replay.json").write_text(replay_text, encoding="utf-8")
    p1p_raw["branch_replay_report_sha256"] = hashlib.sha256(replay_text.encode()).hexdigest()
    (root / "planner_diagnostic_raw.json").write_text(json.dumps(p1p_raw), encoding="utf-8")
    (root / "planner_diagnostic.json").write_text(
        json.dumps(build_planner_diagnostic_report(p1p_raw)), encoding="utf-8")


def test_rejected_p2_cannot_advance_and_cached_legacy_requires_fresh_root(tmp_path):
    _qualify_before_p2(tmp_path)
    _receipt(tmp_path / "structured", promoted=False, paired=False, success=0)
    report = build_progressive_capability_report(tmp_path, protocol=PROTOCOL)
    assert report["completed_prefix"] == 4
    assert report["first_blocked_phase"] == "P2"
    plan = build_next_phase_execution_plan(executor_config=EXECUTOR, evidence_root=tmp_path, repo_root=ROOT)
    assert plan["status"] == "MANUAL_REQUIRED"
    assert plan["phase_id"] == "P2"
    assert not plan["commands"]


def test_p2_needs_paired_same_scenario_promoted_and_nonzero_success(tmp_path):
    _qualify_before_p2(tmp_path)
    path, payload = _receipt(tmp_path, paired=False)
    assert build_progressive_capability_report(tmp_path, protocol=PROTOCOL)["first_blocked_phase"] == "P2"
    path, payload = _receipt(tmp_path, success=0)
    assert build_progressive_capability_report(tmp_path, protocol=PROTOCOL)["first_blocked_phase"] == "P2"
    path, payload = _receipt(tmp_path, success=0.2)
    assert build_progressive_capability_report(tmp_path, protocol=PROTOCOL)["phases"][4]["evidence_condition_met"]
    payload["result"]["results"][1]["comparison_contract"]["scenario"] = "basic"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert build_progressive_capability_report(tmp_path, protocol=PROTOCOL)["first_blocked_phase"] == "P2"
    assert not build_progressive_capability_report(tmp_path, protocol=PROTOCOL)["phases"][4]["evidence_condition_met"]


def test_p1p_report_requires_raw_rows_and_verified_p1_checkpoint(tmp_path):
    _qualify_before_p2(tmp_path)
    _receipt(tmp_path / "structured", stage="real-wiring-2k")
    report = build_progressive_capability_report(tmp_path, protocol=PROTOCOL)
    assert report["first_blocked_phase"] == "P2"

    report_path = tmp_path / "planner_diagnostic.json"
    saved_report = report_path.read_text(encoding="utf-8")
    forged = json.loads(saved_report)
    forged["status"] = "BLOCKED"
    report_path.write_text(json.dumps(forged), encoding="utf-8")
    report = build_progressive_capability_report(tmp_path, protocol=PROTOCOL)
    assert report["first_blocked_phase"] == "P1P"
    assert report["phases"][3]["detail"] == "planner_report_does_not_match_raw_evidence"
    report_path.write_text(saved_report, encoding="utf-8")

    replay_path = tmp_path / "vizdoom_branch_replay.json"
    saved_replay = replay_path.read_text(encoding="utf-8")
    replay = json.loads(saved_replay)
    replay["real_backend_checked"] = False
    replay_path.write_text(json.dumps(replay), encoding="utf-8")
    report = build_progressive_capability_report(tmp_path, protocol=PROTOCOL)
    assert report["first_blocked_phase"] == "P1P"
    assert report["phases"][3]["detail"] == "missing_or_unqualified_bound_branch_replay"
    replay_path.write_text(saved_replay, encoding="utf-8")

    raw_path = tmp_path / "planner_diagnostic_raw.json"
    saved_raw = raw_path.read_text(encoding="utf-8")
    raw = json.loads(saved_raw)
    raw["tests"][0]["candidate_groups"][0]["realized_returns"][0] += 1
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    assert build_progressive_capability_report(tmp_path, protocol=PROTOCOL)["first_blocked_phase"] == "P1P"
    raw_path.write_text(saved_raw, encoding="utf-8")

    world = tmp_path / "structured" / "p1-world.pt"
    world.write_bytes(b"different-checkpoint")
    report = build_progressive_capability_report(tmp_path, protocol=PROTOCOL)
    assert report["first_blocked_phase"] == "P1P"
    assert report["phases"][3]["detail"] == "planner_checkpoint_not_bound_to_verified_p1"


def test_p9_only_uses_verified_p3_checkpoint_hashes(tmp_path):
    path, payload = _receipt(tmp_path / "structured", stage="real-control-25k")
    world = tmp_path / "world.pt"; actor = tmp_path / "actor.pt"
    world.write_bytes(b"world"); actor.write_bytes(b"actor")
    payload["stable_checkpoint_artifacts"] = {
        key: {"path": str(file), "sha256": hashlib.sha256(file.read_bytes()).hexdigest()}
        for key, file in (("world", world), ("actor", actor))
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _planner_checkpoints(tmp_path) == (str(world), str(actor))
    world.write_bytes(b"altered")
    assert _planner_checkpoints(tmp_path) is None
    world.write_bytes(b"world")
    payload["stable_checkpoint"]["stage"] = "real-wiring-2k"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _planner_checkpoints(tmp_path) is None


def test_p2_promotion_compares_incumbent_on_p2_scenario(monkeypatch, tmp_path):
    import awa.v2.game.vizdoom_campaign as module

    campaign = object.__new__(ViZDoomTrainingCampaign)
    campaign.out = tmp_path
    campaign.track = "structured"
    campaign.seed = 216
    campaign.device = "cpu"
    campaign.precision = "fp32"
    campaign.stages = [ViZDoomCampaignStage(name="p1", scenario="basic", target_transitions=1),
                       ViZDoomCampaignStage(name="p2", scenario="my_way_home", target_transitions=2)]
    campaign.promotion = PromotionPolicy(max_planner_success_regression=0.1)
    campaign.ledger = SimpleNamespace(run=lambda name, fn: fn(),
                                      phase=lambda name: SimpleNamespace(status="pending", result=None))
    campaign.replay = [None, None]
    campaign._collect_to = lambda stage: {}
    campaign._training_dataset = lambda stage: tmp_path / "data.npz"
    import numpy as np
    np.savez(tmp_path / "data.npz", rewards=np.asarray([0.0]), scenario_ids=np.asarray(["my_way_home"]))
    stable = {}

    def promote(stage, world, actor, metrics):
        entry = {"stage": stage, "world": str(world), "actor": str(actor), "metrics": metrics,
                 "world_sha256": hashlib.sha256(Path(world).read_bytes()).hexdigest(),
                 "actor_sha256": hashlib.sha256(Path(actor).read_bytes()).hexdigest()}
        stable.clear(); stable.update(entry)
        return entry

    campaign.registry = SimpleNamespace(current=lambda: stable or None, promote=promote)

    def fake_train(data, stage_dir, **kwargs):
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "vizdoom_world.pt").write_bytes(b"world")
        (stage_dir / "vizdoom_actor.pt").write_bytes(b"actor")
        return SimpleNamespace(to_dict=lambda: {})

    monkeypatch.setattr(module, "train_vizdoom_stack", fake_train)
    evaluations = []

    def evaluate(stage, world, actor, report):
        evaluations.append((stage.scenario, str(world)))
        score = 0.8 if stage.name == "p1" else 0.0 if "p1" in str(world) else 0.2
        return {"actor_success_rate": score, "planner_success_rate": score,
                "actor_mean_return": score, "planner_mean_return": score}

    campaign._evaluate = evaluate
    result = campaign.run()
    assert result["results"][1]["promoted"] is True
    assert result["results"][1]["incumbent_evaluation"]["planner_success_rate"] == 0
    assert evaluations[1][0] == evaluations[2][0] == "my_way_home"
    assert result["stable"]["stage"] == "p2"


def test_executor_does_not_report_pass_on_zero_evidence(monkeypatch, tmp_path):
    monkeypatch.setattr("awa.v2.research_os.progressive_executor.shutil.which", lambda _: "/usr/bin/fake")
    result = execute_next_progressive_phase(
        executor_config=EXECUTOR, evidence_root=tmp_path, repo_root=ROOT,
        runner=lambda *a, **kw: SimpleNamespace(returncode=0, stdout="ok", stderr=""),
    )
    assert result["execution"]["status"] == "FAILED"
    assert result["after"]["first_blocked_phase"] == "P0"
