from __future__ import annotations

import json
import socket
import threading
from pathlib import Path

import numpy as np
import pytest

from awa.v2.ablation import AblationSpec, run_ablation_suite
from awa.v2.benchmark_tracks import BenchmarkTrack, RepresentationContract, BenchmarkManifest, assert_representation_compatible
from awa.v2.failure_triage import FailureEvidence, FailureClass, triage_failure
from awa.v2.game.bridge_protocol import BridgeCapabilities, BridgeSessionServer, JsonLineGameBridgeClient, PROTOCOL_VERSION
from awa.v2.qualification_report import MilestoneRow, build_milestone_report, write_milestone_report
from awa.v2.telemetry import TelemetryRecorder
from awa.v2.training.process_rollout import ProcessArenaRolloutPool
from awa.v2.curriculum import ProceduralTaskFactory


class _TinyBridgeEnv:
    def __init__(self):
        self.x = 0.0
        self.goal = np.asarray([1.0, 0.0], np.float32)

    def reset(self, seed=None):
        self.x = 0.0 if seed is None else float(int(seed) % 3) / 10.0
        return np.asarray([self.x, 0.0], np.float32), {"seed": seed}

    def goal_vector(self):
        return self.goal.copy()

    def step(self, action):
        self.x += float(np.asarray(action).reshape(-1)[0])
        done = self.x >= 1.0
        return np.asarray([self.x, 0.0], np.float32), float(self.x), done, False, {"success": done}

    def snapshot(self):
        return json.dumps({"x": self.x}).encode("utf-8")

    def restore(self, snapshot):
        self.x = float(json.loads(snapshot.decode("utf-8"))["x"])
        return np.asarray([self.x, 0.0], np.float32), {"restored": True}


def test_bridge_protocol_roundtrip_snapshot_and_ids():
    a, b = socket.socketpair()
    caps = BridgeCapabilities(2, 2, 1, supports_snapshot=True)
    server = BridgeSessionServer(_TinyBridgeEnv, caps)
    thread = threading.Thread(target=server.serve_socket, args=(a,), daemon=True); thread.start()
    client = JsonLineGameBridgeClient("unused", 0)
    client.sock = b; client._buffer = bytearray()
    # hello via the real request path
    result = client._request("hello", {"protocol": PROTOCOL_VERSION})
    client.capabilities = BridgeCapabilities.from_dict(result["capabilities"])
    obs, goal, _ = client.reset(seed=4)
    snap = client.snapshot()
    tr = client.step([0.5])
    restored, restored_goal, info = client.restore(snap)
    assert tr.observation[0] > obs[0]
    assert np.allclose(restored, obs) and np.allclose(restored_goal, goal)
    assert info["restored"] is True
    client.close(); thread.join(timeout=2)


def test_bridge_capabilities_validate_pixel_contract():
    with pytest.raises(ValueError):
        BridgeCapabilities(2, 2, 1, supports_pixels=True)
    caps = BridgeCapabilities(2, 2, 1, supports_pixels=True, frame_shape=(84, 84, 3))
    assert caps.to_dict()["frame_shape"] == [84, 84, 3]


def test_process_rollout_pool_uses_processes_and_preserves_policy_version():
    factory = ProceduralTaskFactory(214)
    tasks = [factory.make(1, .1, 0, "v214"), factory.make(3, .12, 1, "v214")]
    pool = ProcessArenaRolloutPool(workers=1, base_seed=214, start_method="spawn")
    report = pool.collect(tasks, episodes_per_task=1, horizon=12, policy="teacher", policy_version=7)
    assert report.episodes == 2 and report.total_steps > 0
    assert report.policy_version == 7
    assert all(row["policy_version"] == 7 for row in report.rows)


def test_telemetry_is_append_only_summarizable_and_rejects_nan(tmp_path):
    recorder = TelemetryRecorder(tmp_path / "events.jsonl", run_id="test")
    recorder.event("train", step=1, metrics={"loss": 2.0})
    recorder.event("train", step=2, metrics={"loss": 1.0})
    summary = recorder.summarize()
    assert summary.events == 2 and summary.event_counts["train"] == 2
    assert summary.latest_metrics["loss"] == 1.0 and summary.numeric_means["loss"] == 1.5
    with pytest.raises(ValueError): recorder.event("bad", metrics={"loss": float("nan")})


def test_failure_triage_distinguishes_world_actor_and_risk_evidence():
    world = triage_failure(FailureEvidence(False, world_prediction_error=.8))
    actor = triage_failure(FailureEvidence(False, actor_return=0., planner_return=.8, planner_available=True))
    risk = triage_failure(FailureEvidence(False, goal_progress=0., risk_rejections=5))
    assert world.primary == FailureClass.WORLD_MODEL
    assert actor.primary == FailureClass.ACTOR
    assert risk.primary == FailureClass.RISK


def test_structured_and_pixel_benchmarks_cannot_be_silently_mixed(tmp_path):
    structured = RepresentationContract(BenchmarkTrack.STRUCTURED, observation_dim=32)
    pixel = RepresentationContract(BenchmarkTrack.PIXEL, telemetry_dim=4, encoder_name="vjepa2", encoder_fingerprint="abc123", frame_shape=(224, 224, 3), clip_length=16)
    with pytest.raises(ValueError): assert_representation_compatible(structured, pixel)
    manifest = BenchmarkManifest("structured-main", structured, "a" * 64, "game-ood-v1", (1, 2, 3))
    path = manifest.save(tmp_path / "manifest.json")
    saved = json.loads(path.read_text())
    assert saved["representation"]["track"] == "structured"
    assert saved["representation_fingerprint"] == structured.fingerprint


def test_pixel_track_requires_encoder_identity():
    with pytest.raises(ValueError):
        RepresentationContract(BenchmarkTrack.PIXEL, frame_shape=(84,84,3), clip_length=8)


def test_ablation_runner_uses_paired_seed_comparison():
    specs = (AblationSpec("full"), AblationSpec("no_planner", planner=False))
    def evaluator(spec, seed):
        return {"success_rate": .8 + seed * .001 - (0.2 if not spec.planner else 0.0)}
    report = run_ablation_suite(evaluator, seeds=[1,2,3], specs=specs, resamples=200)
    assert report["summary"]["no_planner"]["delta_vs_full"]["mean"] < -0.15


def test_milestone_report_tracks_desired_learning_directions(tmp_path):
    rows = []
    for seed in (1,2,3):
        rows += [
            MilestoneRow(100_000, seed, .4, .5, .3, .5, .7, .1, 8, 100),
            MilestoneRow(1_000_000, seed, .8, .85, .7, .2, .2, .5, 3, 35),
        ]
    report = build_milestone_report(rows, resamples=200)
    assert all(t["improved"] for t in report["trends"].values())
    jp, mp = write_milestone_report(report, tmp_path)
    assert jp.exists() and mp.exists() and "1,000,000" in mp.read_text()
