from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.evolving_campaign import CampaignCell, orchestrator_from_config
from awa.v2.game import ACTION_DIM, GOAL_DIM, OBS_DIM
from awa.v2.meta_exploration import DeclarativeExplorationPolicy, DeclarativePolicyMutator
from awa.v2.native_campaign import NativeProceduralCampaignRunner, NativeProceduralRunnerConfig


def test_v225_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_plain_v2_import_is_lazy():
    env = dict(os.environ)
    root = Path(__file__).resolve().parents[1]
    env["PYTHONPATH"] = str(root / "src")
    code = "import sys, awa.v2; print(int('torch' in sys.modules), int('awa.v2.game.vizdoom_env' in sys.modules))"
    cp = subprocess.run([sys.executable, "-c", code], env=env, check=True, capture_output=True, text=True)
    assert cp.stdout.strip() == "0 0"


def test_native_orchestrator_mutates_only_execution_relevant_axes(tmp_path):
    config = {
        "active_policy": {"policy_id": "active"},
        "meta_exploration": {"mutation": {"step": 0.20}},
        "validation": {
            "seeds": [1, 2], "tasks": ["procedural"], "splits": ["heldout"],
            "milestones": [32], "minimum_seeds": 2,
        },
        "runner": {
            "type": "native_procedural",
            "config": {"tasks_per_batch": 24, "minimum_transitions": 32},
        },
    }
    o = orchestrator_from_config(config, tmp_path / "campaign")
    assert o.mutation_fields == (
        "quality_weight", "information_weight", "novelty_weight", "transfer_weight"
    )
    assert o.candidate_equivalence_key is not None
    mutator = DeclarativePolicyMutator(0.20, fields=o.mutation_fields)
    neighbors = mutator.neighbors(o.active_policy)
    assert len(neighbors) == 9
    signatures = {o.candidate_equivalence_key(p) for p in neighbors}
    assert 2 <= len(signatures) < len(neighbors)


def _collector(tasks, output, *, episodes_per_task, horizon, policy):
    n = 48
    z = np.zeros
    np.savez_compressed(
        output,
        observations=z((n, OBS_DIM), dtype=np.float32),
        goals=z((n, GOAL_DIM), dtype=np.float32),
        actions=z((n, ACTION_DIM), dtype=np.float32),
        rewards=z(n, dtype=np.float32),
        next_observations=z((n, OBS_DIM), dtype=np.float32),
        next_goals=z((n, GOAL_DIM), dtype=np.float32),
        dones=np.asarray([False] * (n - 1) + [True]),
        constraints=z((n, 4), dtype=np.float32),
    )
    return SimpleNamespace(transitions=n)


class _TrainReport:
    def to_dict(self):
        return {"diagnostic": 1.0}


def test_cumulative_milestone_training_collects_only_delta_and_warm_starts(tmp_path):
    calls = {"collect": 0, "train": []}

    def collect(*args, **kwargs):
        calls["collect"] += 1
        return _collector(*args, **kwargs)

    def train(dataset, tasks, out_dir, **kwargs):
        with np.load(dataset, allow_pickle=False) as z:
            n = len(z["rewards"])
        calls["train"].append({
            "rows": n,
            "resume_world": kwargs.get("resume_world_checkpoint"),
            "resume_actor": kwargs.get("resume_actor_checkpoint"),
            "planner": kwargs.get("run_planner_diagnostics"),
        })
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "game_world.pt").write_bytes(f"world-{len(calls['train'])}".encode())
        (out / "game_actor.pt").write_bytes(f"actor-{len(calls['train'])}".encode())
        return _TrainReport()

    def evaluate(*args, **kwargs):
        return {"success_rate": .5, "mean_return": 1.0, "constraint_violations": 0.0, "inference_latency_ms": .1}

    cfg = NativeProceduralRunnerConfig(
        minimum_transitions=32,
        tasks_per_batch=4,
        eval_tasks=2,
        sequence_length=2,
        hidden=16,
        world_epochs=1,
        actor_epochs=1,
        calibration_epochs=0,
        one_step_aux_epochs=0,
        batch_size=8,
        reuse_training_artifacts=True,
        cumulative_milestone_training=True,
        run_planner_diagnostics=False,
    )
    runner = NativeProceduralCampaignRunner(cfg, collect_fn=collect, train_fn=train, eval_fn=evaluate)
    policy = DeclarativeExplorationPolicy("active")
    root = tmp_path / "runs"

    runner.run(CampaignCell("active", 7, "procedural", "heldout", 32), policy, root / "m32")
    runner.run(CampaignCell("active", 7, "procedural", "heldout", 64), policy, root / "m64")

    # 32 rows require one 48-row collection batch; the next milestone collects only
    # the additional 32 rows, so total collection calls are 2 instead of 3.
    assert calls["collect"] == 2
    assert [x["rows"] for x in calls["train"]] == [32, 32]
    assert calls["train"][0]["resume_world"] is None
    assert calls["train"][1]["resume_world"] is not None
    assert calls["train"][1]["resume_actor"] is not None
    assert calls["train"][0]["planner"] is False
    assert calls["train"][1]["planner"] is False

    receipts = []
    for p in (root / "_training_cache").glob("*/training_artifact_receipt.json"):
        receipts.append(json.loads(p.read_text()))
    final = max(receipts, key=lambda x: int(x["retained_transitions"]))
    assert final["retained_transitions"] == 64
    assert final["incremental_retained_transitions"] == 32
    assert final["parent_training_identity_sha256"]
    artifact_root = root / "_training_cache" / final["training_identity_sha256"]
    with np.load(artifact_root / "training_dataset.npz", allow_pickle=False) as z:
        assert len(z["rewards"]) == 64
    with np.load(artifact_root / "training_delta.npz", allow_pickle=False) as z:
        assert len(z["rewards"]) == 32
