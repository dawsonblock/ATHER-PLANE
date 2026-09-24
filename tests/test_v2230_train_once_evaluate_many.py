from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.evolving_campaign import CampaignCell
from awa.v2.game import ACTION_DIM, GOAL_DIM, OBS_DIM
from awa.v2.meta_exploration import DeclarativeExplorationPolicy
from awa.v2.native_campaign import NativeProceduralCampaignRunner, NativeProceduralRunnerConfig


def test_v223_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


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


def test_train_once_evaluate_many_reuses_exact_training_artifact(tmp_path):
    calls = {"train": 0, "eval": 0}

    def train(dataset, tasks, out_dir, **kwargs):
        calls["train"] += 1
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "game_world.pt").write_bytes(b"same-world")
        (out / "game_actor.pt").write_bytes(b"same-actor")
        return _TrainReport()

    def evaluate(world_checkpoint, actor_checkpoint, tasks, **kwargs):
        calls["eval"] += 1
        novelty = str(tasks[0].novelty_class)
        score = 0.7 if "transfer" in novelty else 0.6
        return {
            "success_rate": score,
            "mean_return": score * 2,
            "constraint_violations": 0.0,
            "inference_latency_ms": 0.2,
        }

    cfg = NativeProceduralRunnerConfig(
        minimum_transitions=32, tasks_per_batch=3, eval_tasks=3,
        sequence_length=2, hidden=16, world_epochs=1, actor_epochs=1,
        calibration_epochs=0, batch_size=8, one_step_aux_epochs=0,
        reuse_training_artifacts=True,
    )
    runner = NativeProceduralCampaignRunner(cfg, collect_fn=_collector, train_fn=train, eval_fn=evaluate)
    policy = DeclarativeExplorationPolicy("active")
    heldout = runner.run(CampaignCell("active", 5, "procedural", "heldout", 32), policy, tmp_path / "runs" / "heldout")
    transfer = runner.run(CampaignCell("active", 5, "procedural", "transfer", 32), policy, tmp_path / "runs" / "transfer")

    assert calls == {"train": 1, "eval": 2}
    assert heldout.record.checkpoint_sha256 == transfer.record.checkpoint_sha256
    assert heldout.provenance.dataset_sha256 == transfer.provenance.dataset_sha256
    assert heldout.provenance.environment_sha256 != transfer.provenance.environment_sha256
    assert set(heldout.provenance.scenario_ids).isdisjoint(set(transfer.provenance.scenario_ids))
    assert heldout.record.metrics["training_cache_reused"] == 0.0
    assert transfer.record.metrics["training_cache_reused"] == 1.0
    assert transfer.record.metrics["success_rate"] == pytest.approx(0.7)
    assert (tmp_path / "runs" / "transfer" / "evaluation_receipt.json").exists()


def test_corrupt_training_cache_fails_closed(tmp_path):
    def train(dataset, tasks, out_dir, **kwargs):
        out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "game_world.pt").write_bytes(b"world")
        (out / "game_actor.pt").write_bytes(b"actor")
        return _TrainReport()

    def evaluate(*args, **kwargs):
        return {"success_rate": .5, "mean_return": 1.0, "constraint_violations": 0.0, "inference_latency_ms": .1}

    cfg = NativeProceduralRunnerConfig(minimum_transitions=32, eval_tasks=2, reuse_training_artifacts=True)
    runner = NativeProceduralCampaignRunner(cfg, collect_fn=_collector, train_fn=train, eval_fn=evaluate)
    policy = DeclarativeExplorationPolicy("active")
    c1 = CampaignCell("active", 8, "procedural", "heldout", 32)
    runner.run(c1, policy, tmp_path / "runs" / "h")
    cache = next((tmp_path / "runs" / "_training_cache").iterdir())
    (cache / "training_dataset.npz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        runner.run(CampaignCell("active", 8, "procedural", "transfer", 32), policy, tmp_path / "runs" / "t")
