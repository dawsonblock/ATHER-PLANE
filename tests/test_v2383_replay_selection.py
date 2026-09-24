import numpy as np
import pytest

from awa.v2.replay.sharded_store import ShardedReplayStore


def test_stage_budget_keeps_recent_successes_and_prior_episode_boundary(tmp_path):
    store = ShardedReplayStore(tmp_path / "replay", shard_size=4)
    store.append({
        "episode_ids": np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]),
        "dones": np.asarray([False, True] * 6),
        "rewards": np.asarray([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.float32),
    })
    target = store.materialize(tmp_path / "stage.npz", max_transitions=8, preserve_first=4)
    with np.load(target) as rows:
        assert len(rows["rewards"]) == 8
        assert rows["episode_ids"].tolist() == [0, 0, 1, 1, 4, 4, 5, 5]
        assert rows["rewards"].sum() == 2
        assert rows["dones"][3]


def test_stage_budget_rejects_splice_inside_prior_episode(tmp_path):
    store = ShardedReplayStore(tmp_path / "replay")
    store.append({"dones": np.asarray([False, False, True, False, True, True])})
    with pytest.raises(ValueError, match="inside an episode"):
        store.materialize(tmp_path / "invalid.npz", max_transitions=5, preserve_first=2)
