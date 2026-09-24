from __future__ import annotations

from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch

from awa.v2.game.belief import GameBeliefEncoder
from awa.v2.game.hybrid_control import HybridOfflineActorCriticBaseline
from awa.v2.game.vizdoom_env import VIZDOOM_TELEMETRY_DIM
from awa.v2.research_os.planner_collection import PlannerBranchCollector, maximum_branch_steps
from awa.v2.world import MultimodalWorldModel
from test_v2160_vizdoom_training_closure import make_env
from test_v2386_planner_diagnostic import _p1p_replay_fixture


def _collector(tmp_path, *, real=False, make=make_env):
    torch.manual_seed(111)
    encoder = GameBeliefEncoder(obs_dim=VIZDOOM_TELEMETRY_DIM, goal_dim=8,
                                action_dim=4, latent_dim=8, local_dim=8,
                                global_dim=8, goal_latent_dim=8, event_slots=2)
    dim = encoder.belief_dim
    world = MultimodalWorldModel(dim, 4, hidden=16, components=2, horizons=(1, 2))
    actor = HybridOfflineActorCriticBaseline(dim, hidden=16)
    controller = SimpleNamespace(encoder=encoder.eval(), world=world.eval(), actor=actor,
                                 device=torch.device("cpu"))
    world_file, actor_file, replay_file = (tmp_path / name for name in ("world.pt", "actor.pt", "replay.json"))
    world_file.write_bytes(b"unit-world-checkpoint")
    actor_file.write_bytes(b"unit-actor-checkpoint")
    replay_file.write_text(json.dumps(_p1p_replay_fixture([101, 102, 103, 104, 105, 106])))
    return PlannerBranchCollector(make, controller, seeds=[101, 102, 103, 104, 105, 106],
                                  world_checkpoint=world_file, actor_checkpoint=actor_file,
                                  replay_report=replay_file, require_real_backend=real,
                                  max_episode_steps=7)


def test_collector_replays_realized_branches_and_scores_frozen_population(tmp_path):
    assert 25_000_000 < maximum_branch_steps(6, 16, 525) < 30_000_000
    collector = _collector(tmp_path)
    group = collector._group(101, 2, "mixed")
    a, b, c, d = (group[k] for k in ("P1P-A", "P1P-B", "P1P-C", "P1P-D"))
    assert a["predicted_scores"] == a["realized_returns"]
    assert len(a["branch_proofs"]) == 16
    assert all(proof["trace_sha256_by_replay"][0] == proof["trace_sha256_by_replay"][1]
               for proof in a["branch_proofs"])
    assert all(x["action_sequences"] == a["action_sequences"] for x in (b, c, d))
    assert b["origins"].count("random") == 8
    assert a["realized_returns"] == collector._group(101, 2, "mixed")["P1P-A"]["realized_returns"]
    assert all(len(seq) == 2 and all(action[2] == -1. and action[3] == -1.
                                     for action in seq) for seq in a["action_sequences"])
    outcome = collector._episode(101, 2, "mixed", "P1P-A")
    assert outcome["steps"] == 7
    assert np.isfinite(outcome["realized_return"])


def test_collector_rejects_fake_backend_for_empirical_receipt(tmp_path):
    collector = _collector(tmp_path, real=True)
    with pytest.raises(RuntimeError, match="refuses injected/fake backends"):
        collector._group(101, 1, "mixed")


def test_collector_does_not_publish_incomplete_raw_receipt(tmp_path, monkeypatch):
    collector = _collector(tmp_path)
    def stopped(*args):
        raise RuntimeError("simulated host interruption")
    monkeypatch.setattr(collector, "_group", stopped)
    target = tmp_path / "planner_diagnostic_raw.json"
    with pytest.raises(RuntimeError, match="interruption"):
        collector.collect(target)
    assert not target.exists()
    partial = json.loads((tmp_path / "planner_diagnostic_raw.json.partial").read_text())
    assert len(partial["random_baseline"]) == 1
