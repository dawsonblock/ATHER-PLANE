from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import torch

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.ablation import EMPIRICAL_ABLATIONS, build_empirical_ablation_protocol
from awa.v2.curriculum import ProceduralTaskFactory
from awa.v2.evolving_campaign import CampaignCell
from awa.v2.game.belief import GameBeliefEncoder, GameBeliefWorldTrainer
from awa.v2.game.collectors import CoverageArenaPolicy
from awa.v2.game.dataset import collect_game_dataset
from awa.v2.game.procedural_arena import ACTION_DIM
from awa.v2.meta_exploration import DeclarativeExplorationPolicy
from awa.v2.native_campaign import NativeProceduralCampaignRunner, NativeProceduralRunnerConfig
from awa.v2.world import MultimodalWorldModel


def test_v226_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_ablation_protocol_binds_every_system_config_hash():
    p = build_empirical_ablation_protocol(
        seeds=[1, 2, 3, 4, 5], milestones=[25_000, 100_000]
    )
    assert p.systems == tuple(x.ablation_id for x in EMPIRICAL_ABLATIONS)
    bound = dict(p.system_config_sha256)
    assert set(bound) == set(p.systems)
    assert all(len(v) == 64 for v in bound.values())
    assert p.to_dict()["system_config_sha256"]["full_curriculum"] == bound["full_curriculum"]


def test_coverage_collector_is_reproducible_and_dataset_records_identity(tmp_path):
    task = ProceduralTaskFactory(123).make(1, 0.2, 0, "v226-test")
    a = CoverageArenaPolicy(seed=44)
    b = CoverageArenaPolicy(seed=44)
    # The full dataset collector resets both policies using the same task seed.
    out1, out2 = tmp_path / "a.npz", tmp_path / "b.npz"
    collect_game_dataset([task], out1, episodes_per_task=1, horizon=12, policy=a)
    collect_game_dataset([task], out2, episodes_per_task=1, horizon=12, policy=b)
    with np.load(out1) as z1, np.load(out2) as z2:
        assert np.array_equal(z1["actions"], z2["actions"])
    meta = json.loads(out1.with_suffix(".json").read_text())
    assert meta["collector"]["type"] == "coverage"


def _synthetic_batch(batch=2, steps=3):
    g = torch.Generator().manual_seed(7)
    return {
        "observations": torch.randn(batch, steps, 32, generator=g),
        "goals": torch.randn(batch, steps, 13, generator=g),
        "actions": torch.tanh(torch.randn(batch, steps, ACTION_DIM, generator=g)),
        "rewards": torch.randn(batch, steps, generator=g),
        "next_observations": torch.randn(batch, steps, 32, generator=g),
        "next_goals": torch.randn(batch, steps, 13, generator=g),
        "dones": torch.zeros(batch, steps),
    }


def test_world_optimizer_resume_is_numerically_continuous_on_same_batch():
    torch.manual_seed(19)
    enc1 = GameBeliefEncoder()
    world1 = MultimodalWorldModel(enc1.belief_dim, ACTION_DIM, hidden=16, components=3, horizons=(1,2))
    tr1 = GameBeliefWorldTrainer(enc1, world1, lr=7e-4, precision="fp32")
    enc1.eval(); world1.eval()  # deterministic miniature: disable Transformer dropout
    batch = _synthetic_batch()
    tr1.step(batch)

    enc2 = GameBeliefEncoder()
    world2 = MultimodalWorldModel(enc2.belief_dim, ACTION_DIM, hidden=16, components=3, horizons=(1,2))
    enc2.load_state_dict(copy.deepcopy(enc1.state_dict()))
    world2.load_state_dict(copy.deepcopy(world1.state_dict()))
    tr2 = GameBeliefWorldTrainer(enc2, world2, lr=7e-4, precision="fp32")
    enc2.eval(); world2.eval()
    tr2.load_trainer_state_dict(copy.deepcopy(tr1.trainer_state_dict()))

    tr1.step(batch)
    tr2.step(batch)
    assert tr1.global_updates == tr2.global_updates == 2
    for p1, p2 in zip(enc1.parameters(), enc2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-6)
    for p1, p2 in zip(world1.parameters(), world2.parameters()):
        assert torch.allclose(p1, p2, atol=1e-7, rtol=1e-6)


def test_native_aether_collection_bootstraps_then_uses_frozen_parent(tmp_path):
    cfg = NativeProceduralRunnerConfig(
        device="cpu", precision="fp32", tasks_per_batch=4, episodes_per_task=1,
        horizon=12, eval_tasks=1, sequence_length=1, hidden=16,
        world_epochs=1, actor_epochs=1, calibration_epochs=0,
        one_step_aux_epochs=0, batch_size=8, minimum_transitions=16,
        collection_mode="aether_actor", bootstrap_collection_mode="coverage",
        collector_planner_budget=4, collector_planner_horizon=2, collector_planner_stride=3,
    )
    runner = NativeProceduralCampaignRunner(cfg)
    policy = DeclarativeExplorationPolicy("active")
    root = tmp_path / "run"
    runner.run(CampaignCell("active", 5, "procedural", "heldout", 16), policy, root / "m16")
    runner.run(CampaignCell("active", 5, "procedural", "heldout", 32), policy, root / "m32")
    receipts = []
    for p in (root / "_training_cache").glob("*/collection_receipt.json"):
        receipts.append(json.loads(p.read_text()))
    first = min(receipts, key=lambda x: x["retained_transitions"])
    second = max(receipts, key=lambda x: x["retained_transitions"])
    assert first["batches"][0]["actual_collection_mode"] == "coverage"
    assert second["batches"][0]["actual_collection_mode"] == "aether_actor"
