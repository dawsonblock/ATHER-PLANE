from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.ablation import EMPIRICAL_ABLATIONS, build_empirical_ablation_protocol
from awa.v2.curriculum import ProceduralTaskFactory
from awa.v2.game import collect_game_dataset
from awa.v2.game.collectors import CoverageArenaPolicy
from awa.v2.game.variant_runtime import (
    SYSTEM_VARIANTS,
    augment_hindsight_dataset,
    train_and_evaluate_variant,
)
from awa.v2.rng_state import capture_rng_state, restore_rng_state


def _tiny_dataset(tmp_path: Path):
    factory=ProceduralTaskFactory(2270)
    train=[factory.make(s,0.25,s,"v227-test-train") for s in (1,4,10)]
    evaluate=[factory.make(s,0.35,100+s,"v227-test-eval") for s in (2,11)]
    path=tmp_path/"data.npz"
    collect_game_dataset(train,path,episodes_per_task=1,horizon=8,policy=CoverageArenaPolicy(seed=91))
    return path,train,evaluate


def test_v227_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_executable_registry_matches_preregistered_protocol():
    assert [x.variant_id for x in SYSTEM_VARIANTS] == [x.ablation_id for x in EMPIRICAL_ABLATIONS]
    p=build_empirical_ablation_protocol(seeds=[1,2,3,4,5],milestones=[25_000,100_000])
    assert p.protocol_id == "aether-v2.31-executable-component-ablation-v1"
    assert tuple(p.systems) == tuple(x.variant_id for x in SYSTEM_VARIANTS)
    assert all(len(v)==64 for _,v in p.system_config_sha256)


def test_rng_state_round_trip_restores_python_numpy_and_torch():
    random.seed(123); np.random.seed(123); torch.manual_seed(123)
    state=capture_rng_state()
    expected=(random.random(),np.random.random(3),torch.rand(3))
    random.random(); np.random.random(5); torch.rand(5)
    assert restore_rng_state(state)
    actual=(random.random(),np.random.random(3),torch.rand(3))
    assert expected[0] == actual[0]
    assert np.array_equal(expected[1],actual[1])
    assert torch.equal(expected[2],actual[2])


def test_raw_and_belief_baselines_are_real_separate_runtimes(tmp_path):
    dataset,train,evaluate=_tiny_dataset(tmp_path)
    for variant in ("actor_only","belief_actor"):
        out=tmp_path/variant
        tr,ev=train_and_evaluate_variant(
            variant,dataset,train,evaluate,out,device="cpu",sequence_length=1,
            hidden=16,world_epochs=1,actor_epochs=1,representation_epochs=1,
            calibration_epochs=0,batch_size=16,seed=3,horizon=8,voc_epochs=1,
            curriculum_mode="fixed",
        )
        assert tr.variant_id == ev.variant_id == variant
        assert Path(tr.checkpoint).exists()
        assert ev.episodes == len(evaluate)
        assert not (out/"world_stack"/"game_world.pt").exists()


def test_hindsight_replay_adds_grounded_one_step_rows(tmp_path):
    dataset,_,_=_tiny_dataset(tmp_path)
    target=tmp_path/"aug.npz"
    added=augment_hindsight_dataset(dataset,target,max_rows_fraction=.5)
    assert added > 0
    with np.load(dataset) as a, np.load(target) as b:
        assert len(b["observations"]) == len(a["observations"]) + added
        assert "sample_weights" in b
        assert np.allclose(b["sample_weights"][-added:],0.5)
        assert np.all(b["dones"][-added:])



def test_world_and_actor_checkpoints_embed_rng_lineage(tmp_path):
    dataset,train,evaluate=_tiny_dataset(tmp_path)
    out=tmp_path/"world"
    train_and_evaluate_variant(
        "world_actor",dataset,train,evaluate,out,device="cpu",sequence_length=1,
        hidden=16,world_epochs=1,actor_epochs=1,representation_epochs=1,
        calibration_epochs=0,batch_size=16,seed=3,horizon=8,voc_epochs=1,
        curriculum_mode="fixed",
    )
    wc=torch.load(out/"world_stack"/"game_world.pt",map_location="cpu",weights_only=False)
    ac=torch.load(out/"world_stack"/"game_actor.pt",map_location="cpu",weights_only=False)
    assert wc["world_training_rng_state"]["format"] == "awa-v2.27-rng-state-v1"
    assert ac["rng_state"]["format"] == "awa-v2.27-rng-state-v1"

def test_full_curriculum_fails_closed_without_adaptive_dataset_contract(tmp_path):
    dataset,train,evaluate=_tiny_dataset(tmp_path)
    with pytest.raises(ValueError,match="adaptive curriculum"):
        train_and_evaluate_variant(
            "full_curriculum",dataset,train,evaluate,tmp_path/"full",device="cpu",
            sequence_length=1,hidden=16,world_epochs=1,actor_epochs=1,
            calibration_epochs=0,batch_size=16,seed=3,horizon=8,voc_epochs=1,
            curriculum_mode="fixed",
        )
