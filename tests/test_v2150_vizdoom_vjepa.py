from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.representation import RepresentationCache
from awa.v2.video_representation import VJEPA2HFBackbone, ToyVideoBackbone, VideoClipSpec, precompute_game_video_cache
from awa.v2.game.vizdoom_env import (
    ViZDoomActionMapper, ViZDoomAetherEnv, ViZDoomConfig, ViZDoomScenario,
    VIZDOOM_GOAL_DIM, VIZDOOM_TELEMETRY_DIM, resolve_vizdoom_scenario_path,
)
from awa.v2.game.vizdoom_dataset import collect_vizdoom_dataset, ViZDoomExplorerPolicy
from awa.v2.game.vizdoom_vjepa import validate_vjepa_clip_contract, vizdoom_pixel_representation_contract, write_vizdoom_vjepa_manifest


class FakeDoomGame:
    variable_enums={name:name for name in ("HEALTH","ARMOR","SELECTED_WEAPON","SELECTED_WEAPON_AMMO","KILLCOUNT","ITEMCOUNT","POSITION_X","POSITION_Y","POSITION_Z","ANGLE","PITCH","ATTACK_READY","ON_GROUND","DEAD")}
    def __init__(self):
        self.seed=0; self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self.actions=[]
        self._buttons=["TURN_LEFT_RIGHT_DELTA","MOVE_FORWARD_BACKWARD_DELTA","ATTACK","USE"]
    def get_available_buttons(self): return list(self._buttons)
    def set_seed(self,seed): self.seed=int(seed)
    def new_episode(self): self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self.actions=[]
    def get_state(self):
        if self.finished: return None
        frame=np.zeros((3,24,32),dtype=np.uint8)
        frame[0]=np.uint8(min(255,20*self.t)); frame[1,:,min(31,int(self.x)):] = 100; frame[2]=25
        return SimpleNamespace(screen_buffer=frame)
    def make_action(self,a,skip):
        self.actions.append(list(a)); self.angle+=float(a[0]); self.x+=float(a[1])*.1; self.t+=1; self.finished=self.t>=5
        return float(self.x*.01)
    def is_episode_finished(self): return self.finished
    def is_player_dead(self): return self.dead
    def get_episode_time(self): return self.t
    def get_game_variable(self,name):
        vals={"HEALTH":100.,"ARMOR":10.,"SELECTED_WEAPON":2.,"SELECTED_WEAPON_AMMO":20.,"KILLCOUNT":0.,"ITEMCOUNT":0.,"POSITION_X":self.x,"POSITION_Y":2.,"POSITION_Z":0.,"ANGLE":self.angle,"PITCH":0.,"ATTACK_READY":1.,"ON_GROUND":1.,"DEAD":float(self.dead)}
        return vals.get(str(name),0.)
    def save(self,path): Path(path).write_text(json.dumps({"t":self.t,"x":self.x,"angle":self.angle,"finished":self.finished}))
    def load(self,path):
        row=json.loads(Path(path).read_text()); self.t=int(row["t"]); self.x=float(row["x"]); self.angle=float(row["angle"]); self.finished=bool(row["finished"])
    def close(self): pass


def make_env(track="pixel"):
    return ViZDoomAetherEnv(ViZDoomConfig(scenario=ViZDoomScenario.MY_WAY_HOME,track=track,frame_skip=1,max_episode_steps=8),game=FakeDoomGame())


def test_action_mapper_uses_delta_and_binary_controls():
    mapper=ViZDoomActionMapper(["TURN_LEFT_RIGHT_DELTA","MOVE_FORWARD_BACKWARD_DELTA","ATTACK","USE"],turn_delta=10,move_delta=20)
    row=mapper.map([.5,-.25,1,-1])
    assert row == pytest.approx([5.,-5.,1.,0.])


def test_vizdoom_env_structured_and_pixel_contracts():
    structured=make_env("structured"); obs,_=structured.reset(seed=7)
    assert obs.shape==(VIZDOOM_TELEMETRY_DIM,) and structured.goal_vector().shape==(VIZDOOM_GOAL_DIM,)
    nxt,r,term,trunc,_=structured.step([.5,.5,1,-1]); assert nxt.shape==obs.shape and np.isfinite(r) and not trunc
    assert structured.game.actions[-1][0] > 0 and structured.game.actions[-1][1] > 0 and structured.game.actions[-1][2] == 1
    structured.close()
    pixel=make_env("pixel"); frame,_=pixel.reset(seed=7); assert frame.shape==(24,32,3) and frame.dtype==np.uint8; pixel.close()


def test_snapshot_roundtrip_restores_world_state_and_marks_fidelity():
    env=make_env("pixel"); before,_=env.reset(seed=2); snap=env.snapshot(); env.step([.4,.8,-1,-1]); restored,info=env.restore(snap)
    assert np.array_equal(restored,before) and info["snapshot_fidelity"]=="world_state_only"
    env.close()


def test_pixel_dataset_preserves_uint8_and_telemetry(tmp_path):
    path=tmp_path/"doom.npz"
    report=collect_vizdoom_dataset(lambda:make_env("pixel"),path,episodes=2,horizon=8,base_seed=5,policy=ViZDoomExplorerPolicy(5))
    raw=np.load(path,allow_pickle=False)
    assert raw["observations"].dtype==np.uint8
    assert raw["observations"].shape[1:]==(24,32,3)
    assert raw["telemetry"].shape[1]==VIZDOOM_TELEMETRY_DIM
    assert raw["goals"].shape[1]==VIZDOOM_GOAL_DIM
    assert report.transitions>0 and report.track=="pixel"


def test_vjepa_clip_contract_is_fail_closed():
    toy=ToyVideoBackbone(8); toy.frames_per_clip=64
    validate_vjepa_clip_contract(toy,VideoClipSpec(64,1))
    with pytest.raises(ValueError): validate_vjepa_clip_contract(toy,VideoClipSpec(16,1))


class FakeHFModel(nn.Module):
    def __init__(self):
        super().__init__(); self.anchor=nn.Parameter(torch.zeros(())); self.config=SimpleNamespace(hidden_size=8,frames_per_clip=4,_commit_hash="abc"); self.vision_calls=0; self.forward_calls=0
    def get_vision_features(self,**kwargs):
        self.vision_calls+=1; return torch.ones(1,3,8)
    def forward(self,**kwargs):
        self.forward_calls+=1; return SimpleNamespace(last_hidden_state=torch.zeros(1,3,8))


class FakeProcessor:
    def __call__(self,clip,return_tensors="pt",**kwargs): return {"pixel_values_videos":torch.as_tensor(clip).float().unsqueeze(0)}


def test_hf_vjepa_adapter_prefers_official_get_vision_features():
    model=FakeHFModel(); backbone=VJEPA2HFBackbone("fake",model=model,processor=FakeProcessor(),local_files_only=True)
    out=backbone(torch.zeros(1,4,3,8,8))
    assert out.shape==(1,8) and model.vision_calls==1 and model.forward_calls==0


def test_vizdoom_pixel_dataset_feeds_existing_video_cache(tmp_path):
    data=tmp_path/"doom.npz"; collect_vizdoom_dataset(lambda:make_env("pixel"),data,episodes=1,horizon=8,base_seed=9)
    ds=OfflineTransitionDataset(data); backbone=ToyVideoBackbone(10); backbone.frames_per_clip=4
    cache=RepresentationCache(tmp_path/"cache",namespace="doom")
    idx=precompute_game_video_cache(ds,backbone,cache,clip_spec=VideoClipSpec(4,1),index_path=tmp_path/"index.json")
    assert idx["transitions"]==len(ds) and idx["writes"]>0
    assert idx["peak_pending_clips"]<=8
    assert idx["telemetry_dim"]==VIZDOOM_TELEMETRY_DIM
    assert idx["feature_dim"]==10+VIZDOOM_TELEMETRY_DIM


def test_pixel_representation_manifest_binds_encoder_and_clip(tmp_path):
    backbone=ToyVideoBackbone(8); backbone.frames_per_clip=4
    spec=VideoClipSpec(4,1)
    contract=vizdoom_pixel_representation_contract(backbone,frame_shape=(24,32,3),clip_spec=spec,telemetry_dim=VIZDOOM_TELEMETRY_DIM)
    assert contract.track.value=="pixel" and contract.clip_length==4
    path=write_vizdoom_vjepa_manifest(tmp_path/"manifest.json",backbone=backbone,frame_shape=(24,32,3),clip_spec=spec,scenario="my_way_home",dataset_sha256="a"*64)
    saved=json.loads(path.read_text()); assert saved["representation_fingerprint"]==contract.fingerprint


def test_explicit_scenario_path_is_respected(tmp_path):
    cfg=tmp_path/"custom.cfg"; cfg.write_text("# fake")
    path=resolve_vizdoom_scenario_path(SimpleNamespace(),ViZDoomScenario.BASIC,str(cfg))
    assert path==cfg.resolve()


def test_structured_dataset_is_float_and_same_fixed_abi(tmp_path):
    path=tmp_path/"structured.npz"; collect_vizdoom_dataset(lambda:make_env("structured"),path,episodes=1,horizon=5)
    ds=OfflineTransitionDataset(path)
    assert ds.manifest.observation_shape==(VIZDOOM_TELEMETRY_DIM,)
    assert ds.manifest.goal_shape==(VIZDOOM_GOAL_DIM,)
    assert ds.manifest.action_shape==(4,)
