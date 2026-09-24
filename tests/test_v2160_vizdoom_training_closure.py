from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import torch
from torch import nn

from awa.v2.actor_baseline import LatentTransitionDataset
from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.representation import RepresentationCache
from awa.v2.video_representation import VJEPA2HFBackbone, ToyVideoBackbone, VideoClipSpec, precompute_game_video_cache
from awa.v2.world import MultimodalWorldModel
from awa.v2.planners import HybridRiskAwarePolicySeededMPPI
from awa.v2.game.belief import GameBeliefSequenceDataset
from awa.v2.game.hybrid_control import HybridGameActor, HybridOfflineActorCriticBaseline
from awa.v2.game.vizdoom_semantics import ViZDoomScenarioSemantics
from awa.v2.game.vizdoom_env import ViZDoomAetherEnv, ViZDoomConfig, ViZDoomScenario, VIZDOOM_TELEMETRY_DIM
from awa.v2.game.vizdoom_dataset import collect_vizdoom_dataset, ViZDoomExplorerPolicy
from awa.v2.game.vizdoom_training import materialize_vizdoom_cached_features, train_vizdoom_stack
from awa.v2.game.vizdoom_runtime import load_vizdoom_runtime
from awa.v2.game.vizdoom_campaign import ViZDoomTrainingCampaign


class FakeDoomGame:
    variable_enums={name:name for name in ("HEALTH","ARMOR","SELECTED_WEAPON","SELECTED_WEAPON_AMMO","KILLCOUNT","ITEMCOUNT","POSITION_X","POSITION_Y","POSITION_Z","ANGLE","PITCH","ATTACK_READY","ON_GROUND","DEAD")}
    def __init__(self):
        self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self.kills=0; self.health=100.; self.ammo=20.; self._buttons=["TURN_LEFT_RIGHT_DELTA","MOVE_FORWARD_BACKWARD_DELTA","ATTACK","USE"]
    def get_available_buttons(self): return list(self._buttons)
    def set_seed(self,seed): self.seed=int(seed)
    def new_episode(self): self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self.kills=0; self.health=100.; self.ammo=20.
    def get_state(self):
        if self.finished:return None
        frame=np.zeros((3,18,24),dtype=np.uint8); frame[0]=np.uint8(min(255,self.t*30)); frame[1,:,min(23,int(self.x)):] = 100; frame[2]=35
        return SimpleNamespace(screen_buffer=frame)
    def make_action(self,a,skip):
        self.angle+=float(a[0]); self.x+=max(0.,float(a[1]))*.15
        if float(a[2])>0.5 and self.ammo>0: self.ammo-=1.; self.kills=max(self.kills,1 if self.t>=2 else 0)
        self.t+=1; self.finished=self.t>=7
        return float(self.x*.02 + (1.0 if self.finished else 0.0))
    def is_episode_finished(self): return self.finished
    def is_player_dead(self): return self.dead
    def get_episode_time(self): return self.t
    def get_game_variable(self,name):
        vals={"HEALTH":self.health,"ARMOR":0.,"SELECTED_WEAPON":2.,"SELECTED_WEAPON_AMMO":self.ammo,"KILLCOUNT":self.kills,"ITEMCOUNT":0.,"POSITION_X":self.x,"POSITION_Y":0.,"POSITION_Z":0.,"ANGLE":self.angle,"PITCH":0.,"ATTACK_READY":1.,"ON_GROUND":1.,"DEAD":float(self.dead)}
        return vals.get(str(name),0.)
    def save(self,path): Path(path).write_text(json.dumps({"t":self.t,"x":self.x,"angle":self.angle,"finished":self.finished,"kills":self.kills,"ammo":self.ammo}))
    def load(self,path):
        r=json.loads(Path(path).read_text()); self.t=int(r["t"]); self.x=float(r["x"]); self.angle=float(r["angle"]); self.finished=bool(r["finished"]); self.kills=int(r["kills"]); self.ammo=float(r["ammo"])
    def close(self): pass


def make_env(track="structured",scenario=ViZDoomScenario.MY_WAY_HOME):
    return ViZDoomAetherEnv(ViZDoomConfig(scenario=scenario,track=track,frame_skip=1,max_episode_steps=10),game=FakeDoomGame())


def collect(tmp_path,track="structured",episodes=3):
    p=tmp_path/f"{track}.npz"
    collect_vizdoom_dataset(lambda:make_env(track),p,episodes=episodes,horizon=10,base_seed=216,policy=ViZDoomExplorerPolicy(216))
    return p


def test_doom_goal_dimensions_use_generic_temporal_sequence_loader(tmp_path):
    path=collect(tmp_path,"structured")
    ds=OfflineTransitionDataset(path)
    seq=GameBeliefSequenceDataset(ds,sequence_length=3,goal_dim=8,observation_dim=VIZDOOM_TELEMETRY_DIM,action_dim=4)
    row=seq[0]
    assert row["observations"].shape==(3,VIZDOOM_TELEMETRY_DIM) and row["goals"].shape==(3,8)
    assert set(ds.arrays["scenario_ids"].tolist()) == {"my_way_home"}
    assert "terminal_success" in ds.arrays and "health_delta_proxy" in ds.arrays


def test_hybrid_actor_emits_exact_binary_controls():
    actor=HybridGameActor(12,hidden=16)
    out=actor.deterministic_action(torch.randn(7,12))
    assert out.shape==(7,4)
    assert set(out[:,2:].unique().tolist()).issubset({-1.0,1.0})


def test_hybrid_planner_keeps_discrete_controls_exact():
    world=MultimodalWorldModel(10,4,hidden=16,components=2,horizons=(1,2))
    actor=HybridGameActor(10,hidden=16)
    planner=HybridRiskAwarePolicySeededMPPI(world,actor,horizon=3,candidates=8,samples=2)
    result=planner.plan(torch.randn(1,10),budget=8)
    assert result.action.shape==(1,4)
    assert all(abs(float(x))==1.0 for x in result.action[0,2:])


def test_scenario_semantics_are_task_specific():
    vals={"HEALTH":100.,"SELECTED_WEAPON_AMMO":0.,"KILLCOUNT":0.,"ITEMCOUNT":0.}
    read=lambda k: vals.get(k,0.)
    basic=ViZDoomScenarioSemantics("basic"); basic.reset(read); vals["KILLCOUNT"]=1.
    success,labels=basic.update(read,finished=False,truncated=False,player_dead=False,reward=0.,episode_return=0.)
    assert success and labels[2]==1.0
    cover=ViZDoomScenarioSemantics("take_cover"); cover.reset(read)
    success2,labels2=cover.update(read,finished=True,truncated=False,player_dead=False,reward=0.,episode_return=0.)
    assert success2 and labels2[2]==0.0


class BatchProcessor:
    def __init__(self): self.calls=0
    def __call__(self,videos=None,return_tensors="pt",**kwargs):
        self.calls+=1; x=torch.as_tensor(videos).float(); return {"pixel_values_videos":x}
class BatchModel(nn.Module):
    def __init__(self): super().__init__(); self.anchor=nn.Parameter(torch.zeros(())); self.config=SimpleNamespace(hidden_size=6,frames_per_clip=4,_commit_hash="batch") ; self.calls=0
    def get_vision_features(self,pixel_values_videos=None,**kwargs):
        self.calls+=1; b=pixel_values_videos.shape[0]; return torch.ones(b,3,6)


def test_hf_vjepa_batches_multiple_clips_in_one_model_call():
    proc=BatchProcessor(); model=BatchModel(); bb=VJEPA2HFBackbone("fake",model=model,processor=proc)
    out=bb(torch.zeros(5,4,3,8,8))
    assert out.shape==(5,6) and proc.calls==1 and model.calls==1


def test_cached_pixel_features_materialize_with_goals_and_risk(tmp_path):
    raw=collect(tmp_path,"pixel") ; ds=OfflineTransitionDataset(raw); bb=ToyVideoBackbone(7); bb.frames_per_clip=4
    cache=RepresentationCache(tmp_path/"cache",namespace="vizdoom-vjepa2")
    idx=precompute_game_video_cache(ds,bb,cache,clip_spec=VideoClipSpec(4,1),batch_size=4)
    out=materialize_vizdoom_cached_features(raw,cache,idx,tmp_path/"features.npz"); f=OfflineTransitionDataset(out)
    assert f.manifest.observation_shape==(7+VIZDOOM_TELEMETRY_DIM,)
    assert f.manifest.goal_shape==(8,) and f.manifest.constraints_shape==(4,)


def test_structured_doom_uses_temporal_world_and_hybrid_actor(tmp_path):
    raw=collect(tmp_path,"structured",episodes=4)
    rep=train_vizdoom_stack(raw,tmp_path/"trained-s",track="structured",sequence_length=3,world_epochs=1,actor_epochs=1,calibration_epochs=1,batch_size=8,hidden=24)
    assert rep.observation_dim==VIZDOOM_TELEMETRY_DIM and rep.belief_dim>rep.observation_dim
    assert (tmp_path/"trained-s/vizdoom_world.pt").exists() and (tmp_path/"trained-s/vizdoom_actor.pt").exists()


def test_structured_doom_cumulative_resume_keeps_optimizer_and_lineage(tmp_path):
    raw=collect(tmp_path,"structured",episodes=3)
    initial=tmp_path/"first"
    first=train_vizdoom_stack(raw,initial,sequence_length=3,world_epochs=1,actor_epochs=1,
        calibration_epochs=0,batch_size=8,hidden=24,device="cpu")
    second=train_vizdoom_stack(raw,tmp_path/"second",sequence_length=3,world_epochs=1,
        actor_epochs=1,calibration_epochs=0,batch_size=8,hidden=24,device="cpu",
        resume_world_checkpoint=initial/"vizdoom_world.pt",resume_actor_checkpoint=initial/"vizdoom_actor.pt")
    assert second.resumed and second.world_global_updates>first.world_global_updates
    assert second.resume_world_sha256 and second.resume_actor_sha256
    parent=torch.load(initial/"vizdoom_actor.pt",weights_only=False)
    child=torch.load(tmp_path/"second/vizdoom_actor.pt",weights_only=False)
    assert child["state"]["steps"]>parent["state"]["steps"]


def test_pixel_doom_features_use_same_temporal_training_path(tmp_path):
    raw=collect(tmp_path,"pixel",episodes=4); ds=OfflineTransitionDataset(raw); bb=ToyVideoBackbone(8); bb.frames_per_clip=4
    cache=RepresentationCache(tmp_path/"cache2",namespace="vizdoom-vjepa2"); idx=precompute_game_video_cache(ds,bb,cache,clip_spec=VideoClipSpec(4,1),batch_size=4)
    feat=materialize_vizdoom_cached_features(raw,cache,idx,tmp_path/"feat.npz")
    rep=train_vizdoom_stack(feat,tmp_path/"trained-p",track="pixel",sequence_length=3,world_epochs=1,actor_epochs=1,calibration_epochs=1,batch_size=8,hidden=24)
    assert rep.observation_dim==8+VIZDOOM_TELEMETRY_DIM and rep.actor_steps>0


def test_online_pixel_runtime_matches_trained_feature_abi_and_guards_action(tmp_path):
    raw=collect(tmp_path,"pixel",episodes=4); ds=OfflineTransitionDataset(raw); bb=ToyVideoBackbone(8); bb.frames_per_clip=4; spec=VideoClipSpec(4,1)
    cache=RepresentationCache(tmp_path/"cache3",namespace="vizdoom-vjepa2"); idx=precompute_game_video_cache(ds,bb,cache,clip_spec=spec,batch_size=4)
    feat=materialize_vizdoom_cached_features(raw,cache,idx,tmp_path/"feat2.npz")
    train_vizdoom_stack(feat,tmp_path/"trained-r",track="pixel",sequence_length=3,world_epochs=1,actor_epochs=1,calibration_epochs=1,batch_size=8,hidden=24)
    runtime=load_vizdoom_runtime(tmp_path/"trained-r/vizdoom_world.pt",tmp_path/"trained-r/vizdoom_actor.pt",video_backbone=bb,clip_spec=spec,use_planner=False,risk_limit=1.0)
    env=make_env("pixel"); obs,_=env.reset(seed=2); runtime.reset(); action,info=runtime.act(env,obs); env.close()
    assert action.shape==(4,) and all(abs(float(x))==1.0 for x in action[2:]) and np.isfinite(info.risk)


def test_vizdoom_campaign_is_planable_and_resumable(tmp_path):
    cfg={"track":"structured","seed":216,"runtime":{"device":"cpu","precision":"fp32"},"replay":{"shard_size":16},"stages":[{"name":"tiny","scenario":"my_way_home","target_transitions":14,"episodes_per_collection":2,"horizon":10,"sequence_length":3,"world_epochs":1,"actor_epochs":1,"calibration_epochs":1,"batch_size":8,"hidden":24,"eval_episodes":1}]}
    factory=lambda scenario,track: make_env(track,ViZDoomScenario(scenario))
    camp=ViZDoomTrainingCampaign(cfg,tmp_path/"camp",env_factory=factory)
    assert camp.plan()["stages"][0]["target_transitions"]==14
    out=camp.run(stop_after_stage="tiny")
    assert out["transitions"]>=14 and (tmp_path/"camp/experiment_state.json").exists()
    camp2=ViZDoomTrainingCampaign(cfg,tmp_path/"camp",env_factory=factory)
    out2=camp2.run(stop_after_stage="tiny")
    assert out2["transitions"]==out["transitions"]


def test_campaign_uses_promoted_planner_for_later_data_aggregation(tmp_path):
    cfg={"track":"structured","seed":216,"runtime":{"device":"cpu","precision":"fp32"},"replay":{"shard_size":16},"stages":[
        {"name":"s1","scenario":"my_way_home","target_transitions":7,"episodes_per_collection":1,"horizon":10,"sequence_length":3,"world_epochs":1,"actor_epochs":1,"calibration_epochs":1,"batch_size":8,"hidden":24,"eval_episodes":1,"planner_fraction":0.0},
        {"name":"s2","scenario":"my_way_home","target_transitions":14,"episodes_per_collection":1,"horizon":10,"sequence_length":3,"world_epochs":1,"actor_epochs":1,"calibration_epochs":1,"batch_size":8,"hidden":24,"eval_episodes":1,"planner_fraction":1.0},
    ]}
    factory=lambda scenario,track: make_env(track,ViZDoomScenario(scenario))
    camp=ViZDoomTrainingCampaign(cfg,tmp_path/"dagger",env_factory=factory)
    out=camp.run()
    rows=list(camp.replay.iter_shards())
    assert out["transitions"]>=14 and camp.registry.current() is not None
    assert any(bool(np.asarray(r["planner_used"]).any()) for r in rows[1:])
