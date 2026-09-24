from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable
import json
import tempfile
from collections import Counter

import numpy as np

from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.experiment_ledger import ExperimentLedger
from awa.v2.replay.sharded_store import ShardedReplayStore
from awa.v2.representation import RepresentationCache
from awa.v2.video_representation import VideoClipSpec, precompute_game_video_cache, video_backbone_fingerprint
from awa.v2.campaign import CheckpointRegistry, PromotionPolicy
from .vizdoom_dataset import collect_vizdoom_dataset, ViZDoomExplorerPolicy
from .vizdoom_training import train_vizdoom_stack, materialize_vizdoom_cached_features, _sha256
from .vizdoom_runtime import load_vizdoom_runtime


@dataclass(frozen=True)
class ViZDoomCampaignStage:
    name: str
    scenario: str
    target_transitions: int
    episodes_per_collection: int = 8
    horizon: int = 600
    sequence_length: int = 8
    world_epochs: int = 2
    actor_epochs: int = 4
    calibration_epochs: int = 1
    batch_size: int = 64
    hidden: int = 128
    eval_episodes: int = 4
    planner_fraction: float = 0.0

    @classmethod
    def from_dict(cls, row): return cls(**row)
    def to_dict(self): return asdict(self)




class _RuntimeCollectionPolicy:
    def __init__(self, controller, planner_fraction: float, seed: int):
        self.controller=controller; self.planner_fraction=float(planner_fraction); self.rng=np.random.default_rng(int(seed)); self.last_planner_used=False
        if not 0.0 <= self.planner_fraction <= 1.0: raise ValueError("planner_fraction must be in [0,1]")
    def reset(self): self.controller.reset(); self.last_planner_used=False
    def act(self, env, observation):
        self.last_planner_used=bool(self.rng.random()<self.planner_fraction)
        action,_=self.controller.act(env,observation,force_planner=self.last_planner_used)
        return action

class ViZDoomTrainingCampaign:
    """Durable ViZDoom collection -> representation -> belief/world/actor campaign."""

    def __init__(
        self,
        config: dict[str,Any],
        out_dir: str|Path,
        *,
        env_factory: Callable[[str,str],Any],
        video_backbone=None,
        clip_spec: VideoClipSpec|None=None,
    ):
        self.config=json.loads(json.dumps(config)); self.out=Path(out_dir); self.out.mkdir(parents=True,exist_ok=True)
        self.track=str(self.config.get("track","structured"))
        if self.track not in {"structured","pixel"}: raise ValueError("campaign track must be structured or pixel")
        self.stages=[ViZDoomCampaignStage.from_dict(x) for x in self.config.get("stages",[])]
        if not self.stages: raise ValueError("ViZDoom campaign requires stages")
        targets=[s.target_transitions for s in self.stages]
        if targets != sorted(targets) or len(set(targets)) != len(targets): raise ValueError("stage targets must be strictly increasing")
        if len({s.hidden for s in self.stages}) != 1:
            raise ValueError("cumulative training requires identical model width at every stage")
        self.env_factory=env_factory; self.video_backbone=video_backbone; self.clip_spec=clip_spec or VideoClipSpec()
        if self.track=="pixel" and self.video_backbone is None: raise ValueError("pixel campaign requires a frozen V-JEPA-compatible backbone")
        resolved=json.loads(json.dumps(self.config))
        if self.video_backbone is not None:
            resolved["video_encoder_fingerprint"]=video_backbone_fingerprint(self.video_backbone)
            resolved["clip_spec"]=self.clip_spec.to_dict()
        self.ledger=ExperimentLedger(self.out,resolved,experiment_id="vizdoom_training_campaign")
        self.replay=ShardedReplayStore(self.out/"raw_replay",shard_size=int((self.config.get("replay") or {}).get("shard_size",50_000)))
        self.registry=CheckpointRegistry(self.out/"checkpoints")
        self.promotion=PromotionPolicy.from_dict(self.config.get("promotion"))
        self.cache=RepresentationCache(self.out/"vjepa_cache",namespace="vizdoom-vjepa2") if self.track=="pixel" else None
        self.seed=int(self.config.get("seed",216)); self.device=str((self.config.get("runtime") or {}).get("device","cpu")); self.precision=str((self.config.get("runtime") or {}).get("precision","fp32"))

    def plan(self):
        return {"track":self.track,"current_transitions":len(self.replay),"stages":[s.to_dict() for s in self.stages],"stable_checkpoint":self.registry.current()}

    def _append_collection(self, path: Path):
        with np.load(path,allow_pickle=False) as z: arrays={k:np.asarray(z[k]) for k in z.files}
        if "episode_ids" in arrays:
            offset=sum(s.transitions for s in self.replay.shards)+1_000_000*len(self.replay.shards)
            arrays["episode_ids"]=arrays["episode_ids"].astype(np.int64)+offset
        self.replay.append(arrays)

    def _collect_to(self, stage: ViZDoomCampaignStage):
        batch=0
        while len(self.replay) < int(stage.target_transitions):
            tmp=self.out/f"collect-{stage.name}-{batch:04d}.npz"
            factory=lambda: self.env_factory(stage.scenario,self.track)
            stable=self.registry.current()
            if stable is not None and stage.planner_fraction > 0:
                controller=load_vizdoom_runtime(stable["world"],stable["actor"],device=self.device,video_backbone=self.video_backbone,clip_spec=self.clip_spec,use_planner=True,planner_budget=32,risk_limit=1.0)
                policy=_RuntimeCollectionPolicy(controller,stage.planner_fraction,self.seed+batch)
            else:
                policy=ViZDoomExplorerPolicy(self.seed+batch)
            collect_vizdoom_dataset(factory,tmp,episodes=stage.episodes_per_collection,horizon=stage.horizon,base_seed=self.seed+len(self.replay)+batch*997,policy=policy)
            self._append_collection(tmp); tmp.unlink(missing_ok=True); tmp.with_suffix(".json").unlink(missing_ok=True); batch+=1
        bad=self.replay.verify()
        if bad: raise RuntimeError(f"replay integrity failure: {bad}")
        return {"transitions":len(self.replay),"target":stage.target_transitions,"collections":batch}

    def _training_dataset(self, stage: ViZDoomCampaignStage):
        raw=self.out/f"{stage.name}-raw.npz"
        if stage != self.stages[0]:
            first_collection=self.ledger.phase(f"collect:{self.stages[0].name}")
            if first_collection.status != "completed":
                raise RuntimeError('initial collection must complete before later training')
            first_count=int(first_collection.result["transitions"])
            self.replay.materialize(raw,max_transitions=stage.target_transitions,preserve_first=first_count)
        else:
            self.replay.materialize(raw,max_transitions=stage.target_transitions)
        if self.track=="structured": return raw
        ds=OfflineTransitionDataset(raw)
        index_path=self.out/f"{stage.name}-vjepa-index.json"
        index=precompute_game_video_cache(ds,self.video_backbone,self.cache,clip_spec=self.clip_spec,index_path=index_path,batch_size=int((self.config.get("video") or {}).get("batch_size",8)))
        feat=self.out/f"{stage.name}-features.npz"
        materialize_vizdoom_cached_features(raw,self.cache,index,feat)
        return feat

    @staticmethod
    def _dataset_diagnostics(path: Path) -> dict[str, Any]:
        with np.load(path, allow_pickle=False) as data:
            scenario_counts = {}
            if "scenario_ids" in data:
                scenario_counts = dict(sorted(Counter(str(x) for x in data["scenario_ids"].tolist()).items()))
            rewards = np.asarray(data["rewards"], dtype=np.float64)
            component = {}
            for key in ("terminal_success", "player_dead", "episode_timeout", "health_delta_proxy",
                        "living_step_reward_proxy", "displacement_proxy"):
                if key not in data:
                    continue
                values = np.asarray(data[key])
                if values.dtype.kind == "b":
                    component[key] = {"count": int(values.sum()), "samples": int(values.size)}
                else:
                    component[key] = {"count": int(values.size), "mean": float(values.mean()) if values.size else 0.0,
                                      "min": float(values.min()) if values.size else 0.0,
                                      "max": float(values.max()) if values.size else 0.0,
                                      "semantics": "telemetry-derived proxy"}
            return {
                "scenario_transition_counts": scenario_counts,
                "reward_statistics": {"count": int(rewards.size), "min": float(rewards.min()),
                                      "max": float(rewards.max()), "mean": float(rewards.mean()),
                                      "std": float(rewards.std()),
                                      "positive_fraction": float((rewards > 0).mean())},
                "reward_component_statistics": component,
                "dataset_sha256": _sha256(path),
            }

    def _evaluate(self, stage: ViZDoomCampaignStage, world_checkpoint: Path, actor_checkpoint: Path, train_report: dict[str,Any]):
        actor_success=[]; planner_success=[]; actor_returns=[]; planner_returns=[]
        for use_planner,sink_s,sink_r in ((False,actor_success,actor_returns),(True,planner_success,planner_returns)):
            for ep in range(stage.eval_episodes):
                env=self.env_factory(stage.scenario,self.track)
                try:
                    controller=load_vizdoom_runtime(world_checkpoint,actor_checkpoint,device=self.device,video_backbone=self.video_backbone,clip_spec=self.clip_spec,use_planner=use_planner,planner_budget=32)
                    obs,_=env.reset(seed=self.seed+90_000+ep); controller.reset(); total=0.; done=False; info={}
                    for _ in range(stage.horizon):
                        action,_=controller.act(env,obs,force_planner=use_planner)
                        obs,reward,term,trunc,info=env.step(action); total+=float(reward); done=bool(term or trunc)
                        if done: break
                finally:
                    env.close()
                sink_s.append(float(bool(info.get("success",False)))); sink_r.append(total)
        return {
            "actor_success_rate":float(np.mean(actor_success)) if actor_success else 0.0,
            "planner_success_rate":float(np.mean(planner_success)) if planner_success else 0.0,
            "actor_mean_return":float(np.mean(actor_returns)) if actor_returns else 0.0,
            "planner_mean_return":float(np.mean(planner_returns)) if planner_returns else 0.0,
            "risk_brier":train_report.get("risk_brier"),"horizon_rmse":train_report.get("horizon_rmse",{}),
        }

    def run(self, *, stop_after_stage: str|None=None):
        results=[]
        for stage_index,stage in enumerate(self.stages):
            self.ledger.run(f"collect:{stage.name}",lambda s=stage:self._collect_to(s))
            def train_phase(s=stage):
                data=self._training_dataset(s); stage_dir=self.out/"stages"/s.name
                dataset_diagnostics=self._dataset_diagnostics(data)
                incumbent=self.registry.current()
                if stage_index and (incumbent is None or incumbent.get("stage") != self.stages[stage_index-1].name):
                    raise RuntimeError(f"{s.name}: previous stage must be promoted before cumulative training")
                if incumbent and (_sha256(Path(incumbent["world"])) != incumbent.get("world_sha256")
                                  or _sha256(Path(incumbent["actor"])) != incumbent.get("actor_sha256")):
                    raise RuntimeError(f"{s.name}: stable checkpoint hash mismatch")
                rep=train_vizdoom_stack(data,stage_dir,track=self.track,sequence_length=s.sequence_length,world_epochs=s.world_epochs,actor_epochs=s.actor_epochs,calibration_epochs=s.calibration_epochs,batch_size=s.batch_size,hidden=s.hidden,seed=self.seed,device=self.device,precision=self.precision,
                    resume_world_checkpoint=Path(incumbent["world"]) if incumbent else None,
                    resume_actor_checkpoint=Path(incumbent["actor"]) if incumbent else None)
                comparison={"scenario":s.scenario,"track":self.track,"eval_episodes":s.eval_episodes,"seed_start":self.seed+90_000,
                            "incumbent_stage":incumbent["stage"] if incumbent else None,
                            "incumbent_world_sha256":incumbent.get("world_sha256") if incumbent else None,
                            "incumbent_actor_sha256":incumbent.get("actor_sha256") if incumbent else None}
                baseline=(self._evaluate(s,Path(incumbent["world"]),Path(incumbent["actor"]),{}) if incumbent else None)
                metrics=self._evaluate(s,stage_dir/"vizdoom_world.pt",stage_dir/"vizdoom_actor.pt",rep.to_dict())
                ok,reasons=self.promotion.assess(metrics,baseline)
                promoted=None
                if ok: promoted=self.registry.promote(s.name,stage_dir/"vizdoom_world.pt",stage_dir/"vizdoom_actor.pt",metrics)
                baseline_registration = bool(ok and incumbent is None)
                return {"training":rep.to_dict(),"dataset_diagnostics":dataset_diagnostics,
                        "evaluation":metrics,"incumbent_evaluation":baseline,"comparison_contract":comparison,
                        "promoted":bool(ok and incumbent is not None),"baseline_registered":baseline_registration,
                        "promotion_status":"baseline_registration" if baseline_registration else ("promoted" if ok else "not_promoted"),
                        "promotion_reasons":reasons,"stable":promoted}
            cached=self.ledger.phase(f"train:{stage.name}")
            if cached.status=="completed" and (cached.result or {}).get("comparison_contract") is None and stage.name != self.stages[0].name:
                raise RuntimeError(f"{stage.name}: earlier cached promotion cannot be requalified safely; archive this run and start a new v2.38.6 campaign directory")
            result=self.ledger.run(f"train:{stage.name}",train_phase); results.append({"stage":stage.name,**result})
            if stop_after_stage==stage.name: break
        return {"status":"ok","track":self.track,"transitions":len(self.replay),"results":results,"stable":self.registry.current()}
