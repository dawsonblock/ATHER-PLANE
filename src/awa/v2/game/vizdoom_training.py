from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import torch

from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.representation import RepresentationCache
from awa.v2.world import MultimodalWorldModel, RiskConstraintModel
from awa.v2.actor_baseline import LatentTransitionDataset, WeightedLatentTransitionDataset
from awa.v2.rng_state import capture_rng_state, restore_rng_state
from .belief import GameBeliefEncoder, GameBeliefSequenceDataset, GameBeliefWorldTrainer, encode_game_transitions
from .hybrid_control import HybridOfflineActorCriticBaseline
from .vizdoom_env import VIZDOOM_ACTION_DIM, VIZDOOM_GOAL_DIM, VIZDOOM_TELEMETRY_DIM


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def materialize_vizdoom_cached_features(
    pixel_dataset_path: str | Path,
    cache: RepresentationCache,
    index: dict[str, Any] | str | Path,
    output: str | Path,
) -> Path:
    """Materialize frozen V-JEPA(+telemetry) cache entries as a temporal dataset.

    This is the bridge that v2.15 lacked: cached video features retain the original
    Doom actions/goals/rewards/risk labels and become the observations consumed by
    the same goal-conditioned GRU+global-context trainer as structured telemetry.
    """
    source = OfflineTransitionDataset(pixel_dataset_path)
    if isinstance(index, (str, Path)):
        index = json.loads(Path(index).read_text(encoding="utf-8"))
    if str(index.get("dataset_sha256")) != source.manifest.sha256:
        raise ValueError("V-JEPA representation index does not match source Doom dataset SHA-256")
    if len(index.get("observation_keys", [])) != len(source) or len(index.get("next_observation_keys", [])) != len(source):
        raise ValueError("representation index length does not match Doom dataset")
    obs=[]; nxt=[]
    for ok, nk in zip(index["observation_keys"], index["next_observation_keys"]):
        of=cache.get_by_key(ok); nf=cache.get_by_key(nk)
        if of is None or nf is None:
            raise FileNotFoundError("V-JEPA representation cache entry missing")
        obs.append(of.reshape(-1).cpu().numpy().astype(np.float32))
        nxt.append(nf.reshape(-1).cpu().numpy().astype(np.float32))
    a=source.arrays
    payload={
        "observations":np.asarray(obs,dtype=np.float32),
        "actions":np.asarray(a["actions"],dtype=np.float32),
        "rewards":np.asarray(a["rewards"],dtype=np.float32),
        "next_observations":np.asarray(nxt,dtype=np.float32),
        "dones":np.asarray(a["dones"],dtype=np.bool_),
        "goals":np.asarray(a["goals"],dtype=np.float32),
        "next_goals":np.asarray(a["next_goals"],dtype=np.float32),
    }
    for key in ("constraints","episode_ids","sample_weights","planner_used","scenario_ids",
                "terminal_success","player_dead","episode_timeout","health_delta_proxy",
                "living_step_reward_proxy","displacement_proxy"):
        if key in a: payload[key]=np.asarray(a[key])
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**payload)
    meta={
        "format":"awa-v2.16-vizdoom-feature-dataset-v1",
        "source_dataset":str(Path(pixel_dataset_path)),
        "source_dataset_sha256":source.manifest.sha256,
        "representation_encoder_fingerprint":str(index.get("encoder_fingerprint")),
        "feature_dim":int(payload["observations"].shape[-1]),
        "visual_feature_dim":int(index.get("visual_feature_dim",0)),
        "telemetry_dim":int(index.get("telemetry_dim",0)),
        "transitions":len(source),
        "scenario_ids": sorted(set(str(x) for x in payload.get("scenario_ids", []))),
    }
    path.with_suffix(".json").write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return path


@dataclass(frozen=True)
class ViZDoomStackReport:
    track: str
    transitions: int
    sequences: int
    observation_dim: int
    belief_dim: int
    world_loss: float
    actor_steps: int
    movement_bc: float
    binary_bce: float
    risk_brier: float | None
    horizon_rmse: dict[str,float]
    value_rmse: float
    uncertainty_calibration_loss: float
    dataset_sha256: str
    resumed: bool = False
    world_global_updates: int = 0
    resume_world_sha256: str | None = None
    resume_actor_sha256: str | None = None

    def to_dict(self): return asdict(self)


def _risk_brier(world, states, actions, labels, batch_size=256):
    if labels is None: return None
    device=next(world.parameters()).device; vals=[]
    with torch.no_grad():
        for start in range(0,len(states),batch_size):
            s=torch.as_tensor(states[start:start+batch_size],dtype=torch.float32,device=device)
            a=torch.as_tensor(actions[start:start+batch_size],dtype=torch.float32,device=device)
            y=torch.as_tensor(labels[start:start+batch_size],dtype=torch.float32,device=device)
            p=world.risk.probabilities(s,a); vals.append((p-y).square().mean().cpu())
    return float(torch.stack(vals).mean()) if vals else None


def train_vizdoom_stack(
    dataset_path: str | Path,
    out_dir: str | Path,
    *,
    track: str = "structured",
    sequence_length: int = 8,
    horizons=(1,2,4,8),
    world_epochs: int = 2,
    actor_epochs: int = 4,
    calibration_epochs: int = 2,
    batch_size: int = 64,
    hidden: int = 128,
    seed: int = 216,
    device: str = "cpu",
    precision: str = "fp32",
    one_step_aux_epochs: int = 1,
    resume_world_checkpoint: str | Path | None = None,
    resume_actor_checkpoint: str | Path | None = None,
) -> ViZDoomStackReport:
    """Train Doom structured or V-JEPA feature data in one temporal belief space."""
    if track not in {"structured","pixel"}: raise ValueError("track must be structured or pixel")
    if (resume_world_checkpoint is None) != (resume_actor_checkpoint is None):
        raise ValueError("both world and actor checkpoints are required for cumulative training")
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    dataset=OfflineTransitionDataset(dataset_path)
    if len(dataset.manifest.observation_shape)!=1:
        raise ValueError("v2.16 Doom trainer expects vector observations; materialize V-JEPA cache for pixel training")
    if dataset.manifest.goal_shape != (VIZDOOM_GOAL_DIM,):
        raise ValueError(f"ViZDoom goal vectors must have shape {(VIZDOOM_GOAL_DIM,)}")
    if dataset.manifest.action_shape != (VIZDOOM_ACTION_DIM,):
        raise ValueError("ViZDoom action ABI must be 4-D")
    obs_dim=int(dataset.manifest.observation_shape[0])
    if track=="structured" and obs_dim != VIZDOOM_TELEMETRY_DIM:
        raise ValueError(f"structured ViZDoom telemetry must have {VIZDOOM_TELEMETRY_DIM} values")
    seq=GameBeliefSequenceDataset(dataset,sequence_length=sequence_length,goal_dim=VIZDOOM_GOAL_DIM,observation_dim=obs_dim,action_dim=VIZDOOM_ACTION_DIM)
    encoder=GameBeliefEncoder(obs_dim=obs_dim,goal_dim=VIZDOOM_GOAL_DIM,action_dim=VIZDOOM_ACTION_DIM,latent_dim=48,local_dim=64,global_dim=64,goal_latent_dim=16).to(device)
    world=MultimodalWorldModel(encoder.belief_dim,VIZDOOM_ACTION_DIM,hidden=hidden,components=3,horizons=tuple(horizons)).to(device)
    if dataset.manifest.constraints_shape is not None:
        world.risk=RiskConstraintModel(encoder.belief_dim,VIZDOOM_ACTION_DIM,hidden=hidden,constraints=int(np.prod(dataset.manifest.constraints_shape))).to(device)
    resumed = resume_world_checkpoint is not None
    world_parent_hash = _sha256(Path(resume_world_checkpoint)) if resumed else None
    actor_parent_hash = _sha256(Path(resume_actor_checkpoint)) if resumed else None
    if resumed:
        wc=torch.load(resume_world_checkpoint,map_location=device,weights_only=False)
        ac=torch.load(resume_actor_checkpoint,map_location=device,weights_only=False)
        if wc.get("format")!="awa-v2.16-vizdoom-world-v1" or ac.get("format")!="awa-v2.16-vizdoom-hybrid-actor-v1":
            raise ValueError("unsupported cumulative ViZDoom checkpoint format")
        if wc.get("trainer_state") is None or wc.get("world_training_rng_state") is None or ac.get("rng_state") is None:
            raise ValueError("resume checkpoint lacks optimizer or RNG continuation state")
        if (wc.get("track") != track or ac.get("track") != track or int(wc.get("obs_dim",-1)) != obs_dim
            or int(wc.get("goal_dim",-1)) != VIZDOOM_GOAL_DIM or int(wc.get("action_dim",-1)) != VIZDOOM_ACTION_DIM
            or int(wc.get("belief_dim",-1)) != encoder.belief_dim or int(wc.get("hidden",-1)) != int(hidden)
            or tuple(wc.get("horizons",())) != tuple(horizons) or int(ac.get("state_dim",-1)) != encoder.belief_dim
            or int(ac.get("action_dim",-1)) != VIZDOOM_ACTION_DIM or int(ac.get("hidden",-1)) != int(hidden)
            or wc.get("dataset_sha256") != ac.get("dataset_sha256")):
            raise ValueError("cumulative ViZDoom checkpoint architecture or lineage mismatch")
        encoder.load_state_dict(wc["encoder"],strict=True)
        world.load_state_dict(wc["world"],strict=True)
    trainer=GameBeliefWorldTrainer(encoder,world,lr=7e-4,precision=precision)
    if resumed:
        trainer.load_trainer_state_dict(wc["trainer_state"])
        restore_rng_state(wc["world_training_rng_state"])
    hist=trainer.fit(seq,epochs=world_epochs,batch_size=min(batch_size,len(seq)))
    if int(one_step_aux_epochs)>0 and sequence_length>1:
        aux=GameBeliefSequenceDataset(dataset,sequence_length=1,goal_dim=VIZDOOM_GOAL_DIM,observation_dim=obs_dim,action_dim=VIZDOOM_ACTION_DIM)
        hist.extend(trainer.fit(aux,epochs=int(one_step_aux_epochs),batch_size=min(batch_size,len(aux))))
    qual=trainer.evaluate_horizons(seq,horizons,batch_size=min(batch_size,len(seq)))
    calibration=trainer.calibrate_uncertainty(seq,epochs=calibration_epochs,batch_size=min(batch_size,len(seq)),lr=1e-3)
    world_training_rng_state=capture_rng_state()
    states,next_states=encode_game_transitions(dataset,encoder)
    labels=dataset.arrays.get("constraints")
    risk_brier=_risk_brier(world,states,dataset.arrays["actions"],labels)
    if "sample_weights" in dataset.arrays:
        actor_ds=WeightedLatentTransitionDataset(states,dataset.arrays["actions"],dataset.arrays["rewards"],next_states,dataset.arrays["dones"],dataset.arrays["sample_weights"])
    else:
        actor_ds=LatentTransitionDataset(states,dataset.arrays["actions"],dataset.arrays["rewards"],next_states,dataset.arrays["dones"])
    actor=HybridOfflineActorCriticBaseline(encoder.belief_dim,hidden=hidden,lr=7e-4,bc_weight=3.0,binary_bc_weight=1.0,device=device)
    if resumed:
        actor.load_state_dict(ac["state"],load_optimizers=True)
        restore_rng_state(ac["rng_state"])
    actor_report=actor.fit(actor_ds,epochs=actor_epochs,batch_size=min(batch_size,len(actor_ds)))
    actor_training_rng_state=capture_rng_state()
    report=ViZDoomStackReport(
        track,len(dataset),len(seq),obs_dim,encoder.belief_dim,float(hist[-1].loss if hist else 0.0),actor_report.steps,
        actor_report.movement_bc,actor_report.binary_bce,risk_brier,qual["horizon_rmse"],float(qual["value_rmse"]),float(calibration),dataset.manifest.sha256,
        resumed,int(trainer.global_updates),world_parent_hash,actor_parent_hash,
    )
    encoder_cfg={"obs_dim":obs_dim,"goal_dim":VIZDOOM_GOAL_DIM,"action_dim":VIZDOOM_ACTION_DIM,"latent_dim":48,"local_dim":64,"global_dim":64,"goal_latent_dim":16}
    torch.save({
        "format":"awa-v2.16-vizdoom-world-v1","track":track,"encoder":encoder.state_dict(),"world":world.state_dict(),
        "obs_dim":obs_dim,"goal_dim":VIZDOOM_GOAL_DIM,"action_dim":VIZDOOM_ACTION_DIM,"belief_dim":encoder.belief_dim,
        "hidden":hidden,"horizons":tuple(horizons),"encoder_config":encoder_cfg,"dataset_sha256":dataset.manifest.sha256,
        "risk_hidden":int(world.risk.net[0].out_features),"risk_constraints":int(world.risk.constraints),
        "runtime":{"device":str(device),"precision":str(precision)},
        "trainer_state":trainer.trainer_state_dict(),"world_training_rng_state":world_training_rng_state,
        "resume_world_sha256":world_parent_hash,"resume_actor_sha256":actor_parent_hash,
    },out/"vizdoom_world.pt")
    torch.save({
        "format":"awa-v2.16-vizdoom-hybrid-actor-v1","track":track,"state":actor.state_dict(),"state_dim":encoder.belief_dim,
        "action_dim":VIZDOOM_ACTION_DIM,"hidden":hidden,"dataset_sha256":dataset.manifest.sha256,
        "rng_state":actor_training_rng_state,"resume_world_sha256":world_parent_hash,"resume_actor_sha256":actor_parent_hash,
    },out/"vizdoom_actor.pt")
    (out/"vizdoom_training_report.json").write_text(json.dumps(report.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report
