from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
import numpy as np
import torch

from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.world import MultimodalWorldModel, RiskConstraintModel
from awa.v2.actor_baseline import LatentTransitionDataset, WeightedLatentTransitionDataset, OfflineActorCriticBaseline
from awa.v2.planners import PolicySeededMPPI
from awa.v2.rng_state import capture_rng_state, restore_rng_state
from .procedural_arena import ProceduralArenaEnv, OBS_DIM, ACTION_DIM
from .goal_program import GOAL_DIM
from .belief import GameBeliefEncoder, GameBeliefSequenceDataset, GameBeliefWorldTrainer, encode_game_transitions


@dataclass(frozen=True)
class GameStackReport:
    transitions: int
    sequences: int
    world_loss: float
    actor_steps: int
    actor_success_rate: float
    planner_success_rate: float
    actor_mean_return: float
    planner_mean_return: float
    planner_dependency: float
    horizon_rmse: dict[str, float]
    risk_brier: float | None
    belief_dim: int = 0
    value_rmse: float = 0.0
    uncertainty_calibration_loss: float = 0.0
    actor_constraint_violations: float = 0.0
    planner_constraint_violations: float = 0.0
    actor_inference_latency_ms: float = 0.0
    planner_inference_latency_ms: float = 0.0
    planner_calls_per_episode: float = 0.0
    planner_world_model_calls_per_episode: float = 0.0
    planner_diagnostics_ran: bool = True
    resumed_from_checkpoint: bool = False
    world_global_updates: int = 0

    def to_dict(self): return asdict(self)


@torch.no_grad()
def _run_controller(task, encoder, actor, world=None, planner=False, horizon=120, seed_offset=0):
    env=ProceduralArenaEnv(task,horizon=horizon)
    obs=env.reset(seed=task.seed+seed_offset)
    device=next(encoder.parameters()).device
    temporal=encoder.initial(1,device); prev_action=torch.zeros(1,ACTION_DIM,device=device); step=0
    done=False; total=0.0; steps=0; search=0; violations=0; world_calls=0; decision_ms=[]
    mppi = PolicySeededMPPI(world,actor.actor,[-1]*ACTION_DIM,[1]*ACTION_DIM,horizon=4,candidates=24) if planner and world is not None else None
    while not done:
        temporal,belief=encoder.observe(temporal,obs,prev_action,env.goal_vector(),step)
        t0=time.perf_counter()
        if mppi is None:
            action=actor.action(belief).squeeze(0).cpu().numpy()
        else:
            result=mppi.plan(belief,budget=24); action=result.action.squeeze(0).cpu().numpy(); search+=1; world_calls+=int(result.world_model_calls)
        decision_ms.append((time.perf_counter()-t0)*1000.0)
        obs,reward,done,info=env.step(action); total+=float(reward); steps+=1
        violations += int(np.any(env.constraint_labels() > 0.5))
        prev_action=torch.as_tensor(action,dtype=torch.float32,device=device).unsqueeze(0); step+=1
    return (bool(info.get("success",False)), total, search/max(1,steps), float(violations),
            int(search), int(world_calls), float(np.mean(decision_ms)) if decision_ms else 0.0)


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


def train_game_stack(
    dataset_path: str | Path,
    tasks,
    out_dir: str | Path,
    *,
    sequence_length: int = 5,
    horizons=(1,2,4),
    world_epochs: int = 2,
    actor_epochs: int = 4,
    calibration_epochs: int = 2,
    batch_size: int = 64,
    hidden: int = 96,
    seed: int = 0,
    device: str = "cpu",
    precision: str = "fp32",
    one_step_aux_epochs: int = 1,
    resume_world_checkpoint: str | Path | None = None,
    resume_actor_checkpoint: str | Path | None = None,
    run_planner_diagnostics: bool = True,
) -> GameStackReport:
    """Train the game stack in goal-conditioned temporal belief space.

    v2.26 supports optimizer-complete cumulative milestone training by restoring
    belief/world weights, AdamW state, GradScaler state when active, global update
    count, and the offline actor/critic optimizer state from a grounded checkpoint. Planner
    diagnostics are optional because the canonical heldout/transfer campaign uses
    the frozen actor-only evaluator and should not pay planner rollout cost during
    every training artifact.
    """
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    dataset=OfflineTransitionDataset(dataset_path)
    if tuple(dataset.manifest.observation_shape)!=(OBS_DIM,):
        raise ValueError(f"procedural game stack expects observation shape {(OBS_DIM,)}, got {dataset.manifest.observation_shape}")
    if dataset.manifest.goal_shape!=(GOAL_DIM,):
        raise ValueError("v2.10 game stack requires goal-conditioned dataset; regenerate with awa-v2-game-lab-generate")
    seq=GameBeliefSequenceDataset(dataset,sequence_length=sequence_length)
    encoder=GameBeliefEncoder(latent_dim=32,local_dim=48,global_dim=48,goal_latent_dim=16).to(device)
    world=MultimodalWorldModel(encoder.belief_dim,ACTION_DIM,hidden=hidden,components=3,horizons=tuple(horizons)).to(device)
    if dataset.manifest.constraints_shape is not None:
        world.risk=RiskConstraintModel(encoder.belief_dim,ACTION_DIM,hidden=hidden,constraints=int(np.prod(dataset.manifest.constraints_shape))).to(device)
    resumed = False
    resume_world_trainer_state = None
    if resume_world_checkpoint is not None:
        wc=torch.load(Path(resume_world_checkpoint),map_location=device,weights_only=False)
        if wc.get("format")!="awa-v2.10-game-world-v1":
            raise ValueError("unsupported resume world checkpoint format")
        if int(wc.get("obs_dim",-1))!=OBS_DIM or int(wc.get("goal_dim",-1))!=GOAL_DIM or int(wc.get("action_dim",-1))!=ACTION_DIM:
            raise ValueError("resume world checkpoint dimensions do not match procedural game stack")
        if int(wc.get("belief_dim",-1))!=encoder.belief_dim or int(wc.get("hidden",-1))!=int(hidden):
            raise ValueError("resume world checkpoint architecture mismatch")
        if tuple(int(x) for x in wc.get("horizons",()))!=tuple(int(x) for x in horizons):
            raise ValueError("resume world checkpoint horizons mismatch")
        encoder.load_state_dict(wc["encoder"],strict=True)
        world.load_state_dict(wc["world"],strict=True)
        resume_world_trainer_state = wc.get("trainer_state")
        resumed = True
    trainer=GameBeliefWorldTrainer(encoder,world,lr=7e-4,precision=precision)
    if resume_world_trainer_state is not None:
        trainer.load_trainer_state_dict(resume_world_trainer_state)
        # Restore the exact stochastic continuation after all model/optimizer
        # construction has consumed initialization RNG. Older checkpoints simply
        # continue statistically rather than bitwise.
        restore_rng_state(wc.get("world_training_rng_state"))
    hist=trainer.fit(seq,epochs=world_epochs,batch_size=min(batch_size,len(seq)))
    # One-step auxiliary updates consume structurally weighted replay/HER/counterfactual
    # rows that intentionally do not form long contiguous sequences.
    if int(one_step_aux_epochs)>0 and sequence_length>1:
        aux=GameBeliefSequenceDataset(dataset,sequence_length=1)
        hist.extend(trainer.fit(aux,epochs=int(one_step_aux_epochs),batch_size=min(batch_size,len(aux))))
    qual=trainer.evaluate_horizons(seq,horizons,batch_size=min(batch_size,len(seq)))
    calibration=trainer.calibrate_uncertainty(seq,epochs=calibration_epochs,batch_size=min(batch_size,len(seq)),lr=1e-3)
    world_training_rng_state = capture_rng_state()

    states,next_states=encode_game_transitions(dataset,encoder)
    labels=dataset.arrays.get("constraints")
    risk_brier=_risk_brier(world,states,dataset.arrays["actions"],labels)
    if "sample_weights" in dataset.arrays:
        actor_ds=WeightedLatentTransitionDataset(states,dataset.arrays["actions"],dataset.arrays["rewards"],next_states,dataset.arrays["dones"],dataset.arrays["sample_weights"])
    else:
        actor_ds=LatentTransitionDataset(states,dataset.arrays["actions"],dataset.arrays["rewards"],next_states,dataset.arrays["dones"])
    actor=OfflineActorCriticBaseline(encoder.belief_dim,ACTION_DIM,[-1]*ACTION_DIM,[1]*ACTION_DIM,hidden=hidden,lr=7e-4,bc_weight=3.0,device=device)
    if resume_actor_checkpoint is not None:
        ac=torch.load(Path(resume_actor_checkpoint),map_location=device,weights_only=False)
        if ac.get("format")!="awa-v2.10-game-actor-v1":
            raise ValueError("unsupported resume actor checkpoint format")
        if int(ac.get("state_dim",-1))!=encoder.belief_dim or int(ac.get("action_dim",-1))!=ACTION_DIM or int(ac.get("hidden",-1))!=int(hidden):
            raise ValueError("resume actor checkpoint architecture mismatch")
        actor.load_state_dict(ac["state"],load_optimizers=True)
        restore_rng_state(ac.get("rng_state"))
        resumed = True
    actor_report=actor.fit(actor_ds,epochs=actor_epochs,batch_size=min(batch_size,len(actor_ds)))
    actor_training_rng_state = capture_rng_state()

    eval_tasks=list(tasks)[:min(6,len(tasks))]
    actor_rows=[_run_controller(t,encoder,actor,None,False,seed_offset=31) for t in eval_tasks]
    planner_rows=(
        [_run_controller(t,encoder,actor,world,True,seed_offset=31) for t in eval_tasks]
        if run_planner_diagnostics else []
    )
    report=GameStackReport(
        len(dataset),len(seq),float(hist[-1].loss if hist else 0.0),actor_report.steps,
        float(np.mean([x[0] for x in actor_rows])) if actor_rows else 0.0,
        float(np.mean([x[0] for x in planner_rows])) if planner_rows else 0.0,
        float(np.mean([x[1] for x in actor_rows])) if actor_rows else 0.0,
        float(np.mean([x[1] for x in planner_rows])) if planner_rows else 0.0,
        float(np.mean([x[2] for x in planner_rows])) if planner_rows else 0.0,
        qual["horizon_rmse"],risk_brier,encoder.belief_dim,float(qual["value_rmse"]),float(calibration),
        float(np.mean([x[3] for x in actor_rows])) if actor_rows else 0.0,
        float(np.mean([x[3] for x in planner_rows])) if planner_rows else 0.0,
        float(np.mean([x[6] for x in actor_rows])) if actor_rows else 0.0,
        float(np.mean([x[6] for x in planner_rows])) if planner_rows else 0.0,
        float(np.mean([x[4] for x in planner_rows])) if planner_rows else 0.0,
        float(np.mean([x[5] for x in planner_rows])) if planner_rows else 0.0,
        bool(run_planner_diagnostics),
        bool(resumed),
        int(trainer.global_updates),
    )
    torch.save({
        "format":"awa-v2.10-game-world-v1","encoder":encoder.state_dict(),"world":world.state_dict(),
        "obs_dim":OBS_DIM,"goal_dim":GOAL_DIM,"action_dim":ACTION_DIM,"belief_dim":encoder.belief_dim,
        "hidden":hidden,"horizons":tuple(horizons),
        "encoder_config":{"latent_dim":32,"local_dim":48,"global_dim":48,"goal_latent_dim":16},
        "runtime":{"device":str(device),"precision":str(precision)},
        "trainer_state":trainer.trainer_state_dict(),
        "world_training_rng_state": world_training_rng_state,
    },out/"game_world.pt")
    torch.save({"format":"awa-v2.10-game-actor-v1","state":actor.state_dict(),"state_dim":encoder.belief_dim,"action_dim":ACTION_DIM,"hidden":hidden,"rng_state":actor_training_rng_state},out/"game_actor.pt")
    (out/"game_training_report.json").write_text(json.dumps(report.to_dict(),indent=2,sort_keys=True),encoding="utf-8")
    return report
