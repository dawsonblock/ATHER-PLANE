from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import time
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from awa.v2.actor_baseline import LatentTransitionDataset, OfflineActorCriticBaseline
from awa.v2.compute_ledger import PhysicalComputeLedger
from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.planners import PolicySeededICEM, PolicySeededMPPI
from awa.v2.reasoning.effort import AdaptiveReasoningEffortController, EffortLevel
from awa.v2.rng_state import capture_rng_state, restore_rng_state
from awa.v2.game.voc_training import fit_game_voc
from .belief import GameBeliefEncoder, GameBeliefSequenceDataset, encode_game_transitions
from .controller import AdaptiveGamePolicy
from .goal_program import GOAL_DIM, NAME_TO_CODE
from .procedural_arena import ACTION_DIM, OBS_DIM, ProceduralArenaEnv
from .procedural_runtime import load_procedural_game_stack
from .training import train_game_stack


@dataclass(frozen=True)
class SystemVariant:
    """Executable ablation contract.

    v2.27 deliberately narrows the old broad "reusable learning" label to the
    concrete mechanism that is actually wired into this benchmark: grounded
    hindsight replay. Dormant skill/Engram/MoE families are not credited to a
    variant until they have their own executable and independently measurable path.
    """

    variant_id: str
    representation: str  # raw | belief | world
    world_model: bool = False
    fixed_planner: bool = False
    adaptive_compute: bool = False
    hindsight_replay: bool = False
    adaptive_curriculum: bool = False

    def __post_init__(self):
        if self.representation not in {"raw", "belief", "world"}:
            raise ValueError("representation must be raw/belief/world")
        if self.world_model != (self.representation == "world"):
            raise ValueError("world_model must match world representation")
        if (self.fixed_planner or self.adaptive_compute) and not self.world_model:
            raise ValueError("planning requires a world model")
        if self.fixed_planner and self.adaptive_compute:
            raise ValueError("variant must choose fixed or adaptive planning, not both")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SYSTEM_VARIANTS: tuple[SystemVariant, ...] = (
    SystemVariant("actor_only", "raw"),
    SystemVariant("belief_actor", "belief"),
    SystemVariant("world_actor", "world", world_model=True),
    SystemVariant("world_planner", "world", world_model=True, fixed_planner=True),
    SystemVariant("adaptive_compute", "world", world_model=True, adaptive_compute=True),
    SystemVariant("reusable_replay", "world", world_model=True, adaptive_compute=True, hindsight_replay=True),
    SystemVariant("full_curriculum", "world", world_model=True, adaptive_compute=True, hindsight_replay=True, adaptive_curriculum=True),
)
VARIANT_BY_ID = {v.variant_id: v for v in SYSTEM_VARIANTS}


@dataclass(frozen=True)
class VariantTrainingReport:
    variant_id: str
    transitions: int
    training_seconds: float
    auxiliary_rows: int
    checkpoint: str
    representation_loss: float = 0.0
    actor_steps: int = 0
    source_transitions: int = 0
    effective_training_rows: int = 0
    peak_cuda_memory_bytes: int = 0
    accelerator_wall_hours: float = 0.0

    def to_dict(self):
        return asdict(self)


@dataclass(frozen=True)
class VariantEvaluationReport:
    variant_id: str
    episodes: int
    success_rate: float
    mean_return: float
    constraint_violations: float
    inference_latency_ms: float
    planner_calls_per_episode: float = 0.0
    world_model_calls_per_episode: float = 0.0
    voc_samples: int = 0
    logical_world_model_transitions_per_episode: float = 0.0
    physical_world_model_forwards_per_episode: float = 0.0
    mean_world_model_batch_size: float = 0.0
    logical_transitions_per_forward: float = 0.0

    def to_dict(self):
        return asdict(self)


class BeliefReconstructionTrainer:
    """Train temporal belief without an action-conditioned world model."""

    FORMAT = "awa-v2.27-belief-reconstruction-state-v1"

    def __init__(self, encoder: GameBeliefEncoder, *, lr: float = 7e-4):
        self.encoder = encoder
        self.optimizer = torch.optim.AdamW(encoder.parameters(), lr=float(lr))
        self.global_updates = 0

    def step(self, batch) -> float:
        device = next(self.encoder.parameters()).device
        obs = batch["observations"].to(device)
        goals = batch["goals"].to(device)
        actions = batch["actions"].to(device)
        nxt = batch["next_observations"].to(device)
        ng = batch["next_goals"].to(device)
        B, T, _ = obs.shape
        state = self.encoder.initial(B, device)
        zero = torch.zeros(B, ACTION_DIM, device=device)
        state, belief = self.encoder.observe(state, obs[:, 0], zero, goals[:, 0], 0)
        losses = []
        for t in range(T):
            ro, rg = self.encoder.reconstruct(belief)
            losses.append(F.mse_loss(ro, obs[:, t]) + 0.5 * F.mse_loss(rg, goals[:, t]))
            state, next_belief = self.encoder.observe(state, nxt[:, t], actions[:, t], ng[:, t], t + 1)
            rn, rng = self.encoder.reconstruct(next_belief)
            losses.append(F.mse_loss(rn, nxt[:, t]) + 0.5 * F.mse_loss(rng, ng[:, t]))
            belief = next_belief
        loss = torch.stack(losses).mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 50.0)
        self.optimizer.step()
        self.global_updates += 1
        return float(loss.detach())

    def fit(self, dataset, *, epochs: int, batch_size: int) -> float:
        last = 0.0
        for _ in range(int(epochs)):
            for batch in DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=True):
                last = self.step(batch)
        return last

    def state_dict(self) -> dict[str, Any]:
        return {
            "format": self.FORMAT,
            "optimizer": self.optimizer.state_dict(),
            "global_updates": int(self.global_updates),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format") != self.FORMAT:
            raise ValueError("unsupported belief reconstruction trainer state")
        self.optimizer.load_state_dict(state["optimizer"])
        self.global_updates = int(state.get("global_updates", 0))


def _raw_states(dataset: OfflineTransitionDataset):
    a = dataset.arrays
    s = np.concatenate([a["observations"], a["goals"]], axis=-1).astype(np.float32)
    ns = np.concatenate([a["next_observations"], a["next_goals"]], axis=-1).astype(np.float32)
    return s, ns


def _reach_goal() -> np.ndarray:
    g = np.zeros(GOAL_DIM, dtype=np.float32)
    g[int(NAME_TO_CODE["reach_goal"])] = 1.0
    g[-3] = 0.0
    g[-2] = 0.25
    g[-1] = 1.0
    return g


def augment_hindsight_dataset(source: str | Path, output: str | Path, *, max_rows_fraction: float = 0.5) -> int:
    """Add grounded one-step future-goal relabels without inventing transitions."""
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k]) for k in z.files}
    n = len(arrays["observations"])
    if n == 0:
        raise ValueError("cannot augment empty dataset")
    episode_ids = arrays.get("episode_ids")
    if episode_ids is None:
        # Reconstruct episodes from done boundaries.
        episode_ids = np.zeros(n, dtype=np.int64)
        eid = 0
        for i in range(n):
            episode_ids[i] = eid
            if bool(arrays["dones"][i]):
                eid += 1
    extra: dict[str, list[np.ndarray | float | bool | int]] = {k: [] for k in arrays}
    limit = max(1, int(round(n * float(max_rows_fraction))))
    added = 0
    goal_template = _reach_goal()
    for eid in np.unique(episode_ids):
        idx = np.flatnonzero(episode_ids == eid)
        if idx.size < 2:
            continue
        achieved = arrays["next_observations"][idx[-1]][:2].astype(np.float32)
        for i in idx[:: max(1, idx.size // 4)]:
            if added >= limit:
                break
            s = arrays["observations"][i].copy()
            ns = arrays["next_observations"][i].copy()
            if s.shape[0] < 11 or ns.shape[0] < 11:
                continue
            s[8:10] = achieved; s[10] = 1.0
            ns[8:10] = achieved; ns[10] = 1.0
            d0 = float(np.linalg.norm(s[:2] - achieved))
            d1 = float(np.linalg.norm(ns[:2] - achieved))
            success = d1 <= 0.10
            reward = 1.5 * (d0 - d1) + (2.2 if success else 0.0)
            for key, arr in arrays.items():
                if key == "observations": value = s
                elif key == "next_observations": value = ns
                elif key in {"goals", "next_goals"}: value = goal_template.copy()
                elif key == "rewards": value = np.asarray(reward, dtype=arr.dtype)
                elif key == "dones": value = np.asarray(True, dtype=arr.dtype)
                elif key == "episode_ids": value = np.asarray(int(np.max(episode_ids)) + 1 + added, dtype=arr.dtype)
                else: value = arr[i].copy()
                extra[key].append(value)
            added += 1
        if added >= limit:
            break
    payload = {}
    for key, arr in arrays.items():
        if extra[key]:
            ext = np.asarray(extra[key], dtype=arr.dtype)
            payload[key] = np.concatenate([arr, ext], axis=0)
        else:
            payload[key] = arr
    # Structural replay weights: original rows weight 1, grounded hindsight rows 0.5.
    weights = arrays.get("sample_weights")
    if weights is None:
        payload["sample_weights"] = np.concatenate([
            np.ones(n, dtype=np.float32), np.full(added, 0.5, dtype=np.float32)
        ])
    else:
        payload["sample_weights"] = np.concatenate([
            np.asarray(weights, dtype=np.float32), np.full(added, 0.5, dtype=np.float32)
        ])
    out = Path(output); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    return int(added)


def _evaluate_raw_actor(actor: OfflineActorCriticBaseline, tasks: Iterable, *, horizon: int, seed_offset: int) -> VariantEvaluationReport:
    rows = []
    for task in list(tasks):
        env = ProceduralArenaEnv(task, horizon=int(horizon)); obs = env.reset(seed=task.seed + int(seed_offset))
        done = False; total = 0.0; violations = 0; lat = []; info = {}
        while not done:
            state = np.concatenate([np.asarray(obs, np.float32), env.goal_vector().astype(np.float32)])
            t0 = time.perf_counter(); action = actor.action(state).squeeze(0).cpu().numpy(); lat.append((time.perf_counter()-t0)*1000.0)
            obs, reward, done, info = env.step(action); total += float(reward)
            violations += int(np.any(env.constraint_labels() > 0.5))
        rows.append((bool(info.get("success", False)), total, violations, float(np.mean(lat) if lat else 0.0)))
    return VariantEvaluationReport("actor_only", len(rows), float(np.mean([x[0] for x in rows])), float(np.mean([x[1] for x in rows])), float(np.mean([x[2] for x in rows])), float(np.mean([x[3] for x in rows])))


def _evaluate_belief_actor(encoder: GameBeliefEncoder, actor: OfflineActorCriticBaseline, tasks: Iterable, *, horizon: int, seed_offset: int, variant_id: str = "belief_actor") -> VariantEvaluationReport:
    device = next(encoder.parameters()).device; rows = []
    encoder.eval(); actor.actor.eval()
    for task in list(tasks):
        env = ProceduralArenaEnv(task, horizon=int(horizon)); obs = env.reset(seed=task.seed + int(seed_offset))
        temporal = encoder.initial(1, device); prev = torch.zeros(1, ACTION_DIM, device=device); step = 0
        done=False; total=0.0; violations=0; lat=[]; info={}
        while not done:
            temporal, belief = encoder.observe(temporal, obs, prev, env.goal_vector(), step)
            t0=time.perf_counter(); action=actor.action(belief).squeeze(0).cpu().numpy(); lat.append((time.perf_counter()-t0)*1000.0)
            obs,reward,done,info=env.step(action); total += float(reward); violations += int(np.any(env.constraint_labels()>0.5))
            prev=torch.as_tensor(action,dtype=torch.float32,device=device).unsqueeze(0); step += 1
        rows.append((bool(info.get("success",False)), total, violations, float(np.mean(lat) if lat else 0.0)))
    return VariantEvaluationReport(variant_id, len(rows), float(np.mean([x[0] for x in rows])), float(np.mean([x[1] for x in rows])), float(np.mean([x[2] for x in rows])), float(np.mean([x[3] for x in rows])))


def _evaluate_world_policy(world_ckpt: Path, actor_ckpt: Path, tasks: Iterable, *, variant: SystemVariant, device: str, horizon: int, seed_offset: int, voc_tasks: Iterable | None = None, voc_epochs: int = 40, planner_world_batch_size: int | None = None) -> VariantEvaluationReport:
    runtime = load_procedural_game_stack(world_ckpt, actor_ckpt, device=device)
    planners = {
        "policy_mppi": PolicySeededMPPI(runtime.world, runtime.actor.actor, [-1.0]*ACTION_DIM, [1.0]*ACTION_DIM, horizon=4, candidates=24, world_batch_size=planner_world_batch_size),
        "policy_icem": PolicySeededICEM(runtime.world, runtime.actor.actor, [-1.0]*ACTION_DIM, [1.0]*ACTION_DIM, horizon=4, candidates=32, elites=8, iterations=2, world_batch_size=planner_world_batch_size),
    }
    voc = None; controller = None; voc_samples = 0
    if variant.adaptive_compute:
        vt = list(voc_tasks or list(tasks)[:2])
        choices = (("actor",0), ("policy_mppi",16), ("policy_icem",32))
        voc, vr = fit_game_voc(vt, runtime.encoder, runtime.actor, runtime.world, planners, choices=choices, states_per_task=2, branch_horizon=3, horizon=horizon, epochs=max(1,int(voc_epochs)))
        voc_samples = int(vr.samples)
        levels = [
            EffortLevel("reflex","actor",0,0.0),
            EffortLevel("shallow","policy_mppi",16,0.35),
            EffortLevel("deep","policy_icem",32,1.0),
        ]
        controller = AdaptiveReasoningEffortController(levels, compute_lambda=0.05, latency_lambda=0.001, risk_lambda=0.25)
    rows=[]
    for task in list(tasks):
        env=ProceduralArenaEnv(task,horizon=int(horizon)); obs=env.reset(seed=task.seed+int(seed_offset))
        temporal=runtime.encoder.initial(1,runtime.device); prev=torch.zeros(1,ACTION_DIM,device=runtime.device); step=0
        policy = None
        if variant.adaptive_compute:
            policy = AdaptiveGamePolicy(runtime.encoder,runtime.actor,runtime.world,planners=planners,voc=voc,effort_controller=controller,device=runtime.device)
            policy.reset_episode(task)
        done=False; total=0.0; violations=0; lat=[]; info={}; ledger=PhysicalComputeLedger()
        while not done:
            if variant.fixed_planner:
                temporal,belief=runtime.encoder.observe(temporal,obs,prev,env.goal_vector(),step)
                t0=time.perf_counter(); result=planners["policy_mppi"].plan(belief,actor=runtime.actor.actor,budget=24); elapsed_seconds=time.perf_counter()-t0; elapsed=elapsed_seconds*1000.0
                action=result.action.squeeze(0).cpu().numpy(); ledger.add_planner_result(result,elapsed_seconds=elapsed_seconds); lat.append(elapsed)
                prev=torch.as_tensor(action,dtype=torch.float32,device=runtime.device).unsqueeze(0); step += 1
            elif variant.adaptive_compute:
                t0=time.perf_counter(); action,used=policy.act(env,obs); elapsed=(time.perf_counter()-t0)*1000.0; lat.append(elapsed)
                if policy.last_effort_trace is not None: ledger.add_trace(policy.last_effort_trace)
            else:
                temporal,belief=runtime.encoder.observe(temporal,obs,prev,env.goal_vector(),step)
                t0=time.perf_counter(); action=runtime.actor.action(belief).squeeze(0).cpu().numpy(); lat.append((time.perf_counter()-t0)*1000.0)
                prev=torch.as_tensor(action,dtype=torch.float32,device=runtime.device).unsqueeze(0); step += 1
            obs,reward,done,info=env.step(action); total += float(reward); violations += int(np.any(env.constraint_labels()>0.5))
        comp=ledger.to_dict()
        rows.append((bool(info.get("success",False)),total,violations,float(np.mean(lat) if lat else 0.0),ledger.planner_calls,ledger.logical_world_model_transitions,ledger.physical_world_model_forwards,comp["logical_transitions_per_forward"],comp["logical_transitions_per_forward"]))
    return VariantEvaluationReport(
        variant.variant_id,len(rows),float(np.mean([x[0] for x in rows])),float(np.mean([x[1] for x in rows])),float(np.mean([x[2] for x in rows])),float(np.mean([x[3] for x in rows])),
        float(np.mean([x[4] for x in rows])),float(np.mean([x[5] for x in rows])),voc_samples,
        float(np.mean([x[5] for x in rows])),float(np.mean([x[6] for x in rows])),float(np.mean([x[7] for x in rows])),float(np.mean([x[8] for x in rows])),
    )


def train_and_evaluate_variant(
    variant_id: str,
    dataset_path: str | Path,
    train_tasks,
    eval_tasks,
    out_dir: str | Path,
    *,
    device: str = "cpu",
    sequence_length: int = 5,
    hidden: int = 64,
    world_epochs: int = 2,
    actor_epochs: int = 4,
    representation_epochs: int = 2,
    calibration_epochs: int = 2,
    batch_size: int = 64,
    seed: int = 0,
    horizon: int = 100,
    seed_offset: int = 31,
    voc_epochs: int = 40,
    curriculum_mode: str = "fixed",
    planner_world_batch_size: int | None = None,
) -> tuple[VariantTrainingReport, VariantEvaluationReport]:
    if variant_id not in VARIANT_BY_ID:
        raise ValueError(f"unknown variant {variant_id!r}")
    variant = VARIANT_BY_ID[variant_id]
    if curriculum_mode not in {"fixed", "adaptive"}:
        raise ValueError("curriculum_mode must be fixed/adaptive")
    if variant.adaptive_curriculum and curriculum_mode != "adaptive":
        raise ValueError("full_curriculum requires a dataset collected under adaptive curriculum allocation")
    if not variant.adaptive_curriculum and curriculum_mode != "fixed":
        raise ValueError("non-curriculum ablations require fixed curriculum data")
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    source_dataset = Path(dataset_path)
    source_transitions = len(OfflineTransitionDataset(source_dataset))
    training_dataset = source_dataset; auxiliary_rows = 0
    if variant.hindsight_replay:
        training_dataset = out / "training_with_hindsight.npz"
        auxiliary_rows = augment_hindsight_dataset(source_dataset, training_dataset)
    ds = OfflineTransitionDataset(training_dataset)
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    t0=time.perf_counter(); rep_loss=0.0; actor_steps=0

    if variant.representation == "raw":
        s,ns=_raw_states(ds)
        actor_ds=LatentTransitionDataset(s,ds.arrays["actions"],ds.arrays["rewards"],ns,ds.arrays["dones"])
        actor=OfflineActorCriticBaseline(s.shape[1],ACTION_DIM,[-1]*ACTION_DIM,[1]*ACTION_DIM,hidden=hidden,device=device,lr=7e-4,bc_weight=3.0)
        ar=actor.fit(actor_ds,epochs=actor_epochs,batch_size=min(batch_size,len(actor_ds))); actor_steps=ar.steps
        checkpoint=out/"actor_only.pt"
        torch.save({"format":"awa-v2.27-raw-actor-v1","state_dim":s.shape[1],"hidden":hidden,"state":actor.state_dict(),"rng_state":capture_rng_state()},checkpoint)
        eval_report=_evaluate_raw_actor(actor,eval_tasks,horizon=horizon,seed_offset=seed_offset)
    elif variant.representation == "belief":
        seq=GameBeliefSequenceDataset(ds,sequence_length=min(sequence_length,max(1,len(ds))))
        encoder=GameBeliefEncoder().to(device); trainer=BeliefReconstructionTrainer(encoder)
        rep_loss=trainer.fit(seq,epochs=representation_epochs,batch_size=min(batch_size,len(seq)))
        s,ns=encode_game_transitions(ds,encoder)
        actor_ds=LatentTransitionDataset(s,ds.arrays["actions"],ds.arrays["rewards"],ns,ds.arrays["dones"])
        actor=OfflineActorCriticBaseline(encoder.belief_dim,ACTION_DIM,[-1]*ACTION_DIM,[1]*ACTION_DIM,hidden=hidden,device=device,lr=7e-4,bc_weight=3.0)
        ar=actor.fit(actor_ds,epochs=actor_epochs,batch_size=min(batch_size,len(actor_ds))); actor_steps=ar.steps
        checkpoint=out/"belief_actor.pt"
        torch.save({"format":"awa-v2.27-belief-actor-v1","encoder_config":{"latent_dim":32,"local_dim":48,"global_dim":48,"goal_latent_dim":16},"encoder":encoder.state_dict(),"trainer_state":trainer.state_dict(),"actor_state":actor.state_dict(),"hidden":hidden,"rng_state":capture_rng_state()},checkpoint)
        eval_report=_evaluate_belief_actor(encoder,actor,eval_tasks,horizon=horizon,seed_offset=seed_offset)
    else:
        train_out=out/"world_stack"
        report=train_game_stack(training_dataset,train_tasks,train_out,sequence_length=sequence_length,world_epochs=world_epochs,actor_epochs=actor_epochs,calibration_epochs=calibration_epochs,batch_size=batch_size,hidden=hidden,seed=seed,device=device,precision="fp32",one_step_aux_epochs=1 if variant.hindsight_replay else 0,run_planner_diagnostics=False)
        actor_steps=int(report.actor_steps); rep_loss=float(report.world_loss)
        world_ckpt=train_out/"game_world.pt"; actor_ckpt=train_out/"game_actor.pt"; checkpoint=world_ckpt
        eval_report=_evaluate_world_policy(world_ckpt,actor_ckpt,eval_tasks,variant=variant,device=device,horizon=horizon,seed_offset=seed_offset,voc_tasks=train_tasks,voc_epochs=voc_epochs,planner_world_batch_size=planner_world_batch_size)
    elapsed=time.perf_counter()-t0
    peak_cuda=(int(torch.cuda.max_memory_allocated(device)) if str(device).startswith("cuda") and torch.cuda.is_available() else 0)
    tr=VariantTrainingReport(
        variant.variant_id,len(ds),float(elapsed),int(auxiliary_rows),str(checkpoint),float(rep_loss),int(actor_steps),
        int(source_transitions),int(len(ds)),int(peak_cuda),
        float(elapsed/3600.0 if str(device).startswith("cuda") else 0.0),
    )
    contract = variant.to_dict() | {"curriculum_mode": curriculum_mode}
    (out/"variant_contract.json").write_text(json.dumps(contract,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"training_report.json").write_text(json.dumps(tr.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    (out/"evaluation_report.json").write_text(json.dumps(eval_report.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return tr,eval_report


def evaluate_trained_variant(
    variant_id: str,
    trained_dir: str | Path,
    eval_tasks,
    *,
    device: str = "cpu",
    horizon: int = 100,
    seed_offset: int = 31,
    voc_tasks: Iterable | None = None,
    voc_epochs: int = 40,
    planner_world_batch_size: int | None = None,
) -> VariantEvaluationReport:
    """Evaluate a previously trained v2.27+ ablation runtime without retraining.

    v2.29 uses this to train once per system/seed/milestone and evaluate the same
    frozen checkpoint on both heldout and transfer splits. This closes the old
    possibility of accidentally paying for (or statistically changing) training
    once per evaluation split.
    """
    if variant_id not in VARIANT_BY_ID:
        raise ValueError(f"unknown variant {variant_id!r}")
    variant = VARIANT_BY_ID[variant_id]
    root = Path(trained_dir)
    if variant.representation == "raw":
        ckpt = torch.load(root / "actor_only.pt", map_location=device, weights_only=False)
        if ckpt.get("format") != "awa-v2.27-raw-actor-v1":
            raise ValueError("unsupported raw actor checkpoint format")
        state_dim = int(ckpt["state_dim"]); hidden = int(ckpt["hidden"])
        actor = OfflineActorCriticBaseline(
            state_dim, ACTION_DIM, [-1] * ACTION_DIM, [1] * ACTION_DIM,
            hidden=hidden, device=device, lr=7e-4, bc_weight=3.0,
        )
        actor.load_state_dict(ckpt["state"], load_optimizers=False)
        return _evaluate_raw_actor(actor, eval_tasks, horizon=horizon, seed_offset=seed_offset)
    if variant.representation == "belief":
        ckpt = torch.load(root / "belief_actor.pt", map_location=device, weights_only=False)
        if ckpt.get("format") != "awa-v2.27-belief-actor-v1":
            raise ValueError("unsupported belief actor checkpoint format")
        encoder = GameBeliefEncoder(**dict(ckpt.get("encoder_config", {}))).to(device)
        encoder.load_state_dict(ckpt["encoder"], strict=True)
        hidden = int(ckpt["hidden"])
        actor = OfflineActorCriticBaseline(
            encoder.belief_dim, ACTION_DIM, [-1] * ACTION_DIM, [1] * ACTION_DIM,
            hidden=hidden, device=device, lr=7e-4, bc_weight=3.0,
        )
        actor.load_state_dict(ckpt["actor_state"], load_optimizers=False)
        return _evaluate_belief_actor(
            encoder, actor, eval_tasks, horizon=horizon, seed_offset=seed_offset,
            variant_id=variant_id,
        )
    world_ckpt = root / "world_stack" / "game_world.pt"
    actor_ckpt = root / "world_stack" / "game_actor.pt"
    if not world_ckpt.exists() or not actor_ckpt.exists():
        raise FileNotFoundError("world variant checkpoint pair is incomplete")
    return _evaluate_world_policy(
        world_ckpt, actor_ckpt, eval_tasks, variant=variant, device=device,
        horizon=horizon, seed_offset=seed_offset, voc_tasks=voc_tasks,
        voc_epochs=voc_epochs, planner_world_batch_size=planner_world_batch_size,
    )
