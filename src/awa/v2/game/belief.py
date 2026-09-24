from __future__ import annotations

from dataclasses import dataclass
from contextlib import nullcontext
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from awa.v2.temporal import LocalGlobalTemporalCore, TemporalState
from awa.v2.world import MultimodalWorldModel
from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.world_training import contiguous_sequence_starts
from .procedural_arena import OBS_DIM, ACTION_DIM
from .goal_program import GOAL_DIM


class GameBeliefEncoder(nn.Module):
    """Goal-conditioned local/global temporal belief encoder for game control.

    The encoder is trained with observation/goal reconstruction while the world model
    learns action-conditioned belief dynamics. This prevents the temporal path from
    being a disconnected random feature transform.
    """

    def __init__(
        self,
        obs_dim: int = OBS_DIM,
        goal_dim: int = GOAL_DIM,
        action_dim: int = ACTION_DIM,
        *,
        latent_dim: int = 32,
        local_dim: int = 48,
        global_dim: int = 48,
        goal_latent_dim: int = 16,
        event_slots: int = 8,
        global_stride: int = 4,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim); self.goal_dim = int(goal_dim); self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim); self.local_dim = int(local_dim); self.global_dim = int(global_dim)
        self.goal_latent_dim = int(goal_latent_dim)
        self.obs_encoder = nn.Sequential(nn.Linear(self.obs_dim, self.latent_dim), nn.LayerNorm(self.latent_dim), nn.Tanh())
        heads = 4 if self.global_dim % 4 == 0 else 1
        self.temporal = LocalGlobalTemporalCore(
            self.latent_dim, self.action_dim, local_dim=self.local_dim, global_dim=self.global_dim,
            event_slots=event_slots, global_stride=global_stride, heads=heads,
        )
        self.goal_encoder = nn.Sequential(nn.Linear(self.goal_dim, self.goal_latent_dim), nn.LayerNorm(self.goal_latent_dim), nn.Tanh())
        self.belief_dim = self.local_dim + self.global_dim + self.goal_latent_dim
        self.obs_decoder = nn.Sequential(nn.Linear(self.belief_dim, 96), nn.SiLU(), nn.Linear(96, self.obs_dim))
        self.goal_decoder = nn.Sequential(nn.Linear(self.belief_dim, 64), nn.SiLU(), nn.Linear(64, self.goal_dim))

    def initial(self, batch: int, device=None, dtype=torch.float32) -> TemporalState:
        device = device or next(self.parameters()).device
        return self.temporal.initial(int(batch), device, dtype=dtype)

    def observe(self, state: TemporalState, observation, previous_action, goal, step: int):
        device = next(self.parameters()).device
        obs = torch.as_tensor(observation, dtype=torch.float32, device=device)
        act = torch.as_tensor(previous_action, dtype=torch.float32, device=device)
        g = torch.as_tensor(goal, dtype=torch.float32, device=device)
        if obs.ndim == 1: obs = obs.unsqueeze(0)
        if act.ndim == 1: act = act.unsqueeze(0)
        if g.ndim == 1: g = g.unsqueeze(0)
        latent = self.obs_encoder(obs)
        next_state = self.temporal(state, latent, act, int(step))
        goal_latent = self.goal_encoder(g)
        belief = torch.cat([next_state.local, next_state.global_ctx, goal_latent], dim=-1)
        return next_state, belief

    def reconstruct(self, belief: torch.Tensor):
        return self.obs_decoder(belief), self.goal_decoder(belief)


class GameBeliefSequenceDataset(Dataset):
    def __init__(self, transitions: OfflineTransitionDataset, sequence_length: int = 8, *, goal_dim: int | None = None, observation_dim: int | None = None, action_dim: int | None = None):
        if transitions.manifest.goal_shape is None or len(transitions.manifest.goal_shape) != 1:
            raise ValueError("game belief training requires a 1-D goal vector")
        inferred_goal = int(transitions.manifest.goal_shape[0])
        inferred_obs = int(np.prod(transitions.manifest.observation_shape))
        inferred_action = int(np.prod(transitions.manifest.action_shape))
        if goal_dim is not None and inferred_goal != int(goal_dim):
            raise ValueError(f"goal dimension mismatch: dataset={inferred_goal}, expected={int(goal_dim)}")
        if observation_dim is not None and inferred_obs != int(observation_dim):
            raise ValueError(f"observation dimension mismatch: dataset={inferred_obs}, expected={int(observation_dim)}")
        if action_dim is not None and inferred_action != int(action_dim):
            raise ValueError(f"action dimension mismatch: dataset={inferred_action}, expected={int(action_dim)}")
        self.goal_dim = inferred_goal
        self.observation_dim = inferred_obs
        self.action_dim = inferred_action
        self.transitions = transitions
        self.sequence_length = int(sequence_length)
        self.starts = contiguous_sequence_starts(transitions, self.sequence_length, verify_chain=True)
        if not self.starts:
            raise ValueError("no contiguous game sequences available")

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        start = self.starts[int(idx)]; stop = start + self.sequence_length
        a = self.transitions.arrays
        row = {
            "observations": torch.as_tensor(a["observations"][start:stop]).float(),
            "goals": torch.as_tensor(a["goals"][start:stop]).float(),
            "actions": torch.as_tensor(a["actions"][start:stop]).float(),
            "rewards": torch.as_tensor(a["rewards"][start:stop]).float(),
            "next_observations": torch.as_tensor(a["next_observations"][start:stop]).float(),
            "next_goals": torch.as_tensor(a["next_goals"][start:stop]).float(),
            "dones": torch.as_tensor(a["dones"][start:stop]).float(),
        }
        if "constraints" in a:
            row["constraints"] = torch.as_tensor(a["constraints"][start:stop]).float()
        if "sample_weights" in a:
            row["sample_weights"] = torch.as_tensor(a["sample_weights"][start:stop]).float()
        return row


@dataclass(frozen=True)
class GameBeliefTrainMetrics:
    loss: float
    dynamics_nll: float
    reward_mse: float
    continuation_bce: float
    value_mse: float
    reconstruction_mse: float
    goal_mse: float
    risk_bce: float


class GameBeliefWorldTrainer:
    def __init__(
        self,
        encoder: GameBeliefEncoder,
        world: MultimodalWorldModel,
        *,
        lr: float = 5e-4,
        gamma: float = .99,
        value_weight: float = .25,
        reconstruction_weight: float = .5,
        goal_weight: float = .25,
        risk_weight: float = .25,
        precision: str = "fp32",
    ):
        if int(world.belief_dim) != int(encoder.belief_dim):
            raise ValueError("world/encoder belief dimensions do not match")
        self.encoder=encoder; self.world=world; self.gamma=float(gamma)
        self.value_weight=float(value_weight); self.reconstruction_weight=float(reconstruction_weight)
        self.goal_weight=float(goal_weight); self.risk_weight=float(risk_weight)
        self.precision=str(precision).lower()
        if self.precision not in {"fp32","bf16","fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        self.device_type=next(world.parameters()).device.type
        if self.precision=="fp16" and self.device_type!="cuda":
            raise ValueError("fp16 training is supported only on CUDA; use bf16 or fp32 elsewhere")
        self.optimizer=torch.optim.AdamW(list(encoder.parameters())+list(world.parameters()),lr=float(lr))
        self.scaler=torch.amp.GradScaler("cuda",enabled=(self.precision=="fp16" and self.device_type=="cuda"))
        self.global_updates=0

    def trainer_state_dict(self) -> dict:
        """Serializable optimizer/scaler state for exact cumulative milestone resume."""
        return {
            "format": "awa-v2.26-game-world-trainer-state-v1",
            "precision": self.precision,
            "device_type": self.device_type,
            "global_updates": int(self.global_updates),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }

    def load_trainer_state_dict(self, state: dict) -> None:
        if state.get("format") != "awa-v2.26-game-world-trainer-state-v1":
            raise ValueError("unsupported world trainer state format")
        if str(state.get("precision")) != self.precision:
            raise ValueError("resume world trainer precision mismatch")
        # CUDA optimizer tensors are remapped by torch.load(map_location=...) and
        # optimizer.load_state_dict preserves their target parameter association.
        self.optimizer.load_state_dict(state["optimizer"])
        if self.scaler.is_enabled():
            scaler_state = state.get("scaler")
            if scaler_state is None:
                raise ValueError("fp16 resume checkpoint is missing GradScaler state")
            self.scaler.load_state_dict(scaler_state)
        self.global_updates = int(state.get("global_updates", 0))

    def _autocast(self):
        if self.precision=="fp32": return nullcontext()
        dtype=torch.float16 if self.precision=="fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device_type,dtype=dtype,enabled=True)

    def encode_sequence(self, batch):
        device=next(self.encoder.parameters()).device
        obs=batch["observations"].to(device); goals=batch["goals"].to(device)
        actions=batch["actions"].to(device); nxt=batch["next_observations"].to(device); ng=batch["next_goals"].to(device)
        B,T,_=obs.shape
        state=self.encoder.initial(B,device)
        zero=torch.zeros(B,self.encoder.action_dim,device=device)
        state, belief=self.encoder.observe(state,obs[:,0],zero,goals[:,0],0)
        beliefs=[]; next_beliefs=[]
        for t in range(T):
            beliefs.append(belief)
            state, nb=self.encoder.observe(state,nxt[:,t],actions[:,t],ng[:,t],t+1)
            next_beliefs.append(nb)
            belief=nb
        return torch.stack(beliefs,1),torch.stack(next_beliefs,1)

    def step(self,batch):
        device=next(self.world.parameters()).device
        actions=batch["actions"].to(device); rewards=batch["rewards"].to(device); dones=batch["dones"].to(device)
        obs=batch["observations"].to(device); goals=batch["goals"].to(device)
        next_obs=batch["next_observations"].to(device); next_goals=batch["next_goals"].to(device)
        weights=batch.get("sample_weights")
        if weights is not None: weights=weights.to(device)
        with self._autocast():
            beliefs,next_beliefs=self.encode_sequence(batch)
            B,T,D=beliefs.shape
            flat_b=beliefs.reshape(B*T,D); flat_a=actions.reshape(B*T,-1); flat_n=next_beliefs.reshape(B*T,D)
            nll_rows=self.world.nll_loss(flat_b,flat_a,flat_n.detach(),reduction="none").reshape(B,T)
            _,trunk=self.world.distribution(flat_b,flat_a)
            rp=self.world.reward(trunk).reshape(B,T)
            cp=torch.sigmoid(self.world.cont(trunk)).reshape(B,T)
            reward_err=(rp-rewards).square(); cont_err=F.binary_cross_entropy(cp,1.0-dones,reduction="none")
            # Truncated return-to-go gives the terminal bootstrap head a real target.
            returns=torch.zeros_like(rewards); running=torch.zeros(B,device=device)
            for t in range(T-1,-1,-1):
                running=rewards[:,t]+self.gamma*(1.0-dones[:,t])*running; returns[:,t]=running
            vp=self.world.terminal_value(flat_b).reshape(B,T)
            value_err=(vp-returns.detach()).square()
            ro,rg=self.encoder.reconstruct(flat_b)
            rn,rng=self.encoder.reconstruct(flat_n)
            recon_err=.5*((ro-obs.reshape(B*T,-1)).square().mean(-1).reshape(B,T)+(rn-next_obs.reshape(B*T,-1)).square().mean(-1).reshape(B,T))
            goal_err=.5*((rg-goals.reshape(B*T,-1)).square().mean(-1).reshape(B,T)+(rng-next_goals.reshape(B*T,-1)).square().mean(-1).reshape(B,T))
            risk_err=torch.zeros_like(reward_err)
            if "constraints" in batch:
                labels=batch["constraints"].to(device).reshape(B*T,-1)
                logits=self.world.risk.logits(flat_b,flat_a)
                risk_err=F.binary_cross_entropy_with_logits(logits,labels,reduction="none").mean(-1).reshape(B,T)
            if weights is None:
                w=torch.ones_like(reward_err)
            else:
                w=weights / weights.mean().clamp_min(1e-6)
            def wm(x): return (x*w).mean()
            nll=wm(nll_rows); reward_mse=wm(reward_err); cont_bce=wm(cont_err); value_mse=wm(value_err)
            recon=wm(recon_err); goal_mse=wm(goal_err); risk_bce=wm(risk_err)
            loss=nll+reward_mse+.25*cont_bce+self.value_weight*value_mse+self.reconstruction_weight*recon+self.goal_weight*goal_mse+self.risk_weight*risk_bce
        self.optimizer.zero_grad(set_to_none=True)
        if self.scaler.is_enabled():
            self.scaler.scale(loss).backward(); self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.world.parameters()),50.0)
            self.scaler.step(self.optimizer); self.scaler.update()
        else:
            loss.backward(); torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters())+list(self.world.parameters()),50.0); self.optimizer.step()
        self.global_updates += 1
        return GameBeliefTrainMetrics(*(float(x.detach()) for x in (loss,nll,reward_mse,cont_bce,value_mse,recon,goal_mse,risk_bce)))

    def fit(self,dataset:Dataset,*,epochs=1,batch_size=32,shuffle=True):
        hist=[]
        loader=DataLoader(dataset,batch_size=min(int(batch_size),len(dataset)),shuffle=bool(shuffle))
        for _ in range(int(epochs)):
            for batch in loader: hist.append(self.step(batch))
        return hist

    @torch.no_grad()
    def evaluate_horizons(self,dataset:Dataset,horizons=(1,2,4),batch_size=64):
        hs=sorted({int(h) for h in horizons if int(h)>0})
        rows={h:[] for h in hs}; value=[]; reward=[]
        for batch in DataLoader(dataset,batch_size=min(int(batch_size),len(dataset)),shuffle=False):
            device=next(self.world.parameters()).device
            actions=batch["actions"].to(device); rewards=batch["rewards"].to(device); dones=batch["dones"].to(device)
            b,nb=self.encode_sequence(batch); B,T,D=b.shape
            for h in hs:
                if h>T: continue
                pred=b[:,0]
                for j in range(h): pred=self.world.imagine_step(pred,actions[:,j],deterministic=True)["belief"]
                rows[h].append((pred-nb[:,h-1]).square().mean(-1))
            _,tr=self.world.distribution(b.reshape(B*T,D),actions.reshape(B*T,-1))
            reward.append((self.world.reward(tr).reshape(B,T)-rewards).square())
            returns=torch.zeros_like(rewards); running=torch.zeros(B,device=device)
            for t in range(T-1,-1,-1):
                running=rewards[:,t]+self.gamma*(1-dones[:,t])*running; returns[:,t]=running
            value.append((self.world.terminal_value(b.reshape(B*T,D)).reshape(B,T)-returns).square())
        return {
            "horizon_rmse":{str(h):float(torch.cat(rows[h]).mean().sqrt()) if rows[h] else float("nan") for h in hs},
            "reward_rmse":float(torch.cat([x.reshape(-1) for x in reward]).mean().sqrt()),
            "value_rmse":float(torch.cat([x.reshape(-1) for x in value]).mean().sqrt()),
        }

    def calibrate_uncertainty(self,dataset:Dataset,*,epochs=2,batch_size=64,lr=1e-3):
        opt=torch.optim.Adam(self.world.uncertainty.parameters(),lr=float(lr)); last=0.0
        hs=tuple(self.world.uncertainty.horizons)
        loader=DataLoader(dataset,batch_size=min(int(batch_size),len(dataset)),shuffle=True)
        for _ in range(int(epochs)):
            for batch in loader:
                device=next(self.world.parameters()).device; actions=batch["actions"].to(device)
                with torch.no_grad(): b,nb=self.encode_sequence(batch)
                errs=[]
                for h in hs:
                    hh=min(int(h),b.shape[1]); pred=b[:,0]
                    for j in range(hh): pred=self.world.imagine_step(pred,actions[:,j],deterministic=True)["belief"]
                    errs.append((pred-nb[:,hh-1]).square().mean(-1).sqrt())
                target=torch.stack(errs,-1)
                loss=self.world.uncertainty.calibration_loss(b[:,0].detach(),target)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); last=float(loss.detach())
        return last


@torch.no_grad()
def encode_game_transitions(dataset: OfflineTransitionDataset, encoder: GameBeliefEncoder):
    """Encode transitions sequentially, resetting temporal state at episode boundaries."""
    a=dataset.arrays; device=next(encoder.parameters()).device
    states=[]; next_states=[]; prev_done=True; temporal=None; belief=None; step=0
    for i in range(len(dataset)):
        if prev_done:
            temporal=encoder.initial(1,device); step=0
            zero=torch.zeros(1,encoder.action_dim,device=device)
            temporal,belief=encoder.observe(temporal,a["observations"][i],zero,a["goals"][i],step)
        states.append(belief.squeeze(0).cpu().numpy())
        action=torch.as_tensor(a["actions"][i],dtype=torch.float32,device=device).unsqueeze(0)
        temporal,next_belief=encoder.observe(temporal,a["next_observations"][i],action,a["next_goals"][i],step+1)
        next_states.append(next_belief.squeeze(0).cpu().numpy())
        belief=next_belief; step+=1; prev_done=bool(a["dones"][i])
    return np.asarray(states,np.float32),np.asarray(next_states,np.float32)
