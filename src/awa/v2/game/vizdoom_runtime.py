from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from awa.v2.world import MultimodalWorldModel, RiskConstraintModel
from awa.v2.safety import SafeActionGuard
from awa.v2.planners import HybridRiskAwarePolicySeededMPPI
from awa.v2.video_representation import StreamingGameFeatureEncoder, VideoClipSpec
from .belief import GameBeliefEncoder
from .hybrid_control import HybridOfflineActorCriticBaseline
from .vizdoom_env import VIZDOOM_ACTION_DIM, VIZDOOM_GOAL_DIM, VIZDOOM_TELEMETRY_DIM


@dataclass(frozen=True)
class ViZDoomDecisionInfo:
    source: str
    risk: float
    guard_reason: str
    planner_calls: int
    belief_norm: float

    def to_dict(self): return asdict(self)


class ViZDoomRuntimeController:
    """Canonical online v2.16 Doom control path.

    Structured mode feeds telemetry directly into the temporal encoder. Pixel mode
    maintains the same causal V-JEPA clip contract used by the offline cache and
    concatenates telemetry before entering the temporal belief model.
    """

    def __init__(
        self,
        encoder: GameBeliefEncoder,
        world: MultimodalWorldModel,
        actor: HybridOfflineActorCriticBaseline,
        *,
        track: str,
        video_backbone: torch.nn.Module | None = None,
        clip_spec: VideoClipSpec | None = None,
        use_planner: bool = True,
        planner_budget: int = 64,
        risk_limit: float = 0.85,
        strict_recovery: bool = True,
    ):
        if track not in {"structured","pixel"}: raise ValueError("track must be structured or pixel")
        self.encoder=encoder; self.world=world; self.actor=actor; self.track=track
        self.use_planner=bool(use_planner); self.planner_budget=int(planner_budget)
        self.device=next(encoder.parameters()).device
        self.video = None
        if track=="pixel":
            if video_backbone is None: raise ValueError("pixel runtime requires a V-JEPA-compatible video backbone")
            self.video=StreamingGameFeatureEncoder(video_backbone,clip_spec or VideoClipSpec(),telemetry_dim=VIZDOOM_TELEMETRY_DIM)
        self.planner=HybridRiskAwarePolicySeededMPPI(world,actor.actor,horizon=8,candidates=max(8,self.planner_budget),samples=4) if self.use_planner else None
        self.guard=SafeActionGuard(world.risk,[-1]*VIZDOOM_ACTION_DIM,[1]*VIZDOOM_ACTION_DIM,risk_limit=risk_limit,recovery_action=[0.0,0.0,-1.0,-1.0],strict_recovery=strict_recovery)
        self.reset()

    def reset(self):
        self.temporal=self.encoder.initial(1,self.device)
        self.prev_action=torch.zeros(1,VIZDOOM_ACTION_DIM,device=self.device)
        self.step_index=0
        if self.video is not None: self.video.reset()

    def _feature(self, env, observation) -> torch.Tensor:
        if self.track=="structured":
            feat=torch.as_tensor(observation,dtype=torch.float32,device=self.device).reshape(1,-1)
        else:
            feat=self.video.step(observation,env.telemetry_vector()).to(self.device)
        if feat.shape[-1] != self.encoder.obs_dim:
            raise ValueError(f"runtime observation feature dim {feat.shape[-1]} != trained encoder dim {self.encoder.obs_dim}")
        return feat

    @torch.no_grad()
    def act(self, env, observation, *, force_planner: bool = False):
        feature=self._feature(env,observation)
        goal=torch.as_tensor(env.goal_vector(),dtype=torch.float32,device=self.device).reshape(1,-1)
        self.temporal,belief=self.encoder.observe(self.temporal,feature,self.prev_action,goal,self.step_index)
        planner_calls=0; source="actor"
        if self.planner is not None and force_planner:
            result=self.planner.plan(belief,budget=self.planner_budget)
            proposal=result.action; planner_calls=int(result.world_model_calls); source="planner"
        else:
            proposal=self.actor.action(belief)
        guarded=self.guard.validate(belief,proposal)
        action=guarded.action.clone()
        # Preserve explicit categorical ABI after clipping/recovery.
        action[:,2:4]=torch.where(action[:,2:4]>=0,torch.ones_like(action[:,2:4]),-torch.ones_like(action[:,2:4]))
        self.prev_action=action.detach(); self.step_index+=1
        info=ViZDoomDecisionInfo(source,float(guarded.risk),guarded.reason,planner_calls,float(belief.norm().item()))
        return action.squeeze(0).cpu().numpy().astype(np.float32),info


def load_vizdoom_runtime(
    world_checkpoint: str | Path,
    actor_checkpoint: str | Path,
    *,
    device: str = "cpu",
    video_backbone: torch.nn.Module | None = None,
    clip_spec: VideoClipSpec | None = None,
    use_planner: bool = True,
    planner_budget: int = 64,
    risk_limit: float = 0.85,
) -> ViZDoomRuntimeController:
    wc=torch.load(world_checkpoint,map_location=device,weights_only=False)
    ac=torch.load(actor_checkpoint,map_location=device,weights_only=False)
    if wc.get("format") != "awa-v2.16-vizdoom-world-v1": raise ValueError("unsupported ViZDoom world checkpoint format")
    if ac.get("format") != "awa-v2.16-vizdoom-hybrid-actor-v1": raise ValueError("unsupported ViZDoom actor checkpoint format")
    cfg=wc["encoder_config"]
    encoder=GameBeliefEncoder(**cfg).to(device); encoder.load_state_dict(wc["encoder"])
    world=MultimodalWorldModel(int(wc["belief_dim"]),int(wc["action_dim"]),hidden=int(wc["hidden"]),components=3,horizons=tuple(wc["horizons"])).to(device)
    risk_hidden=int(wc.get("risk_hidden", wc["world"]["risk.net.0.weight"].shape[0]))
    risk_constraints=int(wc.get("risk_constraints", wc["world"]["risk.net.4.weight"].shape[0]))
    world.risk=RiskConstraintModel(int(wc["belief_dim"]),int(wc["action_dim"]),hidden=risk_hidden,constraints=risk_constraints).to(device)
    world.load_state_dict(wc["world"])
    actor=HybridOfflineActorCriticBaseline(int(ac["state_dim"]),hidden=int(ac["hidden"]),device=device)
    actor.load_state_dict(ac["state"],load_optimizers=False)
    encoder.eval(); world.eval(); actor.actor.eval(); actor.critic.eval()
    return ViZDoomRuntimeController(encoder,world,actor,track=str(wc["track"]),video_backbone=video_backbone,clip_spec=clip_spec,use_planner=use_planner,planner_budget=planner_budget,risk_limit=risk_limit)
