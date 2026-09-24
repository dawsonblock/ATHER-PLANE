from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import time
import numpy as np
import torch

from awa.v2.branching import default_voc_features
from awa.v2.reasoning.effort import AdaptiveReasoningEffortController, EffortSignals
from .procedural_arena import ACTION_DIM




@dataclass(frozen=True)
class EffortRuntimeTrace:
    level: str
    planner: str
    budget: int
    predicted_gain: float
    utility: float
    uncertainty: float
    risk: float
    world_model_calls: int
    latency_ms: float
    logical_world_model_transitions: int = 0
    physical_world_model_forwards: int = 0
    max_world_batch_size: int = 0

    def to_dict(self):
        return asdict(self)


def facts_from_env(env) -> set[str]:
    facts={"alive" if env.health>0 else "dead"}
    if env.has_key: facts.add("has_key")
    if env.door_open: facts.add("door_open")
    if env.enemy_present: facts.add("enemy_visible")
    else: facts.add("enemy_defeated")
    if env.object_present: facts.add("object_visible")
    if env.cover_reached: facts.add("in_cover")
    if env.ammo>0.15: facts.add("has_ammo")
    facts.add(f"objective:{env.active_objective}")
    return facts


class AdaptiveGamePolicy:
    """Runtime hierarchy: qualified skill -> actor -> VOC-selected planner.

    The wrapper maintains Aether's temporal belief across an episode. It is suitable
    for transfer evaluation and exact game rollouts; the curriculum stage is never
    consulted when selecting actions.
    """
    def __init__(
        self, encoder, actor_baseline, world, planners=None, voc=None, skills=None, device=None, adapt_fn=None,
        *, effort_controller: AdaptiveReasoningEffortController | None = None, effort_gain_predictor=None,
        effort_signal_fn=None, model_reliability: float = 1.0, resource_pressure: float = 0.0,
    ):
        self.encoder=encoder; self.actor_baseline=actor_baseline; self.actor=actor_baseline.actor
        self.world=world; self.planners=dict(planners or {}); self.voc=voc; self.skills=skills; self.adapt_fn=adapt_fn
        self.device=torch.device(device or next(encoder.parameters()).device)
        self.effort_controller=effort_controller; self.effort_gain_predictor=effort_gain_predictor
        self.effort_signal_fn=effort_signal_fn
        self.model_reliability=float(model_reliability); self.resource_pressure=float(resource_pressure)
        if not 0.0 <= self.model_reliability <= 1.0:
            raise ValueError("model_reliability must lie in [0,1]")
        if not 0.0 <= self.resource_pressure <= 1.0:
            raise ValueError("resource_pressure must lie in [0,1]")
        if self.effort_controller is not None and self.voc is None and self.effort_gain_predictor is None:
            raise ValueError("adaptive effort requires voc or effort_gain_predictor")
        self._task=None; self._temporal=None; self._prev_action=None; self._step=0; self.last_belief=None
        self.last_effort_trace: EffortRuntimeTrace | None = None

    def reset_task(self, task=None):
        self._task=task; self._temporal=self.encoder.initial(1,self.device)
        self._prev_action=torch.zeros(1,ACTION_DIM,device=self.device); self._step=0; self.last_belief=None
        self.last_effort_trace=None

    def reset_episode(self, task=None): self.reset_task(task)
    def begin_task(self, task, preserve_context=False):
        # Context may persist across exposures of *this* task, but never across
        # different tasks. evaluate_game_policy_transfer() calls begin_task once
        # per task, so begin_task is the task-boundary reset regardless of mode.
        self.reset_task(task)
    def end_task(self, task=None): return None

    def _belief(self, env, obs):
        if self._temporal is None: self.reset_task(env.task)
        self._temporal,b=self.encoder.observe(self._temporal,obs,self._prev_action,env.goal_vector(),self._step)
        self.last_belief=b
        return b

    def _skill_action(self, env, belief):
        if self.skills is None: return None
        facts=facts_from_env(env)
        for skill in self.skills.available(facts,belief.squeeze(0),env.goal_vector()):
            model=self.skills.policies.get(skill.name)
            if model is None or model.state_dim!=belief.shape[-1]: continue
            try:
                a=self.skills.act(skill.name,belief.squeeze(0),env.goal_vector())
            except Exception:
                continue
            return np.asarray(a.detach().cpu()).reshape(-1),skill.name
        return None

    def _effort_gains(self, features):
        levels=self.effort_controller.levels
        if self.effort_gain_predictor is not None:
            out=self.effort_gain_predictor(features, levels)
            return np.asarray(out,dtype=np.float32).reshape(-1)
        if self.voc is None or not hasattr(self.voc,"gains") or not hasattr(self.voc,"choices"):
            raise ValueError("VOC must expose gains() and choices for adaptive effort")
        with torch.no_grad():
            raw=self.voc.gains(features).detach().cpu().numpy().reshape(-1)
        by_choice={(str(name),int(budget)):float(raw[i]) for i,(name,budget) in enumerate(self.voc.choices)}
        # Levels absent from the trained VOC are not guessed. They remain effectively
        # unavailable until branch data has trained that exact planner/budget choice.
        return np.asarray([by_choice.get((lvl.planner,int(lvl.budget)),-1e6) for lvl in levels],dtype=np.float32)

    def _effort_signals(self, env, belief, actor_action, features, gains):
        if self.effort_signal_fn is not None:
            out=self.effort_signal_fn(env,belief,actor_action,features,gains)
            if not isinstance(out,EffortSignals):
                raise TypeError("effort_signal_fn must return EffortSignals")
            return out
        feat=np.asarray(features.detach().cpu() if isinstance(features,torch.Tensor) else features,dtype=np.float32).reshape(-1)
        uncertainty=float(max(0.0,np.max(feat[:3]))) if feat.size>=3 else 0.0
        risk=float(np.clip(feat[3],0.0,1.0)) if feat.size>=4 else 0.0
        task=getattr(env,"task",None)
        difficulty=float(np.clip(getattr(task,"difficulty",0.5),0.0,1.0))
        novelty_class=str(getattr(task,"novelty_class","train"))
        novelty=0.0 if novelty_class in {"train","game_train"} or "train" in novelty_class else 0.75
        finite=np.asarray(gains,dtype=np.float64)
        historical=float(max(0.0,np.max(finite[finite>-1e5]))) if np.any(finite>-1e5) else 0.0
        return EffortSignals(
            uncertainty=uncertainty, novelty=novelty, risk=risk, actor_confidence=0.5,
            historical_planner_benefit=historical, goal_difficulty=difficulty,
            model_reliability=self.model_reliability, resource_pressure=self.resource_pressure,
        )

    @torch.no_grad()
    def act(self, env, observation):
        belief=self._belief(env,observation)
        skill=self._skill_action(env,belief)
        if skill is not None:
            action,name=skill; self._prev_action=torch.as_tensor(action,dtype=torch.float32,device=self.device).unsqueeze(0); self._step+=1
            return action,False
        actor_action=self.actor.deterministic_action(belief)
        action=actor_action; used=False; self.last_effort_trace=None
        if self.effort_controller is not None and self.planners:
            feats=torch.as_tensor([default_voc_features(self.world,belief,actor_action)],dtype=torch.float32,device=self.device)
            gains=self._effort_gains(feats)
            signals=self._effort_signals(env,belief,actor_action,feats,gains)
            decision=self.effort_controller.choose(signals,gains)
            calls=0; logical=0; physical=0; max_batch=0; t0=time.perf_counter()
            if decision.level.planner!="actor" and decision.level.planner in self.planners:
                result=self.planners[decision.level.planner].plan(belief,actor=self.actor,budget=decision.level.budget)
                action=result.action; used=True; calls=int(getattr(result,"world_model_calls",0))
                meta=dict(getattr(result,"metadata",{}) or {})
                logical=int(meta.get("logical_world_model_transitions",calls))
                physical=int(meta.get("physical_world_model_forwards",0))
                max_batch=int(meta.get("max_batch_size",0))
            latency=(time.perf_counter()-t0)*1000.0
            self.last_effort_trace=EffortRuntimeTrace(
                decision.level.name,decision.level.planner,int(decision.level.budget),
                float(decision.expected_gain),float(decision.utility),float(signals.uncertainty),
                float(signals.risk),calls,float(latency),logical,physical,max_batch,
            )
        elif self.voc is not None and self.planners:
            feats=torch.as_tensor([default_voc_features(self.world,belief,actor_action)],dtype=torch.float32,device=self.device)
            choice,_=self.voc.choose(feats)
            c=choice[0]
            if c.planner!="actor" and c.voc>0 and c.planner in self.planners:
                result=self.planners[c.planner].plan(belief,actor=self.actor,budget=c.budget)
                action=result.action; used=True
        arr=action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        self._prev_action=torch.as_tensor(arr,dtype=torch.float32,device=self.device).unsqueeze(0); self._step+=1
        return arr,used

    @torch.no_grad()
    def predict_next(self, env, observation, action):
        if self.last_belief is None: return None
        a=torch.as_tensor(action,dtype=torch.float32,device=self.device)
        if a.ndim==1: a=a.unsqueeze(0)
        imagined=self.world.imagine_step(self.last_belief,a,deterministic=True)["belief"]
        pred_obs,_=self.encoder.reconstruct(imagined)
        return pred_obs.squeeze(0).detach().cpu().numpy()

    def adapt(self, task, trajectory):
        if self.adapt_fn is not None:
            return self.adapt_fn(task,trajectory,self)
        return None
