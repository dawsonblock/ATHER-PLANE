from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import torch

from awa.v2.planners import PolicySeededMPPI
from .procedural_arena import ACTION_DIM
from .procedural_runtime import load_procedural_game_stack
from .teacher import LogicalArenaTeacher, RandomArenaPolicy


class CoverageArenaPolicy:
    """Observation-agnostic low-discrepancy explorer for matched collection baselines.

    It deliberately does not inspect hidden objective or simulator state. Movement follows
    a golden-angle sweep with seeded jitter; attack/interact are sparse deterministic probes.
    This provides broader state coverage than iid random actions without embedding task logic.
    """

    def __init__(self, seed: int = 0, *, probe_period: int = 11):
        self.seed = int(seed)
        self.probe_period = max(3, int(probe_period))
        self._rng = np.random.default_rng(self.seed)
        self._step = 0
        self._phase = 0.0

    def reset_episode(self, env=None):
        task_seed = int(getattr(getattr(env, "task", None), "seed", 0)) if env is not None else 0
        self._rng = np.random.default_rng(self.seed ^ (task_seed * 0x9E3779B1))
        self._step = 0
        self._phase = float(self._rng.uniform(-math.pi, math.pi))

    def act(self, env, observation=None) -> np.ndarray:
        # Golden-angle sweep avoids repeatedly revisiting the same few headings.
        angle = self._phase + self._step * 2.399963229728653
        angle += float(self._rng.normal(0.0, 0.08))
        move = np.asarray([math.cos(angle), math.sin(angle)], dtype=np.float32)
        attack = 1.0 if self._step % self.probe_period == 0 else 0.0
        interact = 1.0 if self._step % (self.probe_period + 4) == 2 else 0.0
        self._step += 1
        return np.asarray([move[0], move[1], attack, interact], dtype=np.float32)

    def receipt(self) -> dict:
        return {"type": "coverage", "seed": self.seed, "probe_period": self.probe_period}


class FrozenAetherArenaPolicy:
    """Frozen Aether actor with optional bounded policy-seeded planning for data collection.

    The collector never trains or mutates its source checkpoint. It maintains temporal belief
    per episode and can invoke a small MPPI search at a fixed stride. This makes autonomous
    collection directly comparable with teacher/coverage collection under the same dataset ABI.
    """

    def __init__(
        self,
        world_checkpoint: str | Path,
        actor_checkpoint: str | Path,
        *,
        device: str | torch.device = "cpu",
        use_planner: bool = False,
        planner_budget: int = 16,
        planner_horizon: int = 4,
        planner_stride: int = 4,
    ):
        self.world_checkpoint = str(world_checkpoint)
        self.actor_checkpoint = str(actor_checkpoint)
        self.runtime = load_procedural_game_stack(world_checkpoint, actor_checkpoint, device=device)
        self.use_planner = bool(use_planner)
        self.planner_budget = max(1, int(planner_budget))
        self.planner_horizon = max(1, int(planner_horizon))
        self.planner_stride = max(1, int(planner_stride))
        self.planner = None
        if self.use_planner:
            self.planner = PolicySeededMPPI(
                self.runtime.world,
                self.runtime.actor.actor,
                [-1.0] * ACTION_DIM,
                [1.0] * ACTION_DIM,
                horizon=self.planner_horizon,
                candidates=self.planner_budget,
            )
        self._temporal = None
        self._prev_action = None
        self._step = 0
        self.planner_calls = 0
        self.world_model_calls = 0

    def reset_episode(self, env=None):
        self._temporal = self.runtime.encoder.initial(1, self.runtime.device)
        self._prev_action = torch.zeros(1, ACTION_DIM, device=self.runtime.device)
        self._step = 0

    @torch.no_grad()
    def act(self, env, observation=None) -> np.ndarray:
        if self._temporal is None:
            self.reset_episode(env)
        self._temporal, belief = self.runtime.encoder.observe(
            self._temporal,
            observation,
            self._prev_action,
            env.goal_vector(),
            self._step,
        )
        action = self.runtime.actor.actor.deterministic_action(belief)
        if self.planner is not None and self._step % self.planner_stride == 0:
            result = self.planner.plan(
                belief,
                actor=self.runtime.actor.actor,
                budget=self.planner_budget,
            )
            action = result.action
            self.planner_calls += 1
            self.world_model_calls += int(getattr(result, "world_model_calls", 0))
        arr = action.squeeze(0).detach().cpu().numpy().astype(np.float32)
        # The procedural ABI has binary attack/interact controls. Keep collected actions
        # physically meaningful even though the generic planner searches continuous vectors.
        arr[2:] = (arr[2:] > 0.0).astype(np.float32)
        self._prev_action = torch.as_tensor(arr, dtype=torch.float32, device=self.runtime.device).unsqueeze(0)
        self._step += 1
        return arr

    def receipt(self) -> dict:
        return {
            "type": "aether_planner" if self.use_planner else "aether_actor",
            "planner_budget": self.planner_budget if self.use_planner else 0,
            "planner_horizon": self.planner_horizon if self.use_planner else 0,
            "planner_stride": self.planner_stride if self.use_planner else 0,
            "planner_calls": int(self.planner_calls),
            "world_model_calls": int(self.world_model_calls),
        }


def collector_receipt(policy) -> dict:
    if hasattr(policy, "receipt"):
        out = dict(policy.receipt())
    elif isinstance(policy, LogicalArenaTeacher):
        out = {"type": "teacher"}
    elif isinstance(policy, RandomArenaPolicy):
        out = {"type": "random"}
    else:
        out = {"type": policy.__class__.__name__}
    return out
