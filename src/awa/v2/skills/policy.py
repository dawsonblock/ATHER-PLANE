from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn


class DistilledContinuousSkill(nn.Module):
    """Goal-optional bounded continuous policy for one reusable skill."""

    def __init__(self, state_dim: int, action_dim: int, low, high, goal_dim: int = 0, hidden: int = 128):
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.action_dim = int(action_dim)
        self.hidden = int(hidden)
        if self.state_dim <= 0 or self.action_dim <= 0 or self.goal_dim < 0 or self.hidden <= 0:
            raise ValueError("skill dimensions/hidden width must be positive")
        low = torch.as_tensor(low, dtype=torch.float32).reshape(-1)
        high = torch.as_tensor(high, dtype=torch.float32).reshape(-1)
        if (
            low.numel() != self.action_dim
            or high.numel() != self.action_dim
            or torch.any(high <= low)
            or torch.any(~torch.isfinite(low))
            or torch.any(~torch.isfinite(high))
        ):
            raise ValueError("invalid continuous action bounds")
        self.register_buffer("low", low)
        self.register_buffer("high", high)
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.action_dim),
        )

    def forward(self, state, goal=None):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.low.device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape [B,{self.state_dim}]")
        if self.goal_dim:
            if goal is None:
                raise ValueError("goal is required")
            goal = torch.as_tensor(goal, dtype=torch.float32, device=state.device)
            if goal.ndim == 1:
                goal = goal.unsqueeze(0)
            if goal.ndim != 2 or goal.shape[-1] != self.goal_dim:
                raise ValueError(f"goal must have shape [B,{self.goal_dim}]")
            if goal.shape[0] == 1 and state.shape[0] > 1:
                goal = goal.expand(state.shape[0], -1)
            if goal.shape[0] != state.shape[0]:
                raise ValueError("state and goal batch sizes must match")
            x = torch.cat([state, goal], dim=-1)
        else:
            x = state
        unit = torch.tanh(self.net(x))
        return self.low + (unit + 1.0) * 0.5 * (self.high - self.low)

    @torch.no_grad()
    def act(self, state, goal=None):
        return self.forward(state, goal)


@dataclass(frozen=True)
class SkillPolicyExample:
    state: np.ndarray
    action: np.ndarray
    goal: np.ndarray | None = None
    weight: float = 1.0


class SkillPolicyBuffer:
    def __init__(self, capacity_per_skill: int = 100_000):
        self.capacity_per_skill = int(capacity_per_skill)
        if self.capacity_per_skill <= 0:
            raise ValueError("capacity_per_skill must be > 0")
        self._items = defaultdict(list)

    def add(self, skill_name: str, state, action, goal=None, weight: float = 1.0):
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("skill example weight must be finite and non-negative")
        rows = self._items[str(skill_name)]
        rows.append(
            SkillPolicyExample(
                np.asarray(state, dtype=np.float32).reshape(-1),
                np.asarray(action, dtype=np.float32).reshape(-1),
                None if goal is None else np.asarray(goal, dtype=np.float32).reshape(-1),
                weight,
            )
        )
        if len(rows) > self.capacity_per_skill:
            del rows[: len(rows) - self.capacity_per_skill]

    def add_trajectory(self, skill_name: str, states, actions, goal=None, weight: float = 1.0):
        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        if states.ndim < 2 or actions.ndim < 2:
            raise ValueError("states/actions must be batched arrays")
        if len(states) < len(actions):
            raise ValueError("states must have at least as many rows as actions")
        goal_array = None if goal is None else np.asarray(goal, dtype=np.float32)
        per_step_goal = (
            goal_array is not None
            and goal_array.ndim > 1
            and len(goal_array) == len(actions)
        )
        for i in range(len(actions)):
            g = goal_array[i] if per_step_goal else goal
            self.add(skill_name, states[i], actions[i], g, weight)

    def arrays(self, skill_name: str):
        rows = self._items.get(str(skill_name), [])
        if not rows:
            raise ValueError(f"no policy examples for skill {skill_name}")
        states = np.stack([r.state for r in rows])
        actions = np.stack([r.action for r in rows])
        weights = np.asarray([r.weight for r in rows], dtype=np.float32)
        if rows[0].goal is None:
            if any(r.goal is not None for r in rows):
                raise ValueError("mixed goal-conditioned and goal-free skill examples")
            goals = None
        else:
            if any(r.goal is None for r in rows):
                raise ValueError("mixed goal-conditioned and goal-free skill examples")
            goals = np.stack([r.goal for r in rows])
        return states, actions, goals, weights

    def __len__(self):
        return sum(len(v) for v in self._items.values())


class SkillPolicyTrainer:
    def __init__(self, model: DistilledContinuousSkill, lr: float = 3e-4):
        self.model = model
        self.opt = torch.optim.AdamW(model.parameters(), lr=float(lr))

    def fit(
        self,
        states,
        actions,
        goals=None,
        weights=None,
        *,
        epochs: int = 20,
        batch_size: int = 128,
        seed: int = 0,
    ):
        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        x = np.asarray(states, dtype=np.float32)
        a = np.asarray(actions, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.model.state_dim:
            raise ValueError("states do not match skill state dimension")
        if a.ndim != 2 or a.shape[1] != self.model.action_dim or len(a) != len(x):
            raise ValueError("actions do not match skill action dimension/batch")
        g = None if goals is None else np.asarray(goals, dtype=np.float32)
        if self.model.goal_dim:
            if g is None or g.ndim != 2 or g.shape != (len(x), self.model.goal_dim):
                raise ValueError("goals do not match skill goal dimension/batch")
        elif g is not None:
            raise ValueError("goals supplied to a goal-free skill")
        w = np.ones(len(x), dtype=np.float32) if weights is None else np.asarray(weights, dtype=np.float32)
        if w.shape != (len(x),) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite non-negative [N]")
        rng = np.random.default_rng(seed)
        hist = []
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                pred = self.model(x[idx], None if g is None else g[idx])
                target = torch.as_tensor(a[idx], dtype=torch.float32, device=pred.device)
                weight = torch.as_tensor(w[idx], dtype=torch.float32, device=pred.device)
                loss = (torch.square(pred - target).mean(dim=-1) * weight).mean()
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                hist.append(float(loss.detach()))
        return {
            "steps": len(hist),
            "final_loss": float(hist[-1]),
            "mean_loss": float(np.mean(hist)),
            "examples": len(x),
        }


class ExecutableSkillLibrary:
    """Runtime library tying qualified SkillSpecs to executable policies/contracts."""

    def __init__(self, registry, contracts=None):
        from .contracts import SkillContractRegistry

        self.registry = registry
        self.contracts = contracts or SkillContractRegistry()
        self.policies: dict[str, DistilledContinuousSkill] = {}

    def attach_policy(self, skill_name: str, policy: DistilledContinuousSkill):
        if self.registry.get(skill_name) is None:
            raise KeyError(f"skill is not qualified in registry: {skill_name}")
        self.policies[str(skill_name)] = policy

    def available(self, facts: set[str], state, goal=None):
        out = []
        for skill in self.registry.applicable(facts):
            if skill.name not in self.policies:
                continue
            contract = self.contracts.get(skill.name)
            if contract is not None:
                d = contract.decision(state, goal)
                if not d.can_start or d.failed:
                    continue
            out.append(skill)
        return out

    def act(self, skill_name: str, state, goal=None):
        if skill_name not in self.policies:
            raise KeyError(f"no executable policy for skill {skill_name}")
        contract = self.contracts.get(skill_name)
        if contract is not None:
            d = contract.decision(state, goal)
            if not d.can_start or d.failed:
                raise RuntimeError(f"skill contract rejected execution: {skill_name}")
        return self.policies[skill_name].act(state, goal)
