from __future__ import annotations

from dataclasses import dataclass, asdict
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class SkillContractDecision:
    precondition_probability: float
    termination_probability: float
    failure_probability: float
    can_start: bool
    should_stop: bool
    failed: bool

    def to_dict(self):
        return asdict(self)


class SkillContractModel(nn.Module):
    """Learned precondition / termination / failure model for a reusable skill."""

    def __init__(self, state_dim: int, goal_dim: int = 0, hidden: int = 64):
        super().__init__()
        self.state_dim = int(state_dim)
        self.goal_dim = int(goal_dim)
        self.hidden = int(hidden)
        if self.state_dim <= 0 or self.goal_dim < 0 or self.hidden <= 0:
            raise ValueError("invalid skill contract dimensions")
        self.net = nn.Sequential(
            nn.Linear(self.state_dim + self.goal_dim, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, self.hidden),
            nn.SiLU(),
            nn.Linear(self.hidden, 3),
        )

    def forward(self, state: torch.Tensor, goal: torch.Tensor | None = None) -> torch.Tensor:
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.ndim != 2 or state.shape[-1] != self.state_dim:
            raise ValueError(f"state must have shape [B,{self.state_dim}]")
        if self.goal_dim:
            if goal is None:
                raise ValueError("goal is required for this contract model")
            goal = torch.as_tensor(goal, dtype=torch.float32, device=device)
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
        return self.net(x)

    @torch.no_grad()
    def decision(
        self,
        state,
        goal=None,
        *,
        precondition_threshold: float = 0.5,
        termination_threshold: float = 0.5,
        failure_threshold: float = 0.5,
    ) -> SkillContractDecision:
        for name, value in {
            "precondition_threshold": precondition_threshold,
            "termination_threshold": termination_threshold,
            "failure_threshold": failure_threshold,
        }.items():
            if not 0 <= float(value) <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        p = torch.sigmoid(self.forward(state, goal))[0].cpu().numpy()
        return SkillContractDecision(
            float(p[0]),
            float(p[1]),
            float(p[2]),
            bool(p[0] >= precondition_threshold),
            bool(p[1] >= termination_threshold),
            bool(p[2] >= failure_threshold),
        )


class SkillContractTrainer:
    def __init__(self, model: SkillContractModel, lr: float = 3e-4):
        self.model = model
        self.opt = torch.optim.Adam(model.parameters(), lr=float(lr))

    def fit(self, states, labels, goals=None, *, epochs: int = 10, batch_size: int = 128, seed: int = 0) -> dict:
        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32)
        y = torch.as_tensor(np.asarray(labels), dtype=torch.float32)
        if x.ndim != 2 or x.shape[1] != self.model.state_dim:
            raise ValueError("states do not match contract state dimension")
        if y.ndim != 2 or y.shape != (len(x), 3):
            raise ValueError("labels must have shape [N,3] for precondition/termination/failure")
        if torch.any(~torch.isfinite(y)) or torch.any((y < 0) | (y > 1)):
            raise ValueError("contract labels must be finite probabilities in [0,1]")
        g = None if goals is None else torch.as_tensor(np.asarray(goals), dtype=torch.float32)
        if self.model.goal_dim:
            if g is None or g.ndim != 2 or g.shape != (len(x), self.model.goal_dim):
                raise ValueError("goals do not match contract goal dimension/batch")
        elif g is not None:
            raise ValueError("goals supplied to a goal-free contract")
        rng = np.random.default_rng(seed)
        losses = []
        for _ in range(epochs):
            order = rng.permutation(len(x))
            for start in range(0, len(x), batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                ii = torch.as_tensor(idx, dtype=torch.long)
                logits = self.model(x[ii], None if g is None else g[ii])
                loss = nn.functional.binary_cross_entropy_with_logits(logits, y[ii].to(logits.device))
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
                losses.append(float(loss.detach()))
        return {
            "steps": len(losses),
            "final_loss": float(losses[-1]),
            "mean_loss": float(np.mean(losses)),
        }


class SkillContractRegistry:
    def __init__(self):
        self._models: dict[str, SkillContractModel] = {}

    def attach(self, skill_name: str, model: SkillContractModel) -> None:
        self._models[str(skill_name)] = model

    def get(self, skill_name: str) -> SkillContractModel | None:
        return self._models.get(str(skill_name))

    def check(self, skill_name: str, state, goal=None, **thresholds) -> SkillContractDecision:
        model = self.get(skill_name)
        if model is None:
            raise KeyError(f"no learned contract for skill {skill_name}")
        return model.decision(state, goal, **thresholds)


class SkillExecutionManager:
    """Combine symbolic skill applicability with learned execution contracts."""

    def __init__(self, registry, contracts: SkillContractRegistry | None = None):
        self.registry = registry
        self.contracts = contracts or SkillContractRegistry()

    def available(self, facts: set[str], state, goal=None):
        out = []
        for skill in self.registry.applicable(set(facts)):
            model = self.contracts.get(skill.name)
            if model is None:
                out.append(skill)
                continue
            decision = model.decision(state, goal)
            if decision.can_start and not decision.failed:
                out.append(skill)
        return out

    def status(self, skill_name: str, state, goal=None) -> SkillContractDecision | None:
        model = self.contracts.get(skill_name)
        return None if model is None else model.decision(state, goal)
