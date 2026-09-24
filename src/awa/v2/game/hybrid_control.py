from __future__ import annotations

import copy
from dataclasses import dataclass, asdict
from typing import Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from awa.planning.actor import TanhGaussianActor
from awa.training.continuous_critic import TwinQCritic, TargetTwinQCritic


class HybridGameActor(nn.Module):
    """Two continuous movement controls plus categorical attack/interact.

    The public action ABI remains four floats for compatibility:
      [turn, forward, attack_sign, interact_sign]
    where the last two entries are exactly -1 or +1 at inference time.
    During actor optimization ``soft_action`` uses tanh(logits) for those entries so
    critic gradients can reach the categorical head without inventing fractional
    actions in the environment/world-model training data.
    """

    def __init__(self, state_dim: int, hidden: int = 256, movement_low: Iterable[float] = (-1.0, -1.0), movement_high: Iterable[float] = (1.0, 1.0)):
        super().__init__()
        self.state_dim = int(state_dim)
        self.movement = TanhGaussianActor(self.state_dim, 2, int(hidden), movement_low, movement_high)
        self.binary = nn.Sequential(
            nn.Linear(self.state_dim, int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), int(hidden)), nn.SiLU(),
            nn.Linear(int(hidden), 2),
        )

    def binary_logits(self, state: torch.Tensor) -> torch.Tensor:
        return self.binary(state)

    def binary_probabilities(self, state: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.binary_logits(state))

    def deterministic_action(self, state: torch.Tensor) -> torch.Tensor:
        move = self.movement.deterministic_action(state)
        bits = torch.where(self.binary_probabilities(state) >= 0.5, torch.ones_like(move), -torch.ones_like(move))
        return torch.cat([move, bits], dim=-1)

    def soft_action(self, state: torch.Tensor) -> torch.Tensor:
        move = self.movement.deterministic_action(state)
        bits = torch.tanh(self.binary_logits(state))
        return torch.cat([move, bits], dim=-1)


@dataclass
class HybridActorReport:
    critic_loss: float
    actor_loss: float
    movement_bc: float
    binary_bce: float
    q_value: float
    steps: int

    def to_dict(self):
        return asdict(self)


class HybridOfflineActorCriticBaseline:
    """TD3+BC-style actor/critic with explicit continuous/discrete action semantics."""

    def __init__(
        self,
        state_dim: int,
        *,
        hidden: int = 256,
        gamma: float = 0.99,
        tau: float = 0.01,
        bc_weight: float = 2.5,
        binary_bc_weight: float = 1.0,
        lr: float = 3e-4,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.gamma = float(gamma); self.tau = float(tau)
        self.bc_weight = float(bc_weight); self.binary_bc_weight = float(binary_bc_weight)
        self.state_dim = int(state_dim); self.action_dim = 4; self.hidden = int(hidden)
        self.actor = HybridGameActor(state_dim, hidden).to(self.device)
        self.target_actor = copy.deepcopy(self.actor).to(self.device)
        for p in self.target_actor.parameters(): p.requires_grad = False
        self.critic = TwinQCritic(state_dim, 4, hidden, [-1]*4, [1]*4).to(self.device)
        self.target_critic = TargetTwinQCritic(self.critic).to(self.device)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.steps = 0

    @torch.no_grad()
    def _target_update(self):
        for t, s in zip(self.target_actor.parameters(), self.actor.parameters()):
            t.data.lerp_(s.data, self.tau)
        self.target_critic.update(self.critic, self.tau)

    def train_batch(self, batch):
        if len(batch) == 6:
            s, a, r, ns, d, w = [x.to(self.device) for x in batch]
            w = w.reshape(-1); w = w / w.mean().clamp_min(1e-6)
        else:
            s, a, r, ns, d = [x.to(self.device) for x in batch]
            w = torch.ones_like(r)
        # Enforce the dataset ABI at the learner boundary.
        target_bits = (a[:, 2:4] > 0).float()
        if torch.any(~torch.isfinite(a)):
            raise ValueError("non-finite hybrid action in actor dataset")
        with torch.no_grad():
            na = self.target_actor.deterministic_action(ns)
            nq = self.target_critic.minimum(ns, na)
            target = r + self.gamma * (1.0 - d) * nq
        q1, q2 = self.critic(s, a)
        critic_rows = torch.nn.functional.smooth_l1_loss(q1, target, reduction="none") + torch.nn.functional.smooth_l1_loss(q2, target, reduction="none")
        critic_loss = (critic_rows * w).mean()
        self.critic_opt.zero_grad(set_to_none=True); critic_loss.backward(); self.critic_opt.step()

        pi_soft = self.actor.soft_action(s)
        q = self.critic.minimum(s, pi_soft)
        movement_rows = (pi_soft[:, :2] - a[:, :2]).square().mean(-1)
        binary_rows = torch.nn.functional.binary_cross_entropy_with_logits(self.actor.binary_logits(s), target_bits, reduction="none").mean(-1)
        move_bc = (movement_rows * w).mean(); binary_bce = (binary_rows * w).mean()
        scale = ((q.abs() * w).mean().detach() + 1e-4).reciprocal()
        actor_loss = -(scale * (q * w).mean()) + self.bc_weight * move_bc + self.binary_bc_weight * binary_bce
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()
        self._target_update(); self.steps += 1
        return float(critic_loss.detach()), float(actor_loss.detach()), float(move_bc.detach()), float(binary_bce.detach()), float(q.mean().detach())

    def fit(self, dataset: Dataset, epochs: int = 20, batch_size: int = 128, shuffle: bool = True):
        if len(dataset) == 0: raise ValueError("empty actor dataset")
        last = (0., 0., 0., 0., 0.)
        for _ in range(int(epochs)):
            for batch in DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=bool(shuffle)):
                last = self.train_batch(batch)
        return HybridActorReport(*last, self.steps)

    @torch.no_grad()
    def action(self, state):
        s = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        if s.ndim == 1: s = s.unsqueeze(0)
        return self.actor.deterministic_action(s)


    def load_state_dict(self, payload, *, load_optimizers: bool = False):
        self.actor.load_state_dict(payload["actor"]); self.target_actor.load_state_dict(payload.get("target_actor", payload["actor"]))
        self.critic.load_state_dict(payload["critic"]); self.target_critic.load_state_dict(payload.get("target_critic", payload["critic"]))
        if load_optimizers:
            if "actor_opt" in payload: self.actor_opt.load_state_dict(payload["actor_opt"])
            if "critic_opt" in payload: self.critic_opt.load_state_dict(payload["critic_opt"])
        self.steps = int(payload.get("steps", 0))
        return self

    def state_dict(self):
        return {
            "actor": self.actor.state_dict(), "target_actor": self.target_actor.state_dict(),
            "critic": self.critic.state_dict(), "target_critic": self.target_critic.state_dict(),
            "actor_opt": self.actor_opt.state_dict(), "critic_opt": self.critic_opt.state_dict(),
            "steps": self.steps, "gamma": self.gamma, "tau": self.tau,
            "bc_weight": self.bc_weight, "binary_bc_weight": self.binary_bc_weight,
        }
