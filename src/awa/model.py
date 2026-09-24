from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from awa.perception.encoders import MLPEncoder
from awa.state.belief import Belief, StochasticBeliefState
from awa.dynamics.fast import GRUDynamics, GatedSSMDynamics
from awa.dynamics.slow import SlowStateModel
from awa.dynamics.experts import TaskConditionedDynamicsExperts


@dataclass
class ModelOutput:
    belief: Belief
    reward: torch.Tensor
    continuation_logit: torch.Tensor
    value: torch.Tensor
    prior: torch.distributions.Normal
    posterior: torch.distributions.Normal
    router_weights: torch.Tensor | None = None


class WorldModel(nn.Module):
    def __init__(
        self, obs_dim: int, latent_dim: int, deterministic_dim: int, stochastic_dim: int,
        action_dim: int, hidden_dim: int = 256, dynamics: str = "gru",
        slow_enabled: bool = False, slow_dim: int = 128, slow_stride: int = 8,
        experts_enabled: bool = False, experts_count: int = 4, experts_top_k: int = 2,
        expert_context_dim: int = 0,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.deterministic_dim = deterministic_dim
        self.stochastic_dim = stochastic_dim
        self.slow_enabled = slow_enabled
        self.slow_dim = slow_dim if slow_enabled else 0
        self.slow_stride = max(1, slow_stride)
        self.experts_enabled = experts_enabled
        self.expert_context_dim = expert_context_dim

        self.encoder = MLPEncoder(obs_dim, latent_dim, hidden_dim)
        self.state = StochasticBeliefState(deterministic_dim, latent_dim, stochastic_dim)
        if dynamics == "gru":
            self.dynamics = GRUDynamics(stochastic_dim, action_dim, deterministic_dim)
        elif dynamics == "ssm":
            self.dynamics = GatedSSMDynamics(stochastic_dim, action_dim, deterministic_dim)
        else:
            raise ValueError(f"Unknown dynamics: {dynamics}")

        fast_dim = deterministic_dim + stochastic_dim
        self.slow_model = SlowStateModel(fast_dim, slow_dim) if slow_enabled else None
        self.belief_dim = fast_dim + self.slow_dim
        self.experts = TaskConditionedDynamicsExperts(
            self.belief_dim, experts_count, hidden_dim, experts_top_k, expert_context_dim
        ) if experts_enabled else None
        self.reward_head = nn.Sequential(nn.Linear(self.belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.cont_head = nn.Sequential(nn.Linear(self.belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.value_head = nn.Sequential(nn.Linear(self.belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.obs_predictor = nn.Sequential(nn.Linear(self.belief_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, latent_dim))

    def initial_belief(self, batch: int, device) -> Belief:
        slow = torch.zeros(batch, self.slow_dim, device=device) if self.slow_enabled else None
        return Belief(
            deterministic=torch.zeros(batch, self.deterministic_dim, device=device),
            stochastic=torch.zeros(batch, self.stochastic_dim, device=device),
            slow=slow,
        )

    def _with_slow(self, h: torch.Tensor, s: torch.Tensor, previous: Belief, step_index: int | None):
        if not self.slow_enabled:
            return Belief(h, s, None)
        slow = previous.slow
        if slow is None:
            slow = torch.zeros(h.shape[0], self.slow_dim, device=h.device, dtype=h.dtype)
        update = step_index is None or (step_index % self.slow_stride == 0)
        if update:
            slow = self.slow_model(slow, torch.cat([h, s], dim=-1))
        return Belief(h, s, slow)

    def features(self, belief: Belief, task_context: torch.Tensor | None = None):
        vec = belief.vector
        if self.experts is None:
            return vec, None
        return self.experts(vec, task_context)

    def observe(self, belief: Belief, action: torch.Tensor, observation: torch.Tensor,
                step_index: int | None = None, task_context: torch.Tensor | None = None) -> ModelOutput:
        h = self.dynamics(belief.deterministic, belief.stochastic, action)
        z = self.encoder(observation)
        s, posterior = self.state.sample_posterior(h, z)
        prior = self.state.prior_dist(h)
        next_belief = self._with_slow(h, s, belief, step_index)
        feat, routing = self.features(next_belief, task_context)
        return ModelOutput(
            next_belief, self.reward_head(feat), self.cont_head(feat), self.value_head(feat),
            prior, posterior, routing,
        )

    def imagine(self, belief: Belief, action: torch.Tensor, step_index: int | None = None,
                task_context: torch.Tensor | None = None):
        h = self.dynamics(belief.deterministic, belief.stochastic, action)
        s, prior = self.state.sample_prior(h)
        b = self._with_slow(h, s, belief, step_index)
        feat, routing = self.features(b, task_context)
        return b, self.reward_head(feat), self.cont_head(feat), self.value_head(feat), prior, routing

    def latent_prediction(self, belief: Belief, task_context: torch.Tensor | None = None) -> torch.Tensor:
        feat, _ = self.features(belief, task_context)
        return self.obs_predictor(feat)

    def latent_prediction_loss(self, belief: Belief, observation: torch.Tensor,
                               task_context: torch.Tensor | None = None,
                               reduction: str = "mean") -> torch.Tensor:
        target = self.encoder(observation).detach()
        pred = self.latent_prediction(belief, task_context)
        return F.mse_loss(pred, target, reduction=reduction)
