from __future__ import annotations
import torch
from torch import nn
from awa.perception.encoders import MLPEncoder
from awa.planning.actor import TanhGaussianActor, CategoricalActor
from awa.v2.temporal import LocalGlobalTemporalCore
from awa.v2.world import MultimodalWorldModel


class AetherV2Agent(nn.Module):
    """v2 composition: representation -> local/global belief -> multimodal world model -> actor.

    `encoder` may be a frozen pretrained representation adapter. The default MLP
    keeps the base package dependency-free and preserves v2.0 behavior.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_type: str = "continuous",
        latent_dim: int = 128,
        local_dim: int = 128,
        global_dim: int = 128,
        hidden: int = 256,
        low=None,
        high=None,
        encoder: nn.Module | None = None,
    ):
        super().__init__()
        self.action_type = action_type
        self.action_dim = action_dim
        self.encoder = encoder if encoder is not None else MLPEncoder(obs_dim, latent_dim, hidden)
        encoder_out = int(getattr(self.encoder, "output_dim", latent_dim))
        self.temporal = LocalGlobalTemporalCore(
            encoder_out, action_dim, local_dim, global_dim, event_slots=8, global_stride=4, heads=4
        )
        belief_dim = local_dim + global_dim
        self.belief_dim = belief_dim
        self.world = MultimodalWorldModel(belief_dim, action_dim, hidden=hidden, components=5)
        if action_type == "continuous":
            self.actor = TanhGaussianActor(belief_dim, action_dim, hidden, low, high)
        elif action_type == "discrete":
            self.actor = CategoricalActor(belief_dim, action_dim, hidden)
        else:
            raise ValueError(action_type)

    def initial_state(self, batch, device):
        return self.temporal.initial(batch, device)

    def observe_latent(self, state, latent, previous_action, step: int):
        """Advance temporal belief from a precomputed representation latent.

        This lets high-cost video encoders run outside the agent and reuse the exact
        projector stored in a qualified world checkpoint.
        """
        z = torch.as_tensor(latent, dtype=state.local.dtype, device=state.local.device)
        if z.ndim == 1:
            z = z.unsqueeze(0)
        state = self.temporal(state, z, previous_action, step)
        return state, state.vector

    def observe(self, state, observation, previous_action, step: int):
        z = self.encoder(observation)
        return self.observe_latent(state, z, previous_action, step)
