from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any

import numpy as np
import torch
from torch import nn

from awa.v2.world import MultimodalWorldModel, RiskConstraintModel
from awa.v2.actor_baseline import OfflineActorCriticBaseline
from awa.v2.representation import IdentityBackbone, module_fingerprint


WORLD_FORMATS = {"awa-v2.2-world-checkpoint-v1", "awa-v2.4-world-checkpoint-v1"}
ACTOR_FORMAT = "awa-v2.4-actor-checkpoint-v1"


@dataclass(frozen=True)
class WorldCheckpointMetadata:
    format: str
    dataset_sha256: str | None
    encoder_fingerprint: str | None
    feature_dim: int
    latent_dim: int
    action_dim: int
    hidden: int
    components: int
    horizons: tuple[int, ...]
    risk_hidden: int
    risk_constraints: int

    def to_dict(self):
        data = asdict(self)
        data["horizons"] = list(self.horizons)
        return data


@dataclass
class LoadedWorldRuntime:
    world: MultimodalWorldModel
    projector: nn.Module
    metadata: WorldCheckpointMetadata
    device: torch.device

    @torch.no_grad()
    def project_features(self, features: Any) -> torch.Tensor:
        x = torch.as_tensor(features, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x = x.reshape(x.shape[0], -1)
        if x.shape[-1] != self.metadata.feature_dim:
            raise ValueError(
                f"feature dim {x.shape[-1]} does not match checkpoint feature_dim "
                f"{self.metadata.feature_dim}"
            )
        return self.projector(x)

    @torch.no_grad()
    def encode_observation(
        self,
        observation: Any,
        feature_encoder: Callable[[Any], Any] | None = None,
    ) -> torch.Tensor:
        features = observation if feature_encoder is None else feature_encoder(observation)
        return self.project_features(features)


def _projector(feature_dim: int, latent_dim: int) -> nn.Module:
    return nn.Sequential(nn.Linear(feature_dim, latent_dim), nn.LayerNorm(latent_dim))


def load_world_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> LoadedWorldRuntime:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    fmt = payload.get("format")
    if fmt not in WORLD_FORMATS:
        raise ValueError(f"unsupported world checkpoint format: {fmt!r}")
    required = ("feature_dim", "latent_dim", "action_dim", "hidden", "components", "horizons", "model_state", "projector_state")
    missing = [k for k in required if k not in payload]
    if missing:
        raise ValueError(f"world checkpoint missing fields: {missing}")
    meta = WorldCheckpointMetadata(
        format=str(fmt),
        dataset_sha256=payload.get("dataset_sha256"),
        encoder_fingerprint=payload.get("encoder_fingerprint"),
        feature_dim=int(payload["feature_dim"]),
        latent_dim=int(payload["latent_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden=int(payload["hidden"]),
        components=int(payload["components"]),
        horizons=tuple(int(v) for v in payload["horizons"]),
        risk_hidden=int(payload.get("risk_hidden") or payload["model_state"]["risk.net.0.weight"].shape[0]),
        risk_constraints=int(payload.get("risk_constraints") or payload["model_state"]["risk.net.4.weight"].shape[0]),
    )
    dev = torch.device(device)
    world = MultimodalWorldModel(
        meta.latent_dim,
        meta.action_dim,
        hidden=meta.hidden,
        components=meta.components,
        horizons=meta.horizons,
    ).to(dev)
    projector = _projector(meta.feature_dim, meta.latent_dim).to(dev)
    # v2.2 checkpoints did not persist risk-head geometry. Reconstruct it from
    # the saved tensor shapes so old checkpoints remain strictly loadable.
    world.risk = RiskConstraintModel(meta.latent_dim, meta.action_dim, hidden=meta.risk_hidden, constraints=meta.risk_constraints).to(dev)
    incompatible = world.load_state_dict(payload["model_state"], strict=False)
    allowed_missing = {k for k in world.state_dict() if k.startswith("state_value.")}
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if unexpected or (missing - allowed_missing):
        raise RuntimeError(f"world checkpoint ABI mismatch; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    if missing:
        # Legacy checkpoints predate the terminal state-value head. Fail neutral:
        # zero bootstrap rather than injecting an untrained random value into search.
        with torch.no_grad():
            for p in world.state_value.parameters(): p.zero_()
    projector.load_state_dict(payload["projector_state"], strict=True)
    world.eval(); projector.eval()
    for module in (world, projector):
        for p in module.parameters():
            p.requires_grad = False
    return LoadedWorldRuntime(world, projector, meta, dev)


def assert_identity_encoder_compatible(runtime: LoadedWorldRuntime) -> str:
    """Require a checkpoint that was built from the deterministic identity backbone.

    This makes state-vector closed-loop benchmark evaluation provenance-safe. A
    vision checkpoint must instead provide the exact frozen feature encoder used to
    build its representation cache.
    """
    fp = module_fingerprint(IdentityBackbone(runtime.metadata.feature_dim))
    expected = runtime.metadata.encoder_fingerprint
    if expected is not None and expected != fp:
        raise ValueError(
            "checkpoint encoder fingerprint is not the identity feature encoder; "
            "provide the matching frozen feature encoder programmatically"
        )
    return fp


def save_actor_checkpoint(
    trainer: OfflineActorCriticBaseline,
    path: str | Path,
    *,
    world_checkpoint_sha256: str | None = None,
    dataset_sha256: str | None = None,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": ACTOR_FORMAT,
        "state_dim": int(trainer.state_dim),
        "action_dim": int(trainer.action_dim),
        "hidden": int(trainer.hidden),
        "low": trainer.actor.low.detach().cpu(),
        "high": trainer.actor.high.detach().cpu(),
        "gamma": float(trainer.gamma),
        "tau": float(trainer.tau),
        "bc_weight": float(trainer.bc_weight),
        "state": trainer.state_dict(),
        "world_checkpoint_sha256": world_checkpoint_sha256,
        "dataset_sha256": dataset_sha256,
    }
    tmp = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(target)
    return target


def load_actor_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> OfflineActorCriticBaseline:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("format") != ACTOR_FORMAT:
        raise ValueError(f"unsupported actor checkpoint format: {payload.get('format')!r}")
    trainer = OfflineActorCriticBaseline(
        int(payload["state_dim"]),
        int(payload["action_dim"]),
        np.asarray(payload["low"], dtype=np.float32),
        np.asarray(payload["high"], dtype=np.float32),
        hidden=int(payload["hidden"]),
        gamma=float(payload.get("gamma", .99)),
        tau=float(payload.get("tau", .01)),
        bc_weight=float(payload.get("bc_weight", 2.5)),
        device=device,
    )
    state = payload["state"]
    trainer.actor.load_state_dict(state["actor"])
    trainer.target_actor.load_state_dict(state["target_actor"])
    trainer.critic.load_state_dict(state["critic"])
    trainer.target_critic.load_state_dict(state["target_critic"])
    trainer.actor_opt.load_state_dict(state["actor_opt"])
    trainer.critic_opt.load_state_dict(state["critic_opt"])
    trainer.steps = int(state.get("steps", 0))
    trainer.actor.eval(); trainer.target_actor.eval(); trainer.critic.eval(); trainer.target_critic.eval()
    return trainer
