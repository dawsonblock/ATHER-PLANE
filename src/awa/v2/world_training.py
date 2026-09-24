from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable
import json
import math

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.representation import RepresentationCache
from awa.v2.world import MultimodalWorldModel


def validate_representation_index(dataset: OfflineTransitionDataset, index: dict) -> None:
    if index.get("format") not in {"awa-representation-index-v1", "awa-game-video-representation-index-v1"}:
        raise ValueError("unsupported representation index format")
    if index.get("dataset_sha256") != dataset.manifest.sha256:
        raise ValueError("representation index dataset hash does not match dataset")
    if int(index.get("transitions", -1)) != len(dataset):
        raise ValueError("representation index transition count does not match dataset")
    for key in ("observation_keys", "next_observation_keys"):
        values = index.get(key)
        if not isinstance(values, list) or len(values) != len(dataset):
            raise ValueError(f"representation index {key} length does not match dataset")


def contiguous_sequence_starts(
    dataset: OfflineTransitionDataset,
    sequence_length: int,
    *,
    verify_chain: bool = True,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> list[int]:
    """Return starts that never cross an episode boundary.

    If verify_chain is enabled, adjacent transitions must also satisfy
    next_observation[t] ~= observation[t+1]. This prevents accidental multi-step
    training on shuffled transition tables.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be >= 1")
    n = len(dataset)
    starts: list[int] = []
    obs = dataset.arrays["observations"]
    nxt = dataset.arrays["next_observations"]
    dones = dataset.arrays["dones"].astype(bool)
    for start in range(0, n - sequence_length + 1):
        end = start + sequence_length
        # A terminal is allowed only on the final transition of the sequence.
        if sequence_length > 1 and bool(dones[start : end - 1].any()):
            continue
        if verify_chain:
            valid = True
            for i in range(start, end - 1):
                if not torch.allclose(
                    torch.as_tensor(nxt[i]),
                    torch.as_tensor(obs[i + 1]),
                    atol=atol,
                    rtol=rtol,
                ):
                    valid = False
                    break
            if not valid:
                continue
        starts.append(start)
    return starts


class CachedFeatureSequenceDataset(Dataset):
    """Episode-safe contiguous sequences backed by the v2.1 feature cache."""

    def __init__(
        self,
        transitions: OfflineTransitionDataset,
        cache: RepresentationCache,
        index: dict | str | Path,
        sequence_length: int,
        *,
        verify_chain: bool = True,
    ):
        self.transitions = transitions
        self.cache = cache
        if not isinstance(index, dict):
            index = json.loads(Path(index).read_text(encoding="utf-8"))
        self.index = index
        validate_representation_index(transitions, index)
        self.encoder_fingerprint = str(index["encoder_fingerprint"])
        self.sequence_length = int(sequence_length)
        self.starts = contiguous_sequence_starts(
            transitions, self.sequence_length, verify_chain=verify_chain
        )
        if not self.starts:
            raise ValueError("no valid contiguous sequences found")

    def __len__(self) -> int:
        return len(self.starts)

    def _feature(self, key: str) -> torch.Tensor:
        value = self.cache.get_by_key(key)
        if value is None:
            raise FileNotFoundError(f"representation cache entry missing: {key}")
        return value.squeeze(0).float()

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        start = self.starts[idx]
        stop = start + self.sequence_length
        obs_keys = self.index["observation_keys"][start:stop]
        next_keys = self.index["next_observation_keys"][start:stop]
        return {
            "features": torch.stack([self._feature(k) for k in obs_keys]),
            "next_features": torch.stack([self._feature(k) for k in next_keys]),
            "actions": torch.as_tensor(self.transitions.arrays["actions"][start:stop]).float(),
            "rewards": torch.as_tensor(self.transitions.arrays["rewards"][start:stop]).float(),
            "dones": torch.as_tensor(self.transitions.arrays["dones"][start:stop]).float(),
            "indices": torch.arange(start, stop, dtype=torch.long),
        }


@dataclass
class WorldTrainMetrics:
    loss: float
    nll: float
    reward_mse: float
    continuation_bce: float
    overshoot_mse: float
    value_mse: float = 0.0


@dataclass
class WorldQualificationReport:
    one_step_nll: float
    reward_rmse: float
    continuation_brier: float
    horizon_rmse: dict[str, float]
    trajectory_rank_correlation: float
    samples: int

    def to_dict(self) -> dict:
        return asdict(self)


class OfflineWorldModelTrainer:
    """Train the multimodal latent dynamics model from cached representation sequences.

    The expensive pretrained backbone remains frozen and outside this loop. The
    supplied projector can be frozen or trainable; by default it is treated as a
    fixed representation contract so world-model qualification stays comparable.
    """

    def __init__(
        self,
        model: MultimodalWorldModel,
        projector: nn.Module,
        *,
        lr: float = 3e-4,
        reward_weight: float = 1.0,
        continuation_weight: float = 0.25,
        overshoot_weight: float = 0.5,
        overshoot_horizon: int = 4,
        train_projector: bool = False,
        value_weight: float = 0.25,
        gamma: float = 0.99,
    ):
        self.model = model
        self.projector = projector
        self.reward_weight = float(reward_weight)
        self.continuation_weight = float(continuation_weight)
        self.overshoot_weight = float(overshoot_weight)
        self.overshoot_horizon = int(overshoot_horizon)
        self.value_weight = float(value_weight)
        self.gamma = float(gamma)
        params = list(model.parameters())
        if train_projector:
            params.extend(p for p in projector.parameters() if p.requires_grad)
        else:
            for p in projector.parameters():
                p.requires_grad = False
        self.optimizer = torch.optim.AdamW(params, lr=lr)

    def _project(self, features: torch.Tensor) -> torch.Tensor:
        shape = features.shape
        flat = features.reshape(-1, shape[-1])
        projected = self.projector(flat)
        return projected.reshape(*shape[:-1], projected.shape[-1])

    def step(self, batch: dict[str, torch.Tensor]) -> WorldTrainMetrics:
        device = next(self.model.parameters()).device
        features = batch["features"].to(device)
        next_features = batch["next_features"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        dones = batch["dones"].to(device)
        states = self._project(features)
        next_states = self._project(next_features)
        B, T, D = states.shape
        flat_s = states.reshape(B * T, D)
        flat_a = actions.reshape(B * T, -1)
        flat_n = next_states.reshape(B * T, D)
        nll = self.model.nll_loss(flat_s, flat_a, flat_n)
        _, trunk = self.model.distribution(flat_s, flat_a)
        reward_pred = self.model.reward(trunk).squeeze(-1)
        reward_target = rewards.reshape(-1)
        reward_mse = F.mse_loss(reward_pred, reward_target)
        cont_pred = torch.sigmoid(self.model.cont(trunk)).squeeze(-1)
        cont_target = 1.0 - dones.reshape(-1)
        cont_bce = F.binary_cross_entropy(cont_pred, cont_target)

        max_h = min(self.overshoot_horizon, T)
        overs = []
        if max_h >= 2:
            for start in range(T - 1):
                imagined = states[:, start]
                for h in range(1, min(max_h, T - start) + 1):
                    imagined = self.model.imagine_step(
                        imagined, actions[:, start + h - 1], deterministic=True
                    )["belief"]
                    target = next_states[:, start + h - 1].detach()
                    if h >= 2:
                        overs.append(F.mse_loss(imagined, target))
        overshoot = torch.stack(overs).mean() if overs else nll * 0.0
        returns = torch.zeros_like(rewards)
        running = torch.zeros(B, device=device)
        for t in range(T - 1, -1, -1):
            running = rewards[:, t] + self.gamma * (1.0 - dones[:, t]) * running
            returns[:, t] = running
        value_pred = self.model.terminal_value(flat_s).reshape(B, T)
        value_mse = F.mse_loss(value_pred, returns.detach())
        loss = (
            nll
            + self.reward_weight * reward_mse
            + self.continuation_weight * cont_bce
            + self.overshoot_weight * overshoot
            + self.value_weight * value_mse
        )
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 100.0)
        self.optimizer.step()
        return WorldTrainMetrics(
            float(loss.detach()),
            float(nll.detach()),
            float(reward_mse.detach()),
            float(cont_bce.detach()),
            float(overshoot.detach()),
            float(value_mse.detach()),
        )

    def fit(self, dataset: Dataset, *, epochs: int = 1, batch_size: int = 32, shuffle: bool = True) -> list[WorldTrainMetrics]:
        history: list[WorldTrainMetrics] = []
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        for _ in range(int(epochs)):
            for batch in loader:
                history.append(self.step(batch))
        return history


@torch.no_grad()
def evaluate_world_model_horizons(
    model: MultimodalWorldModel,
    projector: nn.Module,
    dataset: Dataset,
    horizons: Iterable[int],
    *,
    batch_size: int = 64,
) -> WorldQualificationReport:
    device = next(model.parameters()).device
    hs = sorted({int(h) for h in horizons if int(h) >= 1})
    if not hs:
        raise ValueError("at least one positive horizon is required")
    sq_errors: dict[int, list[torch.Tensor]] = {h: [] for h in hs}
    nlls, reward_sq, cont_sq = [], [], []
    predicted_returns, realized_returns = [], []
    samples = 0
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features = batch["features"].to(device)
        next_features = batch["next_features"].to(device)
        actions = batch["actions"].to(device)
        rewards = batch["rewards"].to(device)
        dones = batch["dones"].to(device)
        B, T, _ = features.shape
        flat = projector(features.reshape(-1, features.shape[-1])).reshape(B, T, -1)
        flat_next = projector(next_features.reshape(-1, next_features.shape[-1])).reshape(B, T, -1)
        D = flat.shape[-1]
        s0 = flat.reshape(B * T, D)
        a0 = actions.reshape(B * T, -1)
        n0 = flat_next.reshape(B * T, D)
        nlls.append(model.nll_loss(s0, a0, n0).detach())
        _, trunk = model.distribution(s0, a0)
        rp = model.reward(trunk).squeeze(-1)
        cp = torch.sigmoid(model.cont(trunk)).squeeze(-1)
        reward_sq.append((rp - rewards.reshape(-1)).square())
        cont_sq.append((cp - (1.0 - dones.reshape(-1))).square())
        for h in hs:
            if h > T:
                continue
            pred = flat[:, 0]
            for j in range(h):
                pred = model.imagine_step(pred, actions[:, j], deterministic=True)["belief"]
            target = flat_next[:, h - 1]
            sq_errors[h].append((pred - target).square().mean(-1))
        # Evaluate whether the model ranks complete candidate action sequences in
        # the same order as their realized rewards. This is more planning-relevant
        # than reconstruction quality alone.
        pred_state = flat[:, 0]
        pred_return = torch.zeros(B, device=device)
        discount = torch.ones(B, device=device)
        for j in range(T):
            step = model.imagine_step(pred_state, actions[:, j], deterministic=True)
            pred_return = pred_return + discount * step["reward"].squeeze(-1)
            discount = discount * step["continuation"].squeeze(-1)
            pred_state = step["belief"]
        predicted_returns.append(pred_return.cpu())
        realized_returns.append(rewards.sum(dim=1).cpu())
        samples += B
    horizon_rmse = {
        str(h): float(torch.cat(sq_errors[h]).mean().sqrt()) if sq_errors[h] else math.nan
        for h in hs
    }
    px = torch.cat(predicted_returns).float()
    ry = torch.cat(realized_returns).float()
    def _rank(v: torch.Tensor) -> torch.Tensor:
        order = torch.argsort(v)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(len(v), dtype=torch.float32)
        return ranks
    if len(px) < 2 or float(px.std(unbiased=False)) == 0.0 or float(ry.std(unbiased=False)) == 0.0:
        rank_corr = 0.0
    else:
        rx, rr = _rank(px), _rank(ry)
        rx = rx - rx.mean(); rr = rr - rr.mean()
        rank_corr = float((rx * rr).mean() / (rx.square().mean().sqrt() * rr.square().mean().sqrt() + 1e-8))
    return WorldQualificationReport(
        one_step_nll=float(torch.stack(nlls).mean()),
        reward_rmse=float(torch.cat(reward_sq).mean().sqrt()),
        continuation_brier=float(torch.cat(cont_sq).mean()),
        horizon_rmse=horizon_rmse,
        trajectory_rank_correlation=rank_corr,
        samples=samples,
    )


@torch.no_grad()
def realized_horizon_errors(
    model: MultimodalWorldModel,
    projector: nn.Module,
    batch: dict[str, torch.Tensor],
    horizons: Iterable[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return initial states and realized RMSE targets for uncertainty calibration."""
    device = next(model.parameters()).device
    features = batch["features"].to(device)
    next_features = batch["next_features"].to(device)
    actions = batch["actions"].to(device)
    B, T, _ = features.shape
    states = projector(features.reshape(-1, features.shape[-1])).reshape(B, T, -1)
    next_states = projector(next_features.reshape(-1, next_features.shape[-1])).reshape(B, T, -1)
    outputs = []
    for h in horizons:
        h = int(h)
        if h > T:
            raise ValueError(f"sequence length {T} is shorter than calibration horizon {h}")
        pred = states[:, 0]
        for j in range(h):
            pred = model.imagine_step(pred, actions[:, j], deterministic=True)["belief"]
        outputs.append((pred - next_states[:, h - 1]).square().mean(-1).sqrt())
    return states[:, 0], torch.stack(outputs, dim=-1)


def calibrate_horizon_uncertainty(
    model: MultimodalWorldModel,
    projector: nn.Module,
    dataset: Dataset,
    *,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> dict[str, float | dict[str, float]]:
    """Fit only the horizon-uncertainty head against realized rollout errors."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.uncertainty.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.uncertainty.parameters(), lr=lr)
    horizons = model.uncertainty.horizons
    last = 0.0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(int(epochs)):
        for batch in loader:
            states, errors = realized_horizon_errors(model, projector, batch, horizons)
            pred = model.uncertainty(states.detach())
            loss = F.smooth_l1_loss(pred, errors.detach())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last = float(loss.detach())
    # Restore world model trainability for subsequent experiments.
    for p in model.parameters():
        p.requires_grad = True
    abs_by_h = {h: [] for h in horizons}
    with torch.no_grad():
        for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
            states, errors = realized_horizon_errors(model, projector, batch, horizons)
            pred = model.uncertainty(states)
            for i, h in enumerate(horizons):
                abs_by_h[h].append((pred[:, i] - errors[:, i]).abs().cpu())
    mae = {str(h): float(torch.cat(v).mean()) for h, v in abs_by_h.items()}
    return {"loss": last, "mae_by_horizon": mae, "mean_mae": float(sum(mae.values()) / len(mae))}
