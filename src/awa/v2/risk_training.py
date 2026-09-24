from __future__ import annotations

from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from awa.v2.world import RiskConstraintModel


@dataclass
class RiskMetrics:
    loss: float
    brier: float
    accuracy: float
    positive_rate: float
    calibration_error: float = 0.0

    def to_dict(self):
        return asdict(self)


def _ece(probs: torch.Tensor, labels: torch.Tensor, bins: int = 10) -> float:
    p = probs.detach().reshape(-1).float()
    y = labels.detach().reshape(-1).float()
    total = max(1, p.numel())
    error = torch.tensor(0.0, device=p.device)
    edges = torch.linspace(0.0, 1.0, bins + 1, device=p.device)
    for i in range(bins):
        if i == bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])
        if mask.any():
            error = error + mask.float().sum() / total * (p[mask].mean() - y[mask].mean()).abs()
    return float(error)


class CachedRiskDataset(Dataset):
    """Attach explicit constraint labels to cached feature transitions."""

    def __init__(self, feature_sequences: Dataset):
        self.base = feature_sequences
        transitions = getattr(feature_sequences, "transitions", None)
        if transitions is None or "constraints" not in transitions.arrays:
            raise ValueError("offline dataset does not contain `constraints` labels")
        labels = transitions.arrays["constraints"]
        if labels.ndim != 2:
            raise ValueError("constraints must have shape [N,C]")
        self.constraints = labels

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        row = self.base[idx]
        indices = row["indices"]
        labels = torch.as_tensor(self.constraints[indices.numpy()]).float()
        return {**row, "constraints": labels}


class RiskConstraintTrainer:
    def __init__(self, model: RiskConstraintModel, projector: nn.Module, lr: float = 3e-4):
        self.model = model
        self.projector = projector
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def _flatten(self, batch):
        device = next(self.model.parameters()).device
        features = batch["features"].to(device)
        actions = batch["actions"].to(device)
        labels = batch["constraints"].to(device)
        B, T, Fdim = features.shape
        states = self.projector(features.reshape(B * T, Fdim))
        return states, actions.reshape(B * T, -1), labels.reshape(B * T, -1)

    def step(self, batch) -> RiskMetrics:
        states, actions, labels = self._flatten(batch)
        logits = self.model.logits(states, actions)
        if labels.shape[-1] != logits.shape[-1]:
            raise ValueError("constraint label count does not match risk model")
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        probs = torch.sigmoid(logits.detach())
        brier = (probs - labels).square().mean()
        acc = ((probs >= 0.5) == (labels >= 0.5)).float().mean()
        return RiskMetrics(float(loss.detach()), float(brier), float(acc), float(labels.mean()), _ece(probs, labels))

    def fit(self, dataset, epochs=1, batch_size=64):
        history = []
        for _ in range(int(epochs)):
            for batch in DataLoader(dataset, batch_size=batch_size, shuffle=True):
                history.append(self.step(batch))
        return history


@torch.no_grad()
def evaluate_risk_model(model: RiskConstraintModel, projector: nn.Module, dataset, batch_size=64) -> RiskMetrics:
    losses, briers, correct, labels_all = [], [], [], []
    device = next(model.parameters()).device
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features = batch["features"].to(device)
        actions = batch["actions"].to(device)
        labels = batch["constraints"].to(device)
        B, T, Fdim = features.shape
        states = projector(features.reshape(B * T, Fdim))
        labels = labels.reshape(B * T, -1)
        logits = model.logits(states, actions.reshape(B * T, -1))
        probs = torch.sigmoid(logits)
        losses.append(F.binary_cross_entropy_with_logits(logits, labels, reduction="none").reshape(-1))
        briers.append((probs - labels).square().reshape(-1))
        correct.append(((probs >= 0.5) == (labels >= 0.5)).float().reshape(-1))
        labels_all.append(labels.reshape(-1))
    probs_all = []
    labels_cat = torch.cat(labels_all)
    # Reconstruct probability vector from Brier/accuracy pass is not possible, so
    # perform one lightweight second pass for a proper calibration metric.
    for batch in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        features = batch["features"].to(device); actions = batch["actions"].to(device)
        labels = batch["constraints"].to(device); B, T, Fdim = features.shape
        states = projector(features.reshape(B * T, Fdim))
        probs_all.append(torch.sigmoid(model.logits(states, actions.reshape(B * T, -1))).reshape(-1).cpu())
    return RiskMetrics(
        float(torch.cat(losses).mean()),
        float(torch.cat(briers).mean()),
        float(torch.cat(correct).mean()),
        float(labels_cat.mean()),
        _ece(torch.cat(probs_all), labels_cat.cpu()),
    )
