from __future__ import annotations

from collections import deque
import numpy as np
import torch


class DistillationBuffer:
    def __init__(self, capacity=200_000):
        capacity = int(capacity)
        if capacity <= 0:
            raise ValueError("distillation capacity must be > 0")
        self.items = deque(maxlen=capacity)

    def add(self, x):
        if x is not None:
            self.items.append(x)

    def arrays(self):
        if not self.items:
            raise ValueError("empty distillation buffer")
        s = np.stack([x.state for x in self.items])
        g = np.stack([x.goal for x in self.items])
        a = np.stack([x.action for x in self.items])
        w = np.asarray([x.weight for x in self.items], dtype=np.float32)
        return s, g, a, w

    def __len__(self):
        return len(self.items)


class PolicyDistiller:
    """Weighted teacher -> actor regression for continuous deterministic actions."""

    def __init__(self, actor, lr=3e-4):
        self.actor = actor
        self.opt = torch.optim.Adam(actor.parameters(), lr=float(lr))

    def step(self, state, target_action, weight=None):
        try:
            device = next(self.actor.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        target = torch.as_tensor(target_action, dtype=torch.float32, device=device)
        pred = self.actor.deterministic_action(state)
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.shape != target.shape:
            raise ValueError(f"teacher action shape {tuple(target.shape)} != actor shape {tuple(pred.shape)}")
        loss = torch.square(pred - target).mean(dim=-1)
        if weight is not None:
            weight = torch.as_tensor(weight, dtype=loss.dtype, device=loss.device)
            if torch.any(~torch.isfinite(weight)) or torch.any(weight < 0):
                raise ValueError("distillation weights must be finite and non-negative")
            loss = loss * weight
        loss = loss.mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return float(loss.detach())

    def fit(self, buffer: DistillationBuffer, epochs=5, batch_size=128, seed=0):
        epochs = int(epochs)
        batch_size = int(batch_size)
        if epochs <= 0:
            raise ValueError("epochs must be > 0")
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        s, _g, a, w = buffer.arrays()
        rng = np.random.default_rng(seed)
        hist = []
        for _ in range(epochs):
            idx = rng.permutation(len(s))
            for i in range(0, len(idx), batch_size):
                j = idx[i : i + batch_size]
                hist.append(self.step(s[j], a[j], w[j]))
        return {
            "steps": len(hist),
            "final_loss": float(hist[-1]),
            "mean_loss": float(np.mean(hist)),
        }
