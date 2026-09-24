from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ActionSpec:
    kind: str
    dim: int
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    @classmethod
    def discrete(cls, n: int) -> "ActionSpec":
        if n <= 0:
            raise ValueError("discrete action count must be positive")
        return cls("discrete", int(n), None, None)

    @classmethod
    def continuous(cls, low, high) -> "ActionSpec":
        lo = np.asarray(low, dtype=np.float32).reshape(-1)
        hi = np.asarray(high, dtype=np.float32).reshape(-1)
        if lo.shape != hi.shape or lo.size == 0:
            raise ValueError("continuous action bounds must have equal non-empty shape")
        if np.any(~np.isfinite(lo)) or np.any(~np.isfinite(hi)) or np.any(hi <= lo):
            raise ValueError("continuous action bounds must be finite and high > low")
        return cls("continuous", int(lo.size), lo, hi)

    def encode_tensor(self, actions: torch.Tensor) -> torch.Tensor:
        if self.kind == "discrete":
            if actions.ndim > 0 and actions.shape[-1] == self.dim and actions.is_floating_point():
                return actions.float()
            return F.one_hot(actions.long(), self.dim).float()
        return actions.float()

    def zero(self, batch: int, device) -> torch.Tensor:
        return torch.zeros(batch, self.dim, device=device)

    def random(self, batch: int, device) -> torch.Tensor:
        if self.kind == "discrete":
            idx = torch.randint(self.dim, (batch,), device=device)
            return F.one_hot(idx, self.dim).float()
        lo = torch.as_tensor(self.low, dtype=torch.float32, device=device)
        hi = torch.as_tensor(self.high, dtype=torch.float32, device=device)
        return lo + torch.rand(batch, self.dim, device=device) * (hi - lo)


def action_spec_from_config(cfg: dict) -> ActionSpec:
    env = cfg["env"]
    kind = env.get("action_type", "discrete")
    if kind == "discrete":
        return ActionSpec.discrete(int(env["action_dim"]))
    if kind != "continuous":
        raise ValueError(f"Unsupported action_type: {kind}")
    dim = int(env["action_dim"])
    low = env.get("action_low", [-1.0] * dim)
    high = env.get("action_high", [1.0] * dim)
    if np.isscalar(low): low = [float(low)] * dim
    if np.isscalar(high): high = [float(high)] * dim
    spec = ActionSpec.continuous(low, high)
    if spec.dim != dim:
        raise ValueError("env.action_dim does not match continuous action bounds")
    return spec
