from __future__ import annotations

from dataclasses import dataclass
import math
import torch


class UnsafeRecoveryError(RuntimeError):
    """Raised when both a proposal and the configured recovery action are unsafe."""


@dataclass
class GuardResult:
    action: torch.Tensor
    accepted: bool
    risk: float
    reason: str
    proposed_risk: float | None = None
    recovery_safe: bool = True


class SafeActionGuard:
    """Fail-closed last-mile guard for bounded continuous actions.

    v2.7 returned the recovery action without checking whether that action was also
    above the learned-risk threshold.  v2.8 validates both the proposal and recovery;
    if neither is acceptable it raises ``UnsafeRecoveryError`` rather than silently
    emitting an unsafe fallback.
    """

    def __init__(self, risk_model, low, high, risk_limit: float = 0.8, recovery_action=None, *, strict_recovery: bool = False):
        self.risk_model = risk_model
        self.risk_limit = float(risk_limit)
        self.strict_recovery = bool(strict_recovery)
        if not math.isfinite(self.risk_limit) or self.risk_limit < 0:
            raise ValueError("risk_limit must be finite and >= 0")
        self.low = torch.as_tensor(low, dtype=torch.float32).reshape(-1)
        self.high = torch.as_tensor(high, dtype=torch.float32).reshape(-1)
        if self.low.numel() == 0 or self.low.shape != self.high.shape:
            raise ValueError("low/high action bounds must be non-empty and equal width")
        if torch.any(~torch.isfinite(self.low)) or torch.any(~torch.isfinite(self.high)):
            raise ValueError("action bounds must be finite")
        if torch.any(self.high <= self.low):
            raise ValueError("each high action bound must exceed low")
        if recovery_action is None:
            recovery_action = (self.low + self.high) * 0.5
        recovery = torch.as_tensor(recovery_action, dtype=torch.float32).reshape(-1)
        if recovery.shape != self.low.shape:
            raise ValueError("recovery_action dimension must match action bounds")
        if torch.any(~torch.isfinite(recovery)):
            raise ValueError("recovery_action must be finite")
        self.recovery = recovery.clamp(self.low, self.high).reshape(1, -1)

    def _risk(self, belief, action) -> float:
        value = float(self.risk_model.aggregate_risk(belief, action).mean().item())
        if not math.isfinite(value):
            return float("inf")
        return value

    @torch.no_grad()
    def validate(self, belief, action):
        device = belief.device if torch.is_tensor(belief) else None
        action = torch.as_tensor(action, dtype=torch.float32, device=device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        if action.shape[-1] != self.low.numel():
            raise ValueError("action dimension does not match guard bounds")
        low = self.low.to(action.device)
        high = self.high.to(action.device)
        clipped = action.clamp(low, high)
        proposed_risk = self._risk(belief, clipped)
        if proposed_risk <= self.risk_limit:
            return GuardResult(clipped, True, proposed_risk, "accepted", proposed_risk, True)

        recovery = self.recovery.to(action.device).expand_as(clipped).clone()
        recovery_risk = self._risk(belief, recovery)
        if recovery_risk > self.risk_limit:
            if self.strict_recovery:
                raise UnsafeRecoveryError(
                    f"proposal risk {proposed_risk:.6g} and recovery risk {recovery_risk:.6g} "
                    f"both exceed limit {self.risk_limit:.6g}"
                )
            return GuardResult(recovery, False, recovery_risk, "unsafe_recovery", proposed_risk, False)
        return GuardResult(recovery, False, recovery_risk, "risk_limit_recovery", proposed_risk, True)
