from __future__ import annotations
import numpy as np


class ActionValidator:
    def __init__(self, action_dim: int): self.action_dim=action_dim
    def validate(self, action: int) -> int:
        if not 0 <= int(action) < self.action_dim: raise ValueError(f"Invalid action {action}; expected [0,{self.action_dim})")
        return int(action)


class ContinuousActionValidator:
    def __init__(self, low, high):
        self.low=np.asarray(low,dtype=np.float32).reshape(-1); self.high=np.asarray(high,dtype=np.float32).reshape(-1)
        if self.low.shape!=self.high.shape or np.any(self.high<=self.low): raise ValueError("invalid continuous bounds")
    def validate(self, action):
        a=np.asarray(action,dtype=np.float32).reshape(-1)
        if a.shape!=self.low.shape or not np.all(np.isfinite(a)): raise ValueError("invalid continuous action shape/value")
        return np.clip(a,self.low,self.high).astype(np.float32)


class ReflexLayer:
    """Hook for non-negotiable deterministic runtime rules."""
    def override(self, observation, proposed_action): return proposed_action
