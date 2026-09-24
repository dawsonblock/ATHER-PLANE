from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

@dataclass(frozen=True)
class HybridGameAction:
    movement: tuple[float,float]
    attack: bool
    interact: bool

class HybridGameActionCodec:
    """Canonical hybrid view over Aether's 4-D game action ABI.

    The environment ABI remains backward-compatible (4 floats), while training and
    planning can treat movement as continuous and attack/interact as categorical.
    """
    action_dim=4
    def __init__(self, threshold: float=0.25): self.threshold=float(threshold)
    def decode(self, action) -> HybridGameAction:
        a=np.asarray(action,dtype=np.float32).reshape(4)
        return HybridGameAction((float(np.clip(a[0],-1,1)),float(np.clip(a[1],-1,1))),bool(a[2]>self.threshold),bool(a[3]>self.threshold))
    def encode(self, movement, attack=False, interact=False, *, high=1.0, low=-1.0):
        m=np.asarray(movement,dtype=np.float32).reshape(2)
        return np.asarray([np.clip(m[0],-1,1),np.clip(m[1],-1,1),high if attack else low,high if interact else low],dtype=np.float32)
    def quantize(self, action):
        h=self.decode(action); return self.encode(h.movement,h.attack,h.interact)
    def enumerate_binary(self, movement):
        return np.stack([self.encode(movement,a,i) for a in (False,True) for i in (False,True)],axis=0)
    def quantize_tensor(self, action: torch.Tensor):
        out=action.clone(); out[...,2:]=(out[...,2:]>self.threshold).to(out.dtype)*2.0-1.0; return out
