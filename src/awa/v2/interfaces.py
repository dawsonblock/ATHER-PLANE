from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol, Any
import torch

@dataclass
class PlanResult:
    action: torch.Tensor
    score: float
    planner: str
    world_model_calls: int
    metadata: dict[str, Any]

class Planner(Protocol):
    name: str
    def plan(self, belief: torch.Tensor, actor=None, budget: int | None = None, goal=None) -> PlanResult: ...

class RepresentationModel(Protocol):
    def encode(self, observation: torch.Tensor) -> torch.Tensor: ...

class PredictiveWorldModel(Protocol):
    def imagine_step(self, belief: torch.Tensor, action: torch.Tensor, deterministic: bool = False): ...

class MemorySystem(Protocol):
    def add(self, *args, **kwargs): ...
    def query(self, *args, **kwargs): ...
