from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol


@dataclass
class Subgoal:
    name: str
    arguments: dict
    constraints: dict


class SemanticReasoner(Protocol):
    def decompose(self, instruction: str, context: dict) -> list[Subgoal]: ...


class RuleReasoner:
    """Tiny deterministic fallback used by tests; replace with an LLM/VLM adapter."""
    def decompose(self, instruction: str, context: dict) -> list[Subgoal]:
        return [Subgoal("execute_instruction", {"text": instruction}, context.get("constraints", {}))]
