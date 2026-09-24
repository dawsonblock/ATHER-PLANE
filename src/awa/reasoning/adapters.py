from __future__ import annotations
import json
from typing import Callable
from awa.reasoning.semantic import Subgoal


class CallableReasoner:
    """Wrap an application-provided LLM/VLM callable without coupling runtime to a vendor API."""
    def __init__(self, fn: Callable[[str], str]):
        self.fn = fn

    def decompose(self, instruction: str, context: dict) -> list[Subgoal]:
        prompt = (
            "Return JSON array of subgoals. Each item must contain name, arguments, constraints.\n"
            f"Instruction: {instruction}\nContext: {json.dumps(context, sort_keys=True)}"
        )
        raw = self.fn(prompt)
        data = json.loads(raw)
        if not isinstance(data, list): raise ValueError("Reasoner must return a JSON list")
        return [Subgoal(str(x["name"]), dict(x.get("arguments", {})), dict(x.get("constraints", {}))) for x in data]
