from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class Skill:
    name: str
    precondition: Callable[[dict], bool]
    policy: Callable
    success: Callable[[dict], bool]
    metadata: dict = field(default_factory=dict)


class SkillRegistry:
    def __init__(self):
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill):
        if skill.name in self._skills:
            raise ValueError(f"Skill already registered: {skill.name}")
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill:
        return self._skills[name]

    def available(self, state: dict):
        return [s for s in self._skills.values() if s.precondition(state)]
