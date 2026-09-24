from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable
import numpy as np


class ObjectiveCode(IntEnum):
    REACH_GOAL = 0
    COLLECT_OBJECT = 1
    DEFEAT_ENEMY = 2
    COLLECT_RESOURCE = 3
    TAKE_COVER = 4
    COLLECT_KEY = 5
    OPEN_DOOR = 6
    SURVIVE = 7
    AVOID_ENEMY = 8
    REMEMBER_GOAL = 9


OBJECTIVE_NAMES = {
    ObjectiveCode.REACH_GOAL: "reach_goal",
    ObjectiveCode.COLLECT_OBJECT: "collect_object",
    ObjectiveCode.DEFEAT_ENEMY: "defeat_enemy",
    ObjectiveCode.COLLECT_RESOURCE: "collect_resource",
    ObjectiveCode.TAKE_COVER: "take_cover",
    ObjectiveCode.COLLECT_KEY: "collect_key",
    ObjectiveCode.OPEN_DOOR: "open_door",
    ObjectiveCode.SURVIVE: "survive",
    ObjectiveCode.AVOID_ENEMY: "avoid_enemy",
    ObjectiveCode.REMEMBER_GOAL: "remember_goal",
}
NAME_TO_CODE = {v: k for k, v in OBJECTIVE_NAMES.items()}
GOAL_DIM = len(ObjectiveCode) + 3


def normalize_objectives(goal: dict | None, stage: int | None = None) -> tuple[str, ...]:
    raw = [] if goal is None else list(goal.get("objectives", goal.get("sequence", ())))
    if raw:
        out = tuple(str(x) for x in raw)
        unknown = [x for x in out if x not in NAME_TO_CODE]
        if unknown:
            raise ValueError(f"unknown game objectives: {unknown}")
        return out
    # Backward-compatible stage mapping for older TaskSpecs.
    s = int(stage or (goal or {}).get("stage", 1))
    mapping = {
        1: ("reach_goal",),
        2: ("reach_goal",),
        3: ("collect_object", "reach_goal"),
        4: ("remember_goal", "reach_goal"),
        5: ("avoid_enemy", "reach_goal"),
        6: ("defeat_enemy", "reach_goal"),
        7: ("collect_resource", "defeat_enemy", "reach_goal"),
        8: ("take_cover", "reach_goal"),
        9: ("collect_key", "open_door", "reach_goal"),
        10: ("reach_goal",),
        11: ("reach_goal",),
        12: ("collect_object", "collect_key", "open_door", "defeat_enemy", "reach_goal"),
    }
    return mapping.get(s, ("reach_goal",))


@dataclass
class GoalProgram:
    objectives: tuple[str, ...]
    target_count: int = 1
    active_index: int = 0
    completed_object_collections: int = 0

    @classmethod
    def from_task(cls, task) -> "GoalProgram":
        return cls(
            normalize_objectives(task.goal, task.stage),
            max(1, int(task.goal.get("target_count", 1))),
        )

    @property
    def done(self) -> bool:
        return self.active_index >= len(self.objectives)

    @property
    def active(self) -> str:
        return "complete" if self.done else self.objectives[self.active_index]

    @property
    def progress(self) -> float:
        if not self.objectives:
            return 1.0
        return float(np.clip(self.active_index / len(self.objectives), 0.0, 1.0))

    def mark_complete(self) -> None:
        if not self.done:
            self.active_index += 1

    def vector(self) -> np.ndarray:
        out = np.zeros(GOAL_DIM, dtype=np.float32)
        if not self.done:
            out[int(NAME_TO_CODE[self.active])] = 1.0
        out[-3] = self.progress
        out[-2] = float(np.clip(self.target_count / 4.0, 0.0, 1.0))
        out[-1] = float(np.clip((len(self.objectives) - self.active_index) / max(1, len(self.objectives)), 0.0, 1.0))
        return out

    def state_dict(self) -> dict:
        return {
            "objectives": list(self.objectives),
            "target_count": int(self.target_count),
            "active_index": int(self.active_index),
            "completed_object_collections": int(self.completed_object_collections),
        }

    @classmethod
    def load(cls, raw: dict) -> "GoalProgram":
        obj = cls(tuple(raw["objectives"]), int(raw.get("target_count", 1)))
        obj.active_index = int(raw.get("active_index", 0))
        obj.completed_object_collections = int(raw.get("completed_object_collections", 0))
        return obj


def objectives_require(objectives: Iterable[str], name: str) -> bool:
    return str(name) in set(map(str, objectives))
