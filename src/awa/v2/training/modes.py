from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TrainingMode(str, Enum):
    EXPLOIT = "exploit"
    EXPLORE = "explore"
    QUALIFY = "qualify"


@dataclass(frozen=True)
class RewardBreakdown:
    task: float
    novelty: float
    information: float
    surprise: float
    total: float
    mode: TrainingMode


class IntrinsicRewardComposer:
    """Task reward plus bounded intrinsic signals used only during exploration.

    Qualification mode intentionally strips all intrinsic bonuses so benchmark reward
    cannot be contaminated by curiosity shaping.
    """

    def __init__(self, novelty_weight: float = 0.05, information_weight: float = 0.05, surprise_weight: float = 0.02, intrinsic_cap: float = 0.25):
        self.novelty_weight = float(novelty_weight)
        self.information_weight = float(information_weight)
        self.surprise_weight = float(surprise_weight)
        self.intrinsic_cap = float(intrinsic_cap)

    def compose(self, task_reward: float, *, novelty: float = 0.0, information: float = 0.0, surprise: float = 0.0, mode: TrainingMode = TrainingMode.EXPLOIT) -> RewardBreakdown:
        task = float(task_reward)
        if mode is TrainingMode.EXPLORE:
            intrinsic = self.novelty_weight * float(novelty) + self.information_weight * float(information) + self.surprise_weight * float(surprise)
            intrinsic = max(-self.intrinsic_cap, min(self.intrinsic_cap, intrinsic))
        else:
            intrinsic = 0.0
        return RewardBreakdown(task, float(novelty), float(information), float(surprise), task + intrinsic, mode)


class TrainingModeController:
    def __init__(self, exploit_steps: int = 7, explore_steps: int = 2, qualify_steps: int = 1):
        if min(exploit_steps, explore_steps, qualify_steps) < 0 or (exploit_steps + explore_steps + qualify_steps) <= 0:
            raise ValueError("mode schedule must contain at least one positive segment")
        self.schedule = ([TrainingMode.EXPLOIT] * int(exploit_steps) + [TrainingMode.EXPLORE] * int(explore_steps) + [TrainingMode.QUALIFY] * int(qualify_steps))
        self.position = 0

    @property
    def mode(self) -> TrainingMode:
        return self.schedule[self.position % len(self.schedule)]

    def step(self) -> TrainingMode:
        out = self.mode
        self.position += 1
        return out

    def state_dict(self) -> dict:
        return {"schedule": [m.value for m in self.schedule], "position": int(self.position)}

    def load_state_dict(self, state: dict) -> None:
        schedule = [TrainingMode(v) for v in state["schedule"]]
        if not schedule:
            raise ValueError("empty training-mode schedule")
        self.schedule = schedule
        self.position = int(state.get("position", 0))
