from __future__ import annotations

from dataclasses import dataclass
import numpy as np

COMBAT_SCENARIOS = {"basic", "defend_the_center", "predict_position", "deadly_corridor", "deathmatch"}
SURVIVAL_SCENARIOS = {"health_gathering", "take_cover"}


@dataclass
class ScenarioSnapshot:
    health: float = 100.0
    ammo: float = 0.0
    kills: float = 0.0
    items: float = 0.0


class ViZDoomScenarioSemantics:
    """Scenario-aware success and constraint semantics.

    The tracker intentionally uses only stable game variables/reward/termination so it
    works across native ViZDoom and DoomGame-compatible test doubles. Scenario-specific
    wrappers can override these rules later without changing the dataset ABI.
    """

    def __init__(self, scenario):
        self.scenario = getattr(scenario, "value", str(scenario))
        self.start = ScenarioSnapshot()
        self.prev = ScenarioSnapshot()
        self._labels = np.zeros(4, dtype=np.float32)

    def reset(self, read):
        snap = ScenarioSnapshot(
            health=float(read("HEALTH")), ammo=float(read("SELECTED_WEAPON_AMMO")),
            kills=float(read("KILLCOUNT")), items=float(read("ITEMCOUNT")),
        )
        self.start = snap; self.prev = ScenarioSnapshot(**snap.__dict__)
        self._labels = np.zeros(4, dtype=np.float32)

    def update(self, read, *, finished: bool, truncated: bool, player_dead: bool, reward: float, episode_return: float) -> tuple[bool, np.ndarray]:
        cur = ScenarioSnapshot(
            health=float(read("HEALTH")), ammo=float(read("SELECTED_WEAPON_AMMO")),
            kills=float(read("KILLCOUNT")), items=float(read("ITEMCOUNT")),
        )
        combat = self.scenario in COMBAT_SCENARIOS
        kill_delta = cur.kills - self.start.kills
        took_damage = cur.health + 1e-6 < self.prev.health
        critical = 0.0 < cur.health < 25.0
        ammo_depleted = combat and cur.ammo <= 0.0
        self._labels = np.asarray([float(player_dead or cur.health <= 0.0), float(critical), float(ammo_depleted), float(took_damage)], dtype=np.float32)

        terminal = bool(finished or truncated)
        if self.scenario == "basic":
            success = kill_delta >= 1.0
        elif self.scenario == "my_way_home":
            success = bool(terminal and not player_dead and episode_return > 0.0)
        elif self.scenario == "health_gathering":
            success = bool(terminal and not player_dead and cur.health > 0.0)
        elif self.scenario == "defend_the_center":
            success = bool(kill_delta >= 1.0 and not player_dead)
        elif self.scenario == "predict_position":
            success = bool(kill_delta >= 1.0)
        elif self.scenario == "take_cover":
            success = bool(terminal and not player_dead and cur.health > 0.0)
        elif self.scenario == "deadly_corridor":
            success = bool(terminal and not player_dead and episode_return > 0.0)
        elif self.scenario == "deathmatch":
            success = bool(kill_delta >= 1.0 or (terminal and not player_dead and episode_return > 0.0))
        else:
            success = bool(terminal and not player_dead and reward > 0.0)
        self.prev = cur
        return bool(success), self._labels.copy()

    def constraint_labels(self) -> np.ndarray:
        return self._labels.copy()
