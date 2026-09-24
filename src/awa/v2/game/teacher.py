from __future__ import annotations
import numpy as np
from .procedural_arena import ProceduralArenaEnv, _unit


class LogicalArenaTeacher:
    """Imperfect rule-based teacher aligned to the explicit objective program."""

    def __init__(self, attack_range: float = 0.30):
        self.attack_range = float(attack_range)

    def _avoid(self, env: ProceduralArenaEnv, direction: np.ndarray) -> np.ndarray:
        delta, distance = env._nearest_obstacle()
        move = _unit(direction)
        if distance < 0.18 and np.linalg.norm(delta) > 1e-6:
            away = -_unit(delta)
            tangent = _unit(np.asarray([-delta[1], delta[0]], dtype=np.float32))
            # Choose the tangent that is better aligned with the intended target.
            if float(np.dot(tangent, move)) < 0:
                tangent = -tangent
            move = _unit(0.50 * move + 0.70 * away + 0.45 * tangent)
        return move

    def act(self, env: ProceduralArenaEnv, observation=None) -> np.ndarray:
        interact = 0.0
        attack = 0.0
        obj = env.active_objective
        target = env.goal

        if obj in {"collect_object", "collect_resource"}:
            target = env.object_pos
            if np.linalg.norm(env.player - env.object_pos) <= 0.15:
                interact = 1.0
        elif obj == "collect_key":
            target = env.key_pos
            if np.linalg.norm(env.player - env.key_pos) <= 0.15:
                interact = 1.0
        elif obj == "open_door":
            target = env.door_pos
            if np.linalg.norm(env.player - env.door_pos) <= 0.18:
                interact = 1.0
        elif obj == "take_cover":
            target = env.cover
        elif obj == "avoid_enemy" and env.enemy_present:
            # Continue toward the eventual goal while strongly repelling from the threat.
            target = env.goal + 0.65 * _unit(env.player - env.enemy_pos)
        elif obj == "defeat_enemy" and env.enemy_present:
            d = float(np.linalg.norm(env.enemy_pos - env.player))
            if env.ammo > 0.02 and d <= self.attack_range:
                attack = 1.0
                target = env.player - 0.20 * _unit(env.enemy_pos - env.player)
            else:
                target = env.enemy_pos
        elif obj == "remember_goal":
            # The environment hides the observation after the reveal window; the
            # teacher uses the latent simulator state only to seed a coherent trace.
            target = env.goal
        elif obj in {"reach_goal", "survive", "complete"}:
            target = env.goal

        move = self._avoid(env, np.asarray(target, dtype=np.float32) - env.player)
        return np.asarray([move[0], move[1], attack, interact], dtype=np.float32)


class RandomArenaPolicy:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, env: ProceduralArenaEnv, observation=None) -> np.ndarray:
        a = self.rng.uniform(-1, 1, size=4).astype(np.float32)
        a[2:] = (a[2:] > 0.65).astype(np.float32)
        return a
