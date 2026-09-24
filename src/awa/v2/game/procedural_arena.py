from __future__ import annotations

from dataclasses import dataclass
import copy
from collections import deque
import numpy as np

from awa.v2.curriculum.task_spec import TaskSpec
from .goal_program import GoalProgram, GOAL_DIM
from .action_codec import HybridGameActionCodec

OBS_DIM = 32
ACTION_DIM = 4


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32)
    n = float(np.linalg.norm(v))
    return np.zeros_like(v) if n < 1e-8 else v / n


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@dataclass(frozen=True)
class ArenaTransitionInfo:
    success: bool
    collision: bool
    damage_taken: float
    collected: bool
    key_collected: bool
    door_opened: bool
    enemy_defeated: bool
    distance_to_goal: float
    failure: bool
    active_objective: str = "complete"
    objective_advanced: bool = False


class ProceduralArenaEnv:
    """Deterministic procedural game lab with explicit objective semantics.

    Curriculum stage is metadata only; mechanics and success conditions are derived
    from TaskSpec.goal['objectives'].  This prevents stage-id shortcuts and ensures
    held-out task/compositional OOD variants change executable game semantics.
    """

    def __init__(self, task: TaskSpec, horizon: int = 160):
        self.task = task
        self.horizon = int(horizon)
        if self.horizon <= 0:
            raise ValueError("horizon must be > 0")
        self.rng = np.random.default_rng(int(task.seed))
        self.action_low = np.full(ACTION_DIM, -1.0, dtype=np.float32)
        self.action_high = np.full(ACTION_DIM, 1.0, dtype=np.float32)
        self.obs_dim = OBS_DIM
        self.action_dim = ACTION_DIM
        self.goal_dim = GOAL_DIM
        self.action_codec = HybridGameActionCodec()
        self._generate_static_world()
        self.reset()

    # ---------------- geometry ----------------
    def _generate_static_world(self) -> None:
        cfg = self.task.environment
        layout_seed = int(cfg.get("layout_seed", self.task.seed))
        r = np.random.default_rng(layout_seed)
        density = float(cfg.get("obstacle_density", cfg.get("object_density", 0.0)))
        count = 0 if density <= 0 else int(np.clip(round(1 + density * 5), 1, 5))
        obstacles: list[tuple[np.ndarray, float]] = []
        for _ in range(count):
            placed = False
            for _try in range(128):
                center = r.uniform(-0.70, 0.70, size=2).astype(np.float32)
                radius = float(r.uniform(0.08, 0.15))
                if all(float(np.linalg.norm(center - c)) > radius + rr + 0.10 for c, rr in obstacles):
                    obstacles.append((center, radius)); placed = True; break
            if not placed:
                break
        self.obstacles = obstacles
        self.cover_radius = 0.18
        self.cover = self._sample_static_clear(r, margin=self.cover_radius + 0.04)

    def _sample_static_clear(self, rng, margin: float = 0.07) -> np.ndarray:
        for _ in range(512):
            p = rng.uniform(-0.80, 0.80, size=2).astype(np.float32)
            if self._point_clear(p, margin=margin, include_door=False):
                return p
        raise RuntimeError("failed to place static point in valid free space")

    def _point_clear(self, p: np.ndarray, *, margin: float = 0.05, include_door: bool = False) -> bool:
        p = np.asarray(p, dtype=np.float32)
        if np.any(np.abs(p) > 0.94 - margin):
            return False
        for center, radius in self.obstacles:
            if float(np.linalg.norm(p - center)) < radius + margin:
                return False
        if include_door and hasattr(self, "door_pos") and not getattr(self, "door_open", False):
            if float(np.linalg.norm(p - self.door_pos)) < 0.10 + margin:
                return False
        return True

    def _reachable_cells(self, start: np.ndarray, *, n: int = 25) -> tuple[set[tuple[int,int]], callable]:
        """Return the free-space component containing start on a conservative grid."""
        def to_cell(p):
            q = np.rint((np.asarray(p, dtype=np.float32) + 0.92) / 1.84 * (n - 1)).astype(int)
            return int(np.clip(q[0], 0, n - 1)), int(np.clip(q[1], 0, n - 1))
        def to_point(c):
            return np.asarray([c[0], c[1]], np.float32) / (n - 1) * 1.84 - 0.92
        start_cell = to_cell(start)
        q = deque([start_cell]); seen = {start_cell}
        while q:
            x, y = q.popleft()
            for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                z = (x+dx, y+dy)
                if not (0 <= z[0] < n and 0 <= z[1] < n) or z in seen:
                    continue
                if not self._point_clear(to_point(z), margin=0.025, include_door=False):
                    continue
                seen.add(z); q.append(z)
        return seen, to_cell

    def _grid_reachable(self, start: np.ndarray, target: np.ndarray) -> bool:
        if hasattr(self, "_reachable_component") and np.allclose(start, self.player):
            seen, to_cell = self._reachable_component
        else:
            seen, to_cell = self._reachable_cells(start)
        return to_cell(target) in seen

    def _sample_point(self, min_from: np.ndarray | None = None, distance: float = 0.35, *, reachable_from: np.ndarray | None = None, extra_away: list[np.ndarray] | None = None) -> np.ndarray:
        extra_away = extra_away or []
        for _ in range(1024):
            p = self.rng.uniform(-0.82, 0.82, size=2).astype(np.float32)
            if not self._point_clear(p, margin=0.065, include_door=False):
                continue
            if min_from is not None and float(np.linalg.norm(p - min_from)) < distance:
                continue
            if any(float(np.linalg.norm(p - q)) < 0.20 for q in extra_away):
                continue
            if reachable_from is not None and not self._grid_reachable(reachable_from, p):
                continue
            return p
        raise RuntimeError("failed to sample a valid reachable game point")

    # ---------------- task/objective semantics ----------------
    def _objectives(self) -> set[str]:
        return set(self.program.objectives)

    def _needs_object(self) -> bool:
        return bool(self._objectives() & {"collect_object", "collect_resource"})

    def _needs_enemy(self) -> bool:
        return bool(self._objectives() & {"defeat_enemy", "avoid_enemy", "take_cover", "survive"})

    def _needs_key_door(self) -> bool:
        return bool(self._objectives() & {"collect_key", "open_door"})

    def _needs_cover(self) -> bool:
        return bool(self._objectives() & {"take_cover", "avoid_enemy"})

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(int(seed))
        self.t = 0
        self.program = GoalProgram.from_task(self.task)
        self.player = self._sample_point()
        self._reachable_component = self._reachable_cells(self.player)
        self.velocity = np.zeros(2, dtype=np.float32)
        self.goal = self._sample_point(self.player, 0.65, reachable_from=self.player)
        occupied = [self.player, self.goal]
        self.object_pos = self._sample_point(self.player, 0.30, reachable_from=self.player, extra_away=occupied)
        occupied.append(self.object_pos)
        self.key_pos = self._sample_point(self.player, 0.30, reachable_from=self.player, extra_away=occupied)
        occupied.append(self.key_pos)
        self.door_pos = self._sample_point(self.goal, 0.18, reachable_from=self.player, extra_away=[self.player, self.key_pos])
        occupied.append(self.door_pos)
        self.enemy_pos = self._sample_point(self.player, 0.55, reachable_from=self.player, extra_away=occupied)
        self.enemy_health = 1.0
        self.health = 1.0
        self.ammo = 1.0
        self.cooldown = 0.0
        self.has_key = False
        self.door_open = False
        self.object_present = self._needs_object()
        self.enemy_present = self._needs_enemy()
        self.collected_count = 0
        self.target_count = self.program.target_count
        self.safe_streak = 0
        self.cover_reached = False
        self.last_collision = False
        self.last_damage = 0.0
        self.last_collected = False
        self.last_key_collected = False
        self.last_door_opened = False
        self.last_enemy_defeated = False
        self._done = False
        self._success = False
        hide = self.task.environment.get("hide_goal_after_steps")
        self._goal_reveal_steps = self.horizon + 1 if hide is None else max(1, int(hide))
        self._advance_objectives()
        return self._obs()

    def _goal_visible(self) -> bool:
        return self.t < self._goal_reveal_steps

    def goal_vector(self) -> np.ndarray:
        return self.program.vector()

    @property
    def active_objective(self) -> str:
        return self.program.active

    def _objective_target(self, objective: str | None = None) -> np.ndarray | None:
        obj = self.active_objective if objective is None else objective
        if obj in {"reach_goal", "remember_goal", "survive"}:
            return self.goal.copy()
        if obj in {"collect_object", "collect_resource"}:
            return self.object_pos.copy()
        if obj == "defeat_enemy" or obj == "avoid_enemy":
            return self.enemy_pos.copy()
        if obj == "take_cover":
            return self.cover.copy()
        if obj == "collect_key":
            return self.key_pos.copy()
        if obj == "open_door":
            return self.door_pos.copy()
        return None

    def _objective_complete(self, objective: str) -> bool:
        if objective == "reach_goal":
            return float(np.linalg.norm(self.goal - self.player)) <= 0.10
        if objective == "remember_goal":
            return self.t >= self._goal_reveal_steps
        if objective == "collect_object":
            return self.collected_count >= self.target_count
        if objective == "collect_resource":
            return self.last_collected or self.collected_count >= 1
        if objective == "defeat_enemy":
            return not self.enemy_present
        if objective == "collect_key":
            return self.has_key
        if objective == "open_door":
            return self.door_open
        if objective == "take_cover":
            return self.cover_reached
        if objective == "avoid_enemy":
            return self.safe_streak >= 4
        if objective == "survive":
            return self.t >= int(self.task.goal.get("survive_steps", 20)) and self.health > 0
        return False

    def _advance_objectives(self) -> bool:
        advanced = False
        # Multiple already-satisfied prerequisites can be consumed in one transition.
        while not self.program.done and self._objective_complete(self.program.active):
            self.program.mark_complete(); advanced = True
        return advanced

    def _nearest_obstacle(self) -> tuple[np.ndarray, float]:
        rows = []
        for center, radius in self.obstacles:
            delta = center - self.player
            rows.append((float(np.linalg.norm(delta)) - radius, delta))
        if self._needs_key_door() and not self.door_open:
            delta = self.door_pos - self.player
            rows.append((float(np.linalg.norm(delta)) - 0.10, delta))
        if not rows:
            return np.zeros(2, dtype=np.float32), 2.0
        dist, delta = min(rows, key=lambda x: x[0])
        return delta.astype(np.float32), float(dist)

    def _obs(self) -> np.ndarray:
        o = np.zeros(OBS_DIM, dtype=np.float32)
        o[0:2] = self.player
        o[2:4] = self.velocity
        o[4] = self.health
        o[5] = self.ammo
        o[6] = self.cooldown
        # v2.10: stage id is deliberately absent. This is objective progress, not curriculum metadata.
        o[7] = self.program.progress
        if self._goal_visible():
            o[8:10] = self.goal
            o[10] = 1.0
        o[11] = float(self.has_key)
        o[12] = float(self.door_open)
        o[13] = float(self.object_present)
        o[14] = float(self.enemy_present)
        o[15] = float(self.enemy_health if self.enemy_present else 0.0)
        if self.object_present:
            o[16:18] = self.object_pos
        if self.enemy_present:
            o[18:20] = self.enemy_pos
        if self._needs_key_door() and not self.has_key:
            o[20:22] = self.key_pos
        if self._needs_key_door():
            o[22:24] = self.door_pos
        delta, distance = self._nearest_obstacle()
        o[24:26] = delta
        o[26] = float(np.clip(distance, -1.0, 2.0))
        o[27] = _clip01(1.0 - float(np.linalg.norm(self.player - self.cover)) / self.cover_radius)
        o[28] = self.t / max(1, self.horizon)
        if bool(self.task.environment.get("reveal_dynamics", False)):
            o[29] = float(self.task.environment.get("gravity_scale", 1.0))
            o[30] = float(self.task.environment.get("friction_scale", 1.0))
            o[31] = float(self.task.environment.get("movement_scale", 1.0))
        return o

    def render_rgb(self, size: int = 64) -> np.ndarray:
        size = int(size)
        if size < 16:
            raise ValueError("render size must be >= 16")
        texture_seed = int(self.task.environment.get("texture_seed", 0))
        lighting_seed = int(self.task.environment.get("lighting_seed", 0))
        rr = np.random.default_rng(texture_seed ^ (lighting_seed << 1))
        base = rr.integers(18, 38, size=3, dtype=np.uint8)
        light = float(0.78 + 0.35 * ((lighting_seed % 997) / 996.0))
        bg = np.clip(base.astype(np.float32) * light, 0, 255).astype(np.uint8)
        img = np.zeros((size, size, 3), dtype=np.uint8); img[...] = bg
        def px(p):
            q = ((np.asarray(p) + 1.0) * 0.5 * (size - 1)).astype(int)
            return int(np.clip(q[0], 0, size - 1)), int(np.clip(q[1], 0, size - 1))
        def dot(p, color, rad=2):
            x, y = px(p)
            img[max(0,y-rad):min(size,y+rad+1), max(0,x-rad):min(size,x+rad+1)] = color
        for c, r in self.obstacles:
            x, y = px(c); pr = max(2, int(r * size * 0.5))
            img[max(0,y-pr):min(size,y+pr+1), max(0,x-pr):min(size,x+pr+1)] = np.clip(np.asarray([70,76,88]) + rr.integers(-12,13,size=3),0,255)
        if self._needs_cover(): dot(self.cover, [50,90,150], 3)
        if self._goal_visible(): dot(self.goal, [80,220,100], 3)
        if self.object_present: dot(self.object_pos, [240,210,60], 2)
        if self._needs_key_door() and not self.has_key: dot(self.key_pos, [220,160,50], 2)
        if self._needs_key_door(): dot(self.door_pos, [150,80,210] if not self.door_open else [90,170,210], 3)
        if self.enemy_present: dot(self.enemy_pos, [220,65,65], 3)
        dot(self.player, [80,170,255], 3)
        return img

    def _collides(self, p: np.ndarray) -> bool:
        for center, radius in self.obstacles:
            if float(np.linalg.norm(p - center)) < radius:
                return True
        if self._needs_key_door() and not self.door_open:
            if float(np.linalg.norm(p - self.door_pos)) < 0.10:
                return True
        return bool(np.any(np.abs(p) > 0.98))

    def _potential_distance(self, objective: str, target: np.ndarray | None) -> float:
        if target is None:
            return 0.0
        if objective == "avoid_enemy":
            # Larger separation is better, capped so the agent still progresses after satisfying safety.
            return -min(float(np.linalg.norm(self.player - target)), 0.55)
        return float(np.linalg.norm(self.player - target))

    def step(self, action):
        if self._done:
            raise RuntimeError("step() called after terminal state; call reset()")
        a = np.asarray(action, dtype=np.float32).reshape(ACTION_DIM)
        a = np.clip(a, self.action_low, self.action_high)
        hybrid = self.action_codec.decode(a)
        move = np.asarray(hybrid.movement, dtype=np.float32)
        attack = hybrid.attack
        interact = hybrid.interact
        self.last_collision = self.last_collected = self.last_key_collected = False
        self.last_door_opened = self.last_enemy_defeated = False
        self.last_damage = 0.0

        active_before = self.active_objective
        target_before = self._objective_target(active_before)
        potential_before = self._potential_distance(active_before, target_before)

        friction = max(0.2, float(self.task.environment.get("friction_scale", 1.0)))
        movement = max(0.2, float(self.task.environment.get("movement_scale", 1.0)))
        gravity = float(self.task.environment.get("gravity_scale", 1.0))
        accel = 0.12 * movement / friction
        gravity_drift = np.asarray([0.0, -0.010 * (gravity - 1.0)], dtype=np.float32)
        self.velocity = (0.55 * self.velocity + accel * move + gravity_drift).astype(np.float32)
        proposed = (self.player + self.velocity).astype(np.float32)
        if self._collides(proposed):
            self.last_collision = True
            x_only = (self.player + np.asarray([self.velocity[0], 0.0], dtype=np.float32)).astype(np.float32)
            y_only = (self.player + np.asarray([0.0, self.velocity[1]], dtype=np.float32)).astype(np.float32)
            if not self._collides(x_only):
                self.player = x_only; self.velocity[1] = 0.0
            elif not self._collides(y_only):
                self.player = y_only; self.velocity[0] = 0.0
            else:
                self.velocity *= -0.10
        else:
            self.player = proposed

        if interact and self.object_present and float(np.linalg.norm(self.player - self.object_pos)) <= 0.14:
            self.collected_count += 1; self.last_collected = True
            if "collect_resource" in self._objectives():
                self.ammo = min(1.0, self.ammo + 0.5); self.health = min(1.0, self.health + 0.25)
            if self.collected_count >= self.target_count:
                self.object_present = False
            else:
                self.object_pos = self._sample_point(self.player, 0.30, reachable_from=self.player, extra_away=[self.goal])
        if self._needs_key_door() and interact and not self.has_key and float(np.linalg.norm(self.player - self.key_pos)) <= 0.14:
            self.has_key = True; self.last_key_collected = True
        if self._needs_key_door() and interact and self.has_key and not self.door_open and float(np.linalg.norm(self.player - self.door_pos)) <= 0.22:
            self.door_open = True; self.last_door_opened = True

        if self.cooldown > 0:
            self.cooldown = max(0.0, self.cooldown - 0.12)
        if self.enemy_present:
            aggression = float(self.task.environment.get("enemy_aggression", 0.45))
            chase = _unit(self.player - self.enemy_pos)
            self.enemy_pos = np.clip(self.enemy_pos + (0.018 + 0.035 * aggression) * chase, -0.95, 0.95).astype(np.float32)
            enemy_dist = float(np.linalg.norm(self.player - self.enemy_pos))
            if attack and self.ammo > 0.02 and self.cooldown <= 0 and enemy_dist <= 0.32:
                self.ammo = max(0.0, self.ammo - 0.12); self.cooldown = 0.35; self.enemy_health -= 0.35
                if self.enemy_health <= 0:
                    self.enemy_present = False; self.enemy_health = 0.0; self.last_enemy_defeated = True
            if self.enemy_present and enemy_dist <= 0.16:
                cover = _clip01(1.0 - float(np.linalg.norm(self.player - self.cover)) / self.cover_radius) if self._needs_cover() else 0.0
                dmg = (0.025 + 0.04 * aggression) * (1.0 - 0.75 * cover)
                self.health = max(0.0, self.health - dmg); self.last_damage = float(dmg)
            if self.enemy_present and enemy_dist >= 0.34 and self.last_damage == 0.0:
                self.safe_streak += 1
            else:
                self.safe_streak = 0
        else:
            self.safe_streak += 1
        if self._needs_cover() and float(np.linalg.norm(self.player - self.cover)) <= self.cover_radius:
            self.cover_reached = True

        self.t += 1
        # Reward progress toward the objective that was active before this action.
        potential_after = self._potential_distance(active_before, target_before)
        progress = potential_before - potential_after
        reward = 1.5 * progress - 0.004 * float(np.square(a[:2]).sum())
        if self.last_collision: reward -= 0.08
        if self.last_damage > 0: reward -= 2.0 * self.last_damage
        if self.last_collected: reward += 0.35
        if self.last_key_collected: reward += 0.25
        if self.last_door_opened: reward += 0.35
        if self.last_enemy_defeated: reward += 0.55

        advanced = self._advance_objectives()
        if advanced:
            reward += 0.20
        self._success = self.program.done
        failure = self.health <= 0.0
        timeout = self.t >= self.horizon
        self._done = bool(self._success or failure or timeout)
        if self._success: reward += 2.0
        if failure: reward -= 1.0
        info = ArenaTransitionInfo(
            self._success, self.last_collision, self.last_damage, self.last_collected,
            self.last_key_collected, self.last_door_opened, self.last_enemy_defeated,
            float(np.linalg.norm(self.goal - self.player)), failure,
            self.active_objective, advanced,
        )
        return self._obs(), float(reward), self._done, info.__dict__.copy()

    def constraint_labels(self) -> np.ndarray:
        return np.asarray([
            float(self.last_collision),
            float(self.last_damage > 0),
            float(self.health <= 0),
            float(self.enemy_present and np.linalg.norm(self.player - self.enemy_pos) <= 0.18),
        ], dtype=np.float32)

    def state_dict(self):
        return {
            "task_id": self.task.task_id, "t": self.t,
            "player": self.player.copy(), "velocity": self.velocity.copy(), "goal": self.goal.copy(),
            "object_pos": self.object_pos.copy(), "key_pos": self.key_pos.copy(), "door_pos": self.door_pos.copy(),
            "enemy_pos": self.enemy_pos.copy(), "enemy_health": self.enemy_health, "health": self.health,
            "ammo": self.ammo, "cooldown": self.cooldown, "has_key": self.has_key, "door_open": self.door_open,
            "object_present": self.object_present, "enemy_present": self.enemy_present,
            "collected_count": self.collected_count, "target_count": self.target_count,
            "safe_streak": self.safe_streak, "cover_reached": self.cover_reached,
            "goal_program": self.program.state_dict(),
            "last_collision": self.last_collision, "last_damage": self.last_damage,
            "last_collected": self.last_collected, "last_key_collected": self.last_key_collected,
            "last_door_opened": self.last_door_opened, "last_enemy_defeated": self.last_enemy_defeated,
            "done": self._done, "success": self._success, "goal_reveal_steps": self._goal_reveal_steps,
            "rng_state": copy.deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state):
        if state.get("task_id") != self.task.task_id:
            raise ValueError("snapshot task_id does not match environment task")
        self.t = int(state["t"])
        for name in ("player", "velocity", "goal", "object_pos", "key_pos", "door_pos", "enemy_pos"):
            setattr(self, name, np.asarray(state[name], dtype=np.float32).copy())
        for name in ("enemy_health", "health", "ammo", "cooldown", "last_damage"):
            setattr(self, name, float(state[name]))
        for name in ("has_key", "door_open", "object_present", "enemy_present", "last_collision", "last_collected", "last_key_collected", "last_door_opened", "last_enemy_defeated", "cover_reached"):
            setattr(self, name, bool(state[name]))
        self.collected_count = int(state["collected_count"]); self.target_count = int(state["target_count"])
        self.safe_streak = int(state.get("safe_streak", 0))
        self.program = GoalProgram.load(state["goal_program"])
        self._done = bool(state["done"]); self._success = bool(state["success"])
        self._goal_reveal_steps = int(state["goal_reveal_steps"])
        self.rng.bit_generator.state = copy.deepcopy(state["rng_state"])
        return self
