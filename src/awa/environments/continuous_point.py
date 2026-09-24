from __future__ import annotations
import numpy as np


class ContinuousPointEnv:
    """Small dependency-free continuous-control benchmark.

    State is a 2-D point and 2-D target. Actions are bounded velocity commands.
    The task is intentionally simple enough for CI while exercising the full
    continuous actor/replay/world-model/planner path.
    """
    def __init__(self, horizon: int = 50, seed: int = 0, step_scale: float = 0.15,
                 success_radius: float = 0.10, action_dim: int = 2):
        if action_dim != 2:
            raise ValueError("ContinuousPointEnv currently uses action_dim=2")
        self.horizon = int(horizon)
        self.step_scale = float(step_scale)
        self.success_radius = float(success_radius)
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.position = np.zeros(2, dtype=np.float32)
        self.target = np.zeros(2, dtype=np.float32)

    @property
    def obs_dim(self): return 4
    @property
    def action_dim(self): return 2
    @property
    def action_low(self): return np.full(2, -1.0, dtype=np.float32)
    @property
    def action_high(self): return np.full(2, 1.0, dtype=np.float32)

    def _obs(self):
        return np.concatenate([self.position, self.target]).astype(np.float32)

    def reset(self):
        self.t = 0
        self.position = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
        self.target = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
        while np.linalg.norm(self.target - self.position) < 0.4:
            self.target = self.rng.uniform(-0.8, 0.8, size=2).astype(np.float32)
        return self._obs()

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(2)
        a = np.clip(a, self.action_low, self.action_high)
        previous = float(np.linalg.norm(self.target - self.position))
        self.position = np.clip(self.position + self.step_scale * a, -1.25, 1.25).astype(np.float32)
        self.t += 1
        distance = float(np.linalg.norm(self.target - self.position))
        success = distance <= self.success_radius
        done = success or self.t >= self.horizon
        progress = previous - distance
        reward = 2.0 * progress - 0.01 * float(np.square(a).sum())
        if success:
            reward += 1.0
        return self._obs(), float(reward), bool(done), {"success": bool(success), "distance": distance}

    def state_dict(self):
        return {"t": self.t, "position": self.position.copy(), "target": self.target.copy(),
                "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.t = int(state["t"])
        self.position = np.asarray(state["position"], dtype=np.float32).copy()
        self.target = np.asarray(state["target"], dtype=np.float32).copy()
        self.rng.bit_generator.state = state["rng_state"]
        return self
