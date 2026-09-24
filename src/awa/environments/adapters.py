from __future__ import annotations
import numpy as np


def _flatten_obs(obs) -> np.ndarray:
    if isinstance(obs, dict):
        parts = [_flatten_obs(obs[k]) for k in sorted(obs)]
        return np.concatenate(parts, axis=0).astype(np.float32, copy=False)
    arr = np.asarray(obs, dtype=np.float32)
    return arr.reshape(-1)


class GymnasiumDiscreteAdapter:
    def __init__(self, env_id: str, seed: int = 0, **make_kwargs):
        try:
            import gymnasium as gym
        except ImportError as e:
            raise ImportError("Install with `pip install -e '.[gym]'`") from e
        self.gym = gym
        self.env = gym.make(env_id, **make_kwargs)
        if not isinstance(self.env.action_space, gym.spaces.Discrete):
            raise TypeError("Discrete adapter requires a gymnasium.spaces.Discrete action space")
        self.seed = seed
        first, _ = self.env.reset(seed=seed)
        self._obs_dim = int(_flatten_obs(first).size)

    @property
    def obs_dim(self): return self._obs_dim
    @property
    def action_dim(self): return int(self.env.action_space.n)
    @property
    def action_type(self): return "discrete"

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed); self.seed += 1
        return _flatten_obs(obs)

    def step(self, action: int):
        obs, reward, terminated, truncated, info = self.env.step(int(action))
        return _flatten_obs(obs), float(reward), bool(terminated or truncated), dict(info)

    def close(self): self.env.close()


class GymnasiumContinuousAdapter:
    def __init__(self, env_id: str, seed: int = 0, **make_kwargs):
        try:
            import gymnasium as gym
        except ImportError as e:
            raise ImportError("Install with `pip install -e '.[gym]'`") from e
        self.gym = gym
        self.env = gym.make(env_id, **make_kwargs)
        if not isinstance(self.env.action_space, gym.spaces.Box):
            raise TypeError("Continuous adapter requires a gymnasium.spaces.Box action space")
        if len(self.env.action_space.shape) != 1:
            raise TypeError("AWA continuous adapter currently requires a 1-D Box action space")
        if not np.all(np.isfinite(self.env.action_space.low)) or not np.all(np.isfinite(self.env.action_space.high)):
            raise TypeError("AWA continuous planning requires finite action bounds")
        self.seed = seed
        first, _ = self.env.reset(seed=seed)
        self._obs_dim = int(_flatten_obs(first).size)

    @property
    def obs_dim(self): return self._obs_dim
    @property
    def action_dim(self): return int(np.prod(self.env.action_space.shape))
    @property
    def action_low(self): return np.asarray(self.env.action_space.low, dtype=np.float32).reshape(-1)
    @property
    def action_high(self): return np.asarray(self.env.action_space.high, dtype=np.float32).reshape(-1)
    @property
    def action_type(self): return "continuous"

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed); self.seed += 1
        return _flatten_obs(obs)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(self.env.action_space.shape)
        a = np.clip(a, self.env.action_space.low, self.env.action_space.high)
        obs, reward, terminated, truncated, info = self.env.step(a)
        return _flatten_obs(obs), float(reward), bool(terminated or truncated), dict(info)

    def close(self): self.env.close()


class DMControlContinuousAdapter:
    """dm_control suite adapter exposing the same continuous interface as Gym."""
    def __init__(self, domain: str, task: str, seed: int = 0, task_kwargs: dict | None = None):
        try:
            from dm_control import suite
        except ImportError as e:
            raise ImportError("Install dm_control separately to use this adapter") from e
        kwargs = dict(task_kwargs or {})
        kwargs.setdefault("random", seed)
        self.env = suite.load(domain, task, task_kwargs=kwargs)
        ts = self.env.reset()
        self._obs_dim = int(_flatten_obs(ts.observation).size)
        spec = self.env.action_spec()
        self._low = np.asarray(spec.minimum, dtype=np.float32).reshape(-1)
        self._high = np.asarray(spec.maximum, dtype=np.float32).reshape(-1)

    @property
    def obs_dim(self): return self._obs_dim
    @property
    def action_dim(self): return int(self._low.size)
    @property
    def action_low(self): return self._low.copy()
    @property
    def action_high(self): return self._high.copy()
    @property
    def action_type(self): return "continuous"

    def reset(self): return _flatten_obs(self.env.reset().observation)

    def step(self, action):
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, self._low, self._high)
        ts = self.env.step(a)
        return _flatten_obs(ts.observation), float(ts.reward or 0.0), bool(ts.last()), {}

    def close(self):
        close = getattr(self.env, "close", None)
        if close is not None: close()

# Backward-compatible alias from v1.4/v1.5.
DMControlObservationAdapter = DMControlContinuousAdapter
