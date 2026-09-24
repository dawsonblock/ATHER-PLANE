from __future__ import annotations
import numpy as np


class DelayedCueEnv:
    """
    Partially observable memory task.

    At reset, a binary cue is shown in obs[0:2].
    After the cue disappears, the agent must wait.
    On the terminal decision step:
      action 0 = choose cue 0
      action 1 = choose cue 1
      action 2 = wait/no-op
    """

    def __init__(self, cue_delay=8, horizon=64, obs_dim=16, seed=0):
        self.cue_delay = cue_delay
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.cue = 0

    @property
    def action_dim(self):
        return 3

    def reset(self):
        self.t = 0
        self.cue = int(self.rng.integers(0, 2))
        return self._obs()

    def _obs(self):
        x = self.rng.normal(0, 0.02, size=self.obs_dim).astype(np.float32)
        if self.t == 0:
            x[self.cue] = 1.0
        if self.t == self.cue_delay:
            x[2] = 1.0  # decision marker
        return x

    def step(self, action: int):
        reward = 0.0
        done = False
        if self.t == self.cue_delay:
            done = True
            reward = 1.0 if action == self.cue else -1.0
        elif action in (0, 1):
            reward = -0.02
        self.t += 1
        if self.t >= self.horizon:
            done = True
        return self._obs(), reward, done, {"cue": self.cue}

    def state_dict(self):
        return {"t": self.t, "cue": self.cue, "rng_state": self.rng.bit_generator.state}

    def load_state_dict(self, state):
        self.t = int(state["t"]); self.cue = int(state["cue"]); self.rng.bit_generator.state = state["rng_state"]
        return self
