from __future__ import annotations
import numpy as np
from awa.environments.delayed_cue import DelayedCueEnv


class RobustDelayedCueEnv(DelayedCueEnv):
    """Delayed-cue task with controlled sensor corruption for robustness testing."""
    def __init__(self, *args, observation_noise: float = 0.0, dropout_prob: float = 0.0,
                 distractor_prob: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_noise = float(observation_noise)
        self.dropout_prob = float(dropout_prob)
        self.distractor_prob = float(distractor_prob)

    def _obs(self):
        x = super()._obs()
        if self.observation_noise > 0:
            x = x + self.rng.normal(0, self.observation_noise, size=x.shape).astype(np.float32)
        if self.dropout_prob > 0:
            mask = self.rng.random(x.shape) < self.dropout_prob
            x = x.copy(); x[mask] = 0.0
        if self.distractor_prob > 0 and self.rng.random() < self.distractor_prob and len(x) > 3:
            idx = int(self.rng.integers(3, len(x)))
            x[idx] += float(self.rng.choice([-1.0, 1.0]))
        return x.astype(np.float32)
