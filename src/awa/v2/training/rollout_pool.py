from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import numpy as np

from .staleness import RolloutEnvelope


@dataclass(frozen=True)
class RolloutTransition:
    observation: np.ndarray
    action: np.ndarray
    reward: float
    next_observation: np.ndarray
    done: bool
    info: dict


class LocalRolloutWorkerPool:
    """Small local rollout pool for game/simulator qualification."""

    def __init__(self, workers: int = 4, base_seed: int = 0):
        self.workers = max(1, int(workers))
        self.base_seed = int(base_seed)

    @staticmethod
    def _reset(env, seed=None):
        try:
            out = env.reset(seed=seed) if seed is not None else env.reset()
        except TypeError:
            out = env.reset()
        return out[0] if isinstance(out, tuple) and len(out) == 2 else out

    @staticmethod
    def _step(env, action):
        out = env.step(action)
        if not isinstance(out, tuple):
            raise TypeError("environment step() must return a tuple")
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return obs, reward, bool(terminated or truncated), info
        if len(out) == 4:
            obs, reward, done, info = out
            return obs, reward, bool(done), info
        raise ValueError(f"unsupported environment step() result length: {len(out)}")

    def _episode(
        self,
        env_factory,
        policy,
        worker_id: int,
        episode_index: int,
        max_steps: int,
        policy_version: int,
    ):
        seed = self.base_seed + worker_id * 1_000_003 + episode_index * 997
        env = env_factory(worker_id, seed)
        transitions = []
        try:
            obs = self._reset(env, seed=seed)
            for _ in range(int(max_steps)):
                action = np.asarray(policy(obs), dtype=np.float32)
                next_obs, reward, done, info = self._step(env, action)
                transitions.append(
                    RolloutTransition(
                        np.asarray(obs),
                        action,
                        float(reward),
                        np.asarray(next_obs),
                        bool(done),
                        dict(info or {}),
                    )
                )
                obs = next_obs
                if done:
                    break
        finally:
            close = getattr(env, "close", None)
            if callable(close):
                close()
        episode_id = f"w{worker_id}-e{episode_index}-s{seed}"
        return RolloutEnvelope(
            transitions,
            int(policy_version),
            worker_id=f"worker-{worker_id}",
            episode_id=episode_id,
        )

    def collect(
        self,
        env_factory,
        policy,
        *,
        episodes_per_worker: int = 1,
        max_steps: int = 1000,
        policy_version: int = 0,
    ) -> list[RolloutEnvelope]:
        episodes_per_worker = int(episodes_per_worker)
        max_steps = int(max_steps)
        if episodes_per_worker < 0:
            raise ValueError("episodes_per_worker must be >= 0")
        if max_steps <= 0:
            raise ValueError("max_steps must be > 0")
        jobs = [
            (w, e)
            for w in range(self.workers)
            for e in range(episodes_per_worker)
        ]
        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            fut = [
                ex.submit(
                    self._episode,
                    env_factory,
                    policy,
                    w,
                    e,
                    max_steps,
                    policy_version,
                )
                for w, e in jobs
            ]
            return [f.result() for f in fut]
