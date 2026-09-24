from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from multiprocessing import get_context
from typing import Any
import numpy as np

from awa.v2.curriculum.task_spec import TaskSpec
from awa.v2.game.procedural_arena import ProceduralArenaEnv
from awa.v2.game.teacher import LogicalArenaTeacher, RandomArenaPolicy


def _task_from_dict(raw: dict[str, Any]) -> TaskSpec:
    data = dict(raw); data["concepts"] = tuple(data.get("concepts", ()))
    return TaskSpec(**data)


def _rollout_job(job: dict[str, Any]) -> dict[str, Any]:
    task = _task_from_dict(job["task"]); seed = int(job["seed"]); horizon = int(job["horizon"])
    env = ProceduralArenaEnv(task, horizon=horizon); obs = env.reset(seed=seed)
    policy_kind = job["policy"]
    policy = LogicalArenaTeacher() if policy_kind == "teacher" else RandomArenaPolicy(seed)
    total = 0.0; steps = 0; success = False
    while steps < horizon:
        action = np.asarray(policy.act(env, obs), dtype=np.float32)
        obs, reward, done, info = env.step(action); total += float(reward); steps += 1
        if done:
            success = bool(info.get("success", False)); break
    return {
        "worker": int(job["worker"]), "episode": int(job["episode"]), "task_id": task.task_id,
        "seed": seed, "steps": steps, "return": total, "success": success,
        "policy_version": int(job["policy_version"]),
    }


@dataclass(frozen=True)
class ProcessRolloutReport:
    episodes: int
    successes: int
    success_rate: float
    mean_return: float
    total_steps: int
    workers: int
    policy_version: int
    rows: tuple[dict[str, Any], ...]

    def to_dict(self):
        d = asdict(self); d["rows"] = list(self.rows); return d


class ProcessArenaRolloutPool:
    """True-process rollout pool for the built-in procedural arena.

    It intentionally accepts serialized TaskSpecs and built-in policies so no opaque
    Python callable has to cross process boundaries.
    """
    def __init__(self, workers: int = 2, *, base_seed: int = 0, start_method: str = "spawn"):
        self.workers = max(1, int(workers)); self.base_seed = int(base_seed); self.start_method = str(start_method)
        if self.start_method not in {"spawn", "fork", "forkserver"}:
            raise ValueError("unsupported multiprocessing start method")

    def collect(self, tasks: list[TaskSpec], *, episodes_per_task: int = 1, horizon: int = 100, policy: str = "teacher", policy_version: int = 0) -> ProcessRolloutReport:
        if not tasks: raise ValueError("tasks required")
        if policy not in {"teacher", "random"}: raise ValueError("policy must be teacher or random")
        if int(episodes_per_task) <= 0 or int(horizon) <= 0: raise ValueError("episode/horizon values must be positive")
        jobs = []; idx = 0
        for task in tasks:
            for ep in range(int(episodes_per_task)):
                worker = idx % self.workers
                jobs.append({"task": task.to_dict(), "worker": worker, "episode": ep, "seed": self.base_seed + idx * 1009, "horizon": int(horizon), "policy": policy, "policy_version": int(policy_version)})
                idx += 1
        ctx = get_context(self.start_method)
        with ProcessPoolExecutor(max_workers=self.workers, mp_context=ctx) as ex:
            rows = list(ex.map(_rollout_job, jobs))
        successes = sum(int(r["success"]) for r in rows); returns = [float(r["return"]) for r in rows]
        return ProcessRolloutReport(len(rows), successes, successes / max(1, len(rows)), float(np.mean(returns)), sum(int(r["steps"]) for r in rows), self.workers, int(policy_version), tuple(rows))
