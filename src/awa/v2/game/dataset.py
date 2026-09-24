from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import numpy as np

from .procedural_arena import ProceduralArenaEnv, OBS_DIM, ACTION_DIM
from .goal_program import GOAL_DIM
from .teacher import LogicalArenaTeacher
from .collectors import collector_receipt
from awa.v2.curriculum.task_spec import TaskSpec


@dataclass(frozen=True)
class GameCollectionReport:
    episodes: int
    transitions: int
    successes: int
    success_rate: float
    mean_return: float
    stages: dict[str, int]

    def to_dict(self): return asdict(self)


def collect_game_dataset(
    tasks: list[TaskSpec],
    output: str | Path,
    *,
    episodes_per_task: int = 2,
    horizon: int = 120,
    policy=None,
    metadata_path: str | Path | None = None,
) -> GameCollectionReport:
    if not tasks:
        raise ValueError("at least one task is required")
    policy = policy or LogicalArenaTeacher()
    obs_rows=[]; next_rows=[]; actions=[]; rewards=[]; dones=[]; constraints=[]
    goals=[]; next_goals=[]; episode_ids=[]; task_ids=[]; stage_rows=[]
    returns=[]; successes=0; stages={}; episode_counter=0
    for task in tasks:
        stages[str(task.stage)] = stages.get(str(task.stage), 0) + int(episodes_per_task)
        for rep in range(int(episodes_per_task)):
            env = ProceduralArenaEnv(task, horizon=horizon)
            obs = env.reset(seed=task.seed + rep * 7919)
            if hasattr(policy, "reset_episode"):
                policy.reset_episode(env)
            done=False; total=0.0; step_idx=0
            while not done:
                goal = env.goal_vector().copy()
                action = np.asarray(policy.act(env, obs), dtype=np.float32).reshape(ACTION_DIM)
                nxt, reward, done, info = env.step(action)
                next_goal = env.goal_vector().copy()
                obs_rows.append(obs.copy()); next_rows.append(nxt.copy()); actions.append(action.copy())
                goals.append(goal); next_goals.append(next_goal)
                rewards.append(float(reward)); dones.append(bool(done)); constraints.append(env.constraint_labels())
                episode_ids.append(episode_counter); task_ids.append(task.task_id); stage_rows.append(task.stage)
                total += float(reward); obs = nxt; step_idx += 1
            returns.append(total); successes += int(bool(info.get("success", False))); episode_counter += 1
    path=Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.asarray(obs_rows,dtype=np.float32),
        goals=np.asarray(goals,dtype=np.float32),
        actions=np.asarray(actions,dtype=np.float32),
        rewards=np.asarray(rewards,dtype=np.float32),
        next_observations=np.asarray(next_rows,dtype=np.float32),
        next_goals=np.asarray(next_goals,dtype=np.float32),
        dones=np.asarray(dones,dtype=np.bool_),
        constraints=np.asarray(constraints,dtype=np.float32),
        episode_ids=np.asarray(episode_ids,dtype=np.int64),
        stages=np.asarray(stage_rows,dtype=np.int16),
    )
    report=GameCollectionReport(episode_counter,len(obs_rows),successes,successes/max(1,episode_counter),float(np.mean(returns)),stages)
    meta = {
        "format":"awa-procedural-game-dataset-v2",
        "observation_dim":OBS_DIM,
        "goal_dim":GOAL_DIM,
        "action_dim":ACTION_DIM,
        "tasks":[t.to_dict() for t in tasks],
        "task_ids":task_ids,
        "collector":collector_receipt(policy),
        "report":report.to_dict(),
    }
    mp = Path(metadata_path) if metadata_path is not None else path.with_suffix(".json")
    mp.write_text(json.dumps(meta,indent=2,sort_keys=True),encoding="utf-8")
    return report
