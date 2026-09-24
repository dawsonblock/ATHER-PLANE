from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable
import hashlib
import json

import numpy as np

from .vizdoom_env import ViZDoomAetherEnv, ViZDoomScenario, VIZDOOM_ACTION_DIM, VIZDOOM_GOAL_DIM


@dataclass(frozen=True)
class ViZDoomCollectionReport:
    scenario: str
    track: str
    episodes: int
    transitions: int
    successes: int
    success_rate: float
    mean_return: float
    frame_shape: tuple[int, int, int] | None
    telemetry_dim: int
    dataset_sha256: str

    def to_dict(self):
        row = asdict(self)
        if self.frame_shape is not None:
            row["frame_shape"] = list(self.frame_shape)
        return row


class ViZDoomExplorerPolicy:
    """Deterministic coverage policy; intentionally not advertised as an expert."""

    def __init__(self, seed: int = 215, attack_probability: float = 0.12, use_probability: float = 0.03):
        self.rng = np.random.default_rng(int(seed))
        self.attack_probability = float(attack_probability)
        self.use_probability = float(use_probability)
        self.step_index = 0

    def reset(self):
        self.step_index = 0

    def act(self, env: ViZDoomAetherEnv, observation: Any) -> np.ndarray:
        self.step_index += 1
        # Mostly move forward. Low-frequency changes of turn direction generate corridors/arcs
        # instead of white-noise spinning and are useful for initial representation datasets.
        turn_phase = (self.step_index // 24) % 4
        base_turn = (0.45, -0.35, 0.15, -0.1)[turn_phase]
        turn = float(np.clip(base_turn + self.rng.normal(0.0, 0.10), -1.0, 1.0))
        forward = float(np.clip(0.85 + self.rng.normal(0.0, 0.10), -1.0, 1.0))
        attack = self.rng.random() < self.attack_probability
        use = self.rng.random() < self.use_probability or self.step_index % 75 == 0
        return np.asarray([turn, forward, 1.0 if attack else -1.0, 1.0 if use else -1.0], dtype=np.float32)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_vizdoom_dataset(
    env_factory: Callable[[], ViZDoomAetherEnv],
    output: str | Path,
    *,
    episodes: int = 4,
    horizon: int = 600,
    base_seed: int = 215,
    policy: Any | None = None,
    metadata_path: str | Path | None = None,
) -> ViZDoomCollectionReport:
    if int(episodes) <= 0 or int(horizon) <= 0:
        raise ValueError("episodes and horizon must be positive")
    policy = policy or ViZDoomExplorerPolicy(base_seed)
    obs_rows=[]; next_rows=[]; telemetry=[]; next_telemetry=[]; goals=[]; next_goals=[]
    actions=[]; rewards=[]; dones=[]; constraints=[]; episode_ids=[]; planner_used=[]
    scenario_ids=[]; terminal_success=[]; player_dead_rows=[]; timeout_rows=[]
    health_delta=[]; living_step_proxy=[]; progress_distance_proxy=[]
    returns=[]; successes=0; scenario_name=None; track=None; frame_shape=None; telemetry_dim=0
    episode_counter=0
    for ep in range(int(episodes)):
        env = env_factory()
        try:
            if hasattr(policy, "reset"): policy.reset()
            obs, _ = env.reset(seed=int(base_seed) + ep * 7919)
            scenario_name = env.config.scenario.value; track = env.config.track
            if track == "pixel": frame_shape = tuple(int(x) for x in np.asarray(obs).shape)
            telemetry_dim = int(env.telemetry_dim)
            total=0.0; done=False; info={}
            for _ in range(int(horizon)):
                goal = env.goal_vector().copy(); tel = env.telemetry_vector().copy()
                action=np.asarray(policy.act(env,obs),dtype=np.float32).reshape(VIZDOOM_ACTION_DIM)
                nxt,reward,terminated,truncated,info=env.step(action); done=bool(terminated or truncated)
                next_goal=env.goal_vector().copy(); next_tel=env.telemetry_vector().copy()
                obs_rows.append(np.asarray(obs).copy()); next_rows.append(np.asarray(nxt).copy())
                telemetry.append(tel); next_telemetry.append(next_tel); goals.append(goal); next_goals.append(next_goal)
                actions.append(action); rewards.append(float(reward)); dones.append(done); constraints.append(env.constraint_labels())
                planner_used.append(bool(getattr(policy,"last_planner_used",False)))
                scenario_ids.append(str(env.config.scenario.value))
                is_last = bool(done)
                terminal_success.append(bool(is_last and info.get("success", False)))
                is_dead = bool(info.get("player_dead", False))
                player_dead_rows.append(is_dead)
                timeout_rows.append(bool(is_last and truncated and not terminated))
                # These are measured telemetry/reward proxies, not claimed engine reward components.
                health_delta.append(float(next_tel[0] - tel[0]))
                living_step_proxy.append(float(reward) if not is_last and not is_dead else 0.0)
                progress_distance_proxy.append(float(np.linalg.norm((next_tel[6:8] - tel[6:8]) * 1024.0)))
                episode_ids.append(episode_counter); total+=float(reward); obs=nxt
                if done: break
            returns.append(total); successes += int(bool(info.get("success",False))); episode_counter += 1
        finally:
            env.close()
    path=Path(output); path.parent.mkdir(parents=True,exist_ok=True)
    obs_array=np.asarray(obs_rows)
    next_array=np.asarray(next_rows)
    # Preserve uint8 pixels without inflating to float32. Structured track is always float32.
    if track == "structured":
        obs_array=obs_array.astype(np.float32); next_array=next_array.astype(np.float32)
    else:
        obs_array=obs_array.astype(np.uint8); next_array=next_array.astype(np.uint8)
    np.savez_compressed(
        path,
        observations=obs_array,
        telemetry=np.asarray(telemetry,dtype=np.float32),
        goals=np.asarray(goals,dtype=np.float32),
        actions=np.asarray(actions,dtype=np.float32),
        rewards=np.asarray(rewards,dtype=np.float32),
        next_observations=next_array,
        next_telemetry=np.asarray(next_telemetry,dtype=np.float32),
        next_goals=np.asarray(next_goals,dtype=np.float32),
        dones=np.asarray(dones,dtype=np.bool_),
        constraints=np.asarray(constraints,dtype=np.float32),
        episode_ids=np.asarray(episode_ids,dtype=np.int64),
        planner_used=np.asarray(planner_used,dtype=np.bool_),
        scenario_ids=np.asarray(scenario_ids,dtype="<U32"),
        terminal_success=np.asarray(terminal_success,dtype=np.bool_),
        player_dead=np.asarray(player_dead_rows,dtype=np.bool_),
        episode_timeout=np.asarray(timeout_rows,dtype=np.bool_),
        health_delta_proxy=np.asarray(health_delta,dtype=np.float32),
        living_step_reward_proxy=np.asarray(living_step_proxy,dtype=np.float32),
        displacement_proxy=np.asarray(progress_distance_proxy,dtype=np.float32),
    )
    sha=_sha256(path)
    report=ViZDoomCollectionReport(
        scenario=str(scenario_name),track=str(track),episodes=episode_counter,transitions=len(actions),successes=successes,
        success_rate=successes/max(1,episode_counter),mean_return=float(np.mean(returns)) if returns else 0.0,
        frame_shape=frame_shape,telemetry_dim=telemetry_dim,dataset_sha256=sha,
    )
    meta={
        "format":"awa-vizdoom-dataset-v1",
        "scenario":scenario_name,"track":track,"observation_shape":list(obs_array.shape[1:]),
        "telemetry_dim":telemetry_dim,"goal_dim":VIZDOOM_GOAL_DIM,"action_dim":VIZDOOM_ACTION_DIM,
        "base_seed":int(base_seed),"episodes":int(episodes),"horizon":int(horizon),"report":report.to_dict(),
        "pixel_dtype":str(obs_array.dtype),
        "scenario_ids": sorted(set(scenario_ids)),
        "reward_component_statistics": {
            "environment_reward": {"count": int(len(rewards)), "min": float(np.min(rewards)),
                                    "max": float(np.max(rewards)), "mean": float(np.mean(rewards)),
                                    "positive_fraction": float(np.mean(np.asarray(rewards) > 0.0))},
            "terminal_success": {"count": int(sum(terminal_success)), "semantics": "final transition with env success flag"},
            "player_dead": {"count": int(sum(player_dead_rows)), "semantics": "environment player-death flag"},
            "timeout": {"count": int(sum(timeout_rows)), "semantics": "truncated final transition"},
            "health_delta_proxy": {"negative_count": int(sum(x < 0 for x in health_delta)),
                                   "mean": float(np.mean(health_delta)), "semantics": "telemetry health change; not a reward component"},
            "living_step_reward_proxy": {"mean": float(np.mean(living_step_proxy)),
                                         "semantics": "non-terminal, non-death environment reward proxy"},
            "displacement_proxy": {"mean": float(np.mean(progress_distance_proxy)),
                                   "semantics": "Euclidean POSITION_X/Y telemetry displacement; not goal progress"},
        },
    }
    mp=Path(metadata_path) if metadata_path is not None else path.with_suffix(".json")
    mp.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report
