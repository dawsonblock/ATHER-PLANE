"""Small paired raw-vs-clipped reward experiment for the v2.38.6 P1D gate."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from awa.v2.game.vizdoom_dataset import ViZDoomExplorerPolicy, collect_vizdoom_dataset
from awa.v2.game.vizdoom_env import ViZDoomAetherEnv, ViZDoomConfig, ViZDoomScenario
from awa.v2.game.vizdoom_qualification import assert_real_backend
from awa.v2.game.vizdoom_runtime import load_vizdoom_runtime
from awa.v2.game.vizdoom_training import train_vizdoom_stack, _sha256


@dataclass(frozen=True)
class RewardDiagnosticConfig:
    scenario: str = "my_way_home"
    track: str = "structured"
    collection_episodes: int = 12
    collection_horizon: int = 400
    minimum_collection_transitions: int = 2048
    collection_seed: int = 7350
    training_seeds: tuple[int, ...] = (7101, 7102, 7103)
    evaluation_seed_ids: tuple[int, ...] = (9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108)
    evaluation_horizon: int = 600
    sequence_length: int = 8
    world_epochs: int = 1
    actor_epochs: int = 2
    calibration_epochs: int = 1
    batch_size: int = 64
    hidden: int = 96
    device: str = "cuda"
    reward_clip_min: float = -1.0
    reward_clip_max: float = 1.0

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RewardDiagnosticConfig":
        values = dict(row)
        for key in ("training_seeds", "evaluation_seed_ids"):
            if key in values:
                values[key] = tuple(int(x) for x in values[key])
        return cls(**values)

    def __post_init__(self):
        if self.scenario != "my_way_home" or self.track != "structured":
            raise ValueError("P1D is restricted to structured my_way_home")
        if (self.collection_episodes < 3 or self.collection_horizon <= 0 or self.evaluation_horizon <= 0
            or self.minimum_collection_transitions <= 0):
            raise ValueError("P1D requires at least three collection episodes and positive horizons")
        if len(self.training_seeds) != 3 or len(set(self.training_seeds)) != 3:
            raise ValueError("P1D requires exactly three distinct paired training seeds")
        if len(self.evaluation_seed_ids) < 6 or len(set(self.evaluation_seed_ids)) != len(self.evaluation_seed_ids):
            raise ValueError("P1D requires at least six distinct fixed evaluation seeds")
        collection_seeds = {self.collection_seed + i * 7919 for i in range(self.collection_episodes)}
        if (collection_seeds & set(self.evaluation_seed_ids)
            or set(self.training_seeds) & set(self.evaluation_seed_ids)
            or collection_seeds & set(self.training_seeds)):
            raise ValueError("P1D collection, model-initialization, and evaluation seeds must be disjoint")
        if self.reward_clip_min >= self.reward_clip_max:
            raise ValueError("reward clip bounds must be increasing")


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _action_key(action: np.ndarray) -> tuple[int, int, int, int]:
    return (int(action[0] > .25) - int(action[0] < -.25),
            int(action[1] > .25) - int(action[1] < -.25),
            int(action[2] > 0), int(action[3] > 0))


def _evaluate_actor(world: Path, actor: Path, cfg: RewardDiagnosticConfig,
                    env_factory, seed: int) -> dict[str, Any]:
    env = env_factory()
    controller = None
    try:
        obs, _ = env.reset(seed=seed)
        controller = load_vizdoom_runtime(world, actor, device=cfg.device, use_planner=False, risk_limit=1.0)
        controller.reset()
        total = 0.0
        rewards: list[float] = []
        actions: list[list[float]] = []
        info: dict[str, Any] = {}
        done = False
        last_terminated = False
        last_truncated = False
        for _ in range(cfg.evaluation_horizon):
            action, _ = controller.act(env, obs, force_planner=False)
            actions.append(np.asarray(action, dtype=np.float32).tolist())
            obs, reward, terminated, truncated, info = env.step(action)
            total += float(reward); rewards.append(float(reward))
            last_terminated = bool(terminated); last_truncated = bool(truncated)
            done = bool(terminated or truncated)
            if done:
                break
        counts = Counter(_action_key(np.asarray(a, dtype=np.float32)) for a in actions)
        n = max(1, len(actions))
        entropy = -sum((count / n) * np.log(count / n) for count in counts.values() if count)
        return {
            "seed": int(seed), "success": bool(info.get("success", False)), "return": float(total),
            "steps": len(actions), "terminated": last_terminated, "truncated": last_truncated,
            "action_histogram": {"|".join(map(str, key)): int(value) for key, value in sorted(counts.items())},
            "action_entropy_nats": float(entropy),
            "action_distribution": {"turn_left_fraction": float(np.mean(np.asarray(actions)[:, 0] < -.25)) if actions else 0.0,
                                    "turn_right_fraction": float(np.mean(np.asarray(actions)[:, 0] > .25)) if actions else 0.0,
                                    "forward_fraction": float(np.mean(np.asarray(actions)[:, 1] > .25)) if actions else 0.0,
                                    "backward_fraction": float(np.mean(np.asarray(actions)[:, 1] < -.25)) if actions else 0.0,
                                    "attack_fraction": float(np.mean(np.asarray(actions)[:, 2] > 0)) if actions else 0.0,
                                    "use_fraction": float(np.mean(np.asarray(actions)[:, 3] > 0)) if actions else 0.0},
            "reward_distribution": {"count": len(rewards), "min": float(min(rewards, default=0.0)),
                                    "max": float(max(rewards, default=0.0)), "mean": float(np.mean(rewards)) if rewards else 0.0,
                                    "positive_fraction": float(np.mean(np.asarray(rewards) > 0)) if rewards else 0.0},
        }
    finally:
        env.close()


def _reward_component_statistics(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    rewards = np.asarray(arrays["rewards"], dtype=np.float64)
    def stat(values, semantics):
        x = np.asarray(values)
        return {"count": int(x.size), "mean": float(x.mean()) if x.size else 0.0,
                "min": float(x.min()) if x.size else 0.0, "max": float(x.max()) if x.size else 0.0,
                "semantics": semantics}
    return {
        "terminal_success": {"count": int(np.asarray(arrays["terminal_success"], dtype=bool).sum()),
                             "semantics": "final-transition success flag emitted by ViZDoomScenarioSemantics"},
        "collision_damage": stat(np.minimum(np.asarray(arrays["health_delta_proxy"], dtype=np.float64), 0.0),
                                 "negative health-change proxy; not an engine reward component"),
        "living_step_cost": stat(np.asarray(arrays["living_step_reward_proxy"], dtype=np.float64),
                                 "nonterminal reward proxy; not an engine reward component"),
        "progress_shaping": stat(np.asarray(arrays["displacement_proxy"], dtype=np.float64),
                                 "POSITION_X/Y displacement proxy, not goal-directed progress"),
        "timeout": {"count": int(np.asarray(arrays["episode_timeout"], dtype=bool).sum()),
                    "semantics": "truncated final-transition flag"},
        "raw_environment_reward": {"count": int(rewards.size), "min": float(rewards.min()),
                                   "max": float(rewards.max()), "mean": float(rewards.mean()),
                                   "std": float(rewards.std()), "positive_fraction": float((rewards > 0).mean()),
                                   "semantics": "actual scalar reward returned by ViZDoom"},
    }


def run_reward_diagnostic(config_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    raw_cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
    cfg = RewardDiagnosticConfig.from_dict(raw_cfg.get("diagnostic") or {})
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    scenario = ViZDoomScenario(cfg.scenario)

    def env_factory(max_steps: int):
        return ViZDoomAetherEnv(ViZDoomConfig(scenario=scenario, track="structured", frame_skip=4,
                                             max_episode_steps=max_steps, visible=False, sound=False))

    probe = env_factory(cfg.evaluation_horizon)
    try:
        backend = assert_real_backend(probe)
    finally:
        probe.close()
    dataset_path = out / "my_way_home_shared_dataset.npz"
    collection = collect_vizdoom_dataset(lambda: env_factory(cfg.collection_horizon), dataset_path, episodes=cfg.collection_episodes,
                                         horizon=cfg.collection_horizon, base_seed=cfg.collection_seed,
                                         policy=ViZDoomExplorerPolicy(cfg.collection_seed))
    if collection.transitions < cfg.minimum_collection_transitions:
        raise RuntimeError(f"P1D collected only {collection.transitions} transitions; need {cfg.minimum_collection_transitions}")
    with np.load(dataset_path, allow_pickle=False) as data:
        arrays = {key: np.asarray(data[key]) for key in data.files}
    if "scenario_ids" not in arrays or set(str(x) for x in arrays["scenario_ids"].tolist()) != {"my_way_home"}:
        raise RuntimeError("P1D refuses a dataset containing any scenario other than my_way_home")
    source_data_sha = hashlib.sha256(b"".join(np.ascontiguousarray(arrays[k]).tobytes()
                                               for k in ("observations", "actions", "next_observations", "dones", "goals", "next_goals"))).hexdigest()
    arms: dict[str, Any] = {}
    for arm, rewards in (("raw_reward", arrays["rewards"].astype(np.float32)),
                         ("clipped_reward", np.clip(arrays["rewards"], cfg.reward_clip_min, cfg.reward_clip_max).astype(np.float32))):
        arm_dir = out / arm; arm_dir.mkdir(parents=True, exist_ok=True)
        arm_dataset = arm_dir / "dataset.npz"
        changed = dict(arrays); changed["rewards"] = rewards
        np.savez_compressed(arm_dataset, **changed)
        per_seed = []
        for seed in cfg.training_seeds:
            seed_dir = arm_dir / f"seed-{seed}"
            train = train_vizdoom_stack(arm_dataset, seed_dir, track="structured",
                                        sequence_length=cfg.sequence_length, world_epochs=cfg.world_epochs,
                                        actor_epochs=cfg.actor_epochs, calibration_epochs=cfg.calibration_epochs,
                                        batch_size=cfg.batch_size, hidden=cfg.hidden, seed=seed,
                                        device=cfg.device, precision="fp32")
            episode_rows = [_evaluate_actor(seed_dir / "vizdoom_world.pt", seed_dir / "vizdoom_actor.pt",
                                            cfg, lambda: env_factory(cfg.evaluation_horizon), eval_seed) for eval_seed in cfg.evaluation_seed_ids]
            per_seed.append({
                "seed": int(seed), "dataset_sha256": _sha256(arm_dataset),
                "source_dataset_sha256": _sha256(dataset_path), "model_initialization_seed": int(seed),
                "shared_nonreward_data_sha256": source_data_sha,
                "training_report": train.to_dict(),
                "checkpoint_sha256": {"world": _sha256(seed_dir / "vizdoom_world.pt"),
                                      "actor": _sha256(seed_dir / "vizdoom_actor.pt")},
                "episodes": episode_rows,
            })
        arms[arm] = {"per_training_seed": per_seed,
                     "reward_distribution": {"count": int(rewards.size), "min": float(rewards.min()),
                                             "max": float(rewards.max()), "mean": float(rewards.mean()),
                                             "std": float(rewards.std()),
                                             "positive_fraction": float((rewards > 0).mean())},
                     "reward_transform": {"kind": "identity" if arm == "raw_reward" else "clip",
                                           "min": None if arm == "raw_reward" else cfg.reward_clip_min,
                                           "max": None if arm == "raw_reward" else cfg.reward_clip_max}}
    raw_report = {
        "format": "awa-v2.38.6-reward-diagnostic-raw-v1", "scenario": cfg.scenario, "track": cfg.track,
        "backend": backend, "training_seeds": list(cfg.training_seeds),
        "evaluation_seed_ids": list(cfg.evaluation_seed_ids),
        "fixed_contract": {"scenario": cfg.scenario, "track": cfg.track, "source_dataset_sha256": _sha256(dataset_path),
                           "shared_nonreward_data_sha256": source_data_sha, "optimizer": "train_vizdoom_stack-default",
                           "world_epochs": cfg.world_epochs, "actor_epochs": cfg.actor_epochs,
                           "calibration_epochs": cfg.calibration_epochs, "batch_size": cfg.batch_size,
                           "hidden": cfg.hidden, "sequence_length": cfg.sequence_length,
                           "paired_model_initialization_seeds": list(cfg.training_seeds),
                           "paired_evaluation_seed_ids": list(cfg.evaluation_seed_ids),
                           "checkpoint_selection": "final checkpoint at identical configured training updates"},
        "collection": collection.to_dict(), "transition_count": int(len(arrays["rewards"])),
        "scenario_transition_counts": {"my_way_home": int(len(arrays["rewards"]))},
        "reward_component_statistics": _reward_component_statistics(arrays), "arms": arms,
        "claim_boundary": "Evaluation success is the environment's existing my_way_home success flag; component proxies are labeled and are not asserted to be exact reward-code terms.",
    }
    raw_path = out / "reward_diagnostic_raw.json"
    raw_path.write_text(json.dumps(raw_report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    from .planner_diagnostic import write_reward_diagnostic_report
    return write_reward_diagnostic_report(raw_path, out / "reward_diagnostic.json")
