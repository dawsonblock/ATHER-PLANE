from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_KEYS = ("observations", "actions", "rewards", "next_observations", "dones")


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class OfflineDatasetManifest:
    path: str
    sha256: str
    transitions: int
    observation_shape: tuple[int, ...]
    action_shape: tuple[int, ...]
    constraints_shape: tuple[int, ...] | None = None
    telemetry_shape: tuple[int, ...] | None = None
    goal_shape: tuple[int, ...] | None = None


class OfflineTransitionDataset(Dataset):
    """Validated NPZ transition dataset for representation/world-model pretraining."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix.lower() != ".npz":
            raise ValueError("v2.1 offline ingestion currently supports .npz datasets")
        data = np.load(self.path, allow_pickle=False)
        missing = [k for k in REQUIRED_KEYS if k not in data]
        if missing:
            raise ValueError(f"dataset missing required keys: {missing}")
        arrays = {k: np.asarray(data[k]) for k in REQUIRED_KEYS}
        if "constraints" in data:
            arrays["constraints"] = np.asarray(data["constraints"])
        has_goals = "goals" in data or "next_goals" in data
        if has_goals:
            if "goals" not in data or "next_goals" not in data:
                raise ValueError("goal-conditioned datasets must provide both goals and next_goals")
            arrays["goals"] = np.asarray(data["goals"])
            arrays["next_goals"] = np.asarray(data["next_goals"])
            if arrays["goals"].shape[1:] != arrays["next_goals"].shape[1:]:
                raise ValueError("next_goals shape must match goals shape")
        if "sample_weights" in data:
            arrays["sample_weights"] = np.asarray(data["sample_weights"], dtype=np.float32).reshape(-1)
            if np.any(~np.isfinite(arrays["sample_weights"])) or np.any(arrays["sample_weights"] <= 0):
                raise ValueError("sample_weights must be finite and > 0")
        if "planner_used" in data:
            arrays["planner_used"] = np.asarray(data["planner_used"], dtype=np.bool_).reshape(-1)
        for key in ("episode_ids", "scenario_ids", "terminal_success", "player_dead", "episode_timeout",
                    "health_delta_proxy", "living_step_reward_proxy", "displacement_proxy"):
            if key in data:
                arrays[key] = np.asarray(data[key])
        has_telemetry = "telemetry" in data or "next_telemetry" in data
        if has_telemetry:
            if "telemetry" not in data or "next_telemetry" not in data:
                raise ValueError("game telemetry datasets must provide both telemetry and next_telemetry")
            arrays["telemetry"] = np.asarray(data["telemetry"])
            arrays["next_telemetry"] = np.asarray(data["next_telemetry"])
            if arrays["telemetry"].shape[1:] != arrays["next_telemetry"].shape[1:]:
                raise ValueError("next_telemetry shape must match telemetry shape")
        n = int(arrays["observations"].shape[0])
        if n <= 0:
            raise ValueError("dataset must contain at least one transition")
        if any(int(v.shape[0]) != n for v in arrays.values()):
            raise ValueError("all dataset arrays must have the same first dimension")
        if arrays["next_observations"].shape[1:] != arrays["observations"].shape[1:]:
            raise ValueError("next_observations shape must match observations shape")
        self.arrays = arrays
        self.manifest = OfflineDatasetManifest(
            path=str(self.path),
            sha256=file_sha256(self.path),
            transitions=n,
            observation_shape=tuple(int(v) for v in arrays["observations"].shape[1:]),
            action_shape=tuple(int(v) for v in arrays["actions"].shape[1:]),
            constraints_shape=(tuple(int(v) for v in arrays["constraints"].shape[1:]) if "constraints" in arrays else None),
            telemetry_shape=(tuple(int(v) for v in arrays["telemetry"].shape[1:]) if "telemetry" in arrays else None),
            goal_shape=(tuple(int(v) for v in arrays["goals"].shape[1:]) if "goals" in arrays else None),
        )

    def __len__(self) -> int:
        return self.manifest.transitions

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        row = {
            "observation": torch.as_tensor(self.arrays["observations"][idx]).float(),
            "action": torch.as_tensor(self.arrays["actions"][idx]).float(),
            "reward": torch.as_tensor(self.arrays["rewards"][idx]).float(),
            "next_observation": torch.as_tensor(self.arrays["next_observations"][idx]).float(),
            "done": torch.as_tensor(self.arrays["dones"][idx]).bool(),
        }
        if "constraints" in self.arrays:
            row["constraints"] = torch.as_tensor(self.arrays["constraints"][idx]).float()
        if "goals" in self.arrays:
            row["goal"] = torch.as_tensor(self.arrays["goals"][idx]).float()
            row["next_goal"] = torch.as_tensor(self.arrays["next_goals"][idx]).float()
        if "sample_weights" in self.arrays:
            row["sample_weight"] = torch.as_tensor(self.arrays["sample_weights"][idx]).float()
        if "telemetry" in self.arrays:
            row["telemetry"] = torch.as_tensor(self.arrays["telemetry"][idx]).float()
            row["next_telemetry"] = torch.as_tensor(self.arrays["next_telemetry"][idx]).float()
        return row

    def write_manifest(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.manifest.__dict__, indent=2, sort_keys=True), encoding="utf-8")


def write_synthetic_dataset(path: str | Path, n: int = 64, obs_dim: int = 12, action_dim: int = 2, seed: int = 7) -> Path:
    rng = np.random.default_rng(seed)
    observations = rng.normal(size=(n, obs_dim)).astype(np.float32)
    actions = np.tanh(rng.normal(size=(n, action_dim))).astype(np.float32)
    drift = np.zeros_like(observations)
    drift[:, :action_dim] = 0.1 * actions
    next_observations = observations + drift + rng.normal(scale=0.01, size=observations.shape).astype(np.float32)
    rewards = (-np.square(next_observations[:, :2]).sum(axis=1)).astype(np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    dones[-1] = True
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=observations,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        dones=dones,
    )
    return path


def precompute_representation_cache(dataset, encoder, cache, index_path: str | Path | None = None) -> dict:
    """Cache current and next observation frozen features for an offline dataset.

    The index binds every transition to content-addressed cache keys plus the exact
    dataset and encoder fingerprints. It is safe to regenerate: existing entries
    are reused.
    """
    from awa.v2.representation import module_fingerprint

    fp = module_fingerprint(encoder.backbone)
    observation_keys: list[str] = []
    next_observation_keys: list[str] = []
    hits = 0
    writes = 0
    for i in range(len(dataset)):
        row = dataset[i]
        for field, sink in (("observation", observation_keys), ("next_observation", next_observation_keys)):
            x = row[field].reshape(1, -1)
            key, _ = cache.key_for(x, fp)
            sink.append(key)
            found = cache.get(x, fp)
            if found is None:
                features = encoder.encode_features(x)
                cache.put(x, fp, features)
                writes += 1
            else:
                hits += 1
    index = {
        "format": "awa-representation-index-v1",
        "dataset_sha256": dataset.manifest.sha256,
        "encoder_fingerprint": fp,
        "transitions": len(dataset),
        "observation_keys": observation_keys,
        "next_observation_keys": next_observation_keys,
        "hits": hits,
        "writes": writes,
    }
    if index_path is not None:
        Path(index_path).write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")
    return index


def write_synthetic_sequence_dataset(
    path: str | Path,
    *,
    episodes: int = 8,
    episode_length: int = 16,
    obs_dim: int = 12,
    action_dim: int = 2,
    constraints: int = 4,
    seed: int = 19,
) -> Path:
    """Create a deterministic chain-consistent dataset for world/risk qualification."""
    if obs_dim < action_dim:
        raise ValueError("obs_dim must be >= action_dim")
    rng = np.random.default_rng(seed)
    obs_rows, action_rows, reward_rows, next_rows, done_rows, constraint_rows = [], [], [], [], [], []
    for _ in range(int(episodes)):
        state = rng.normal(scale=0.35, size=obs_dim).astype(np.float32)
        for t in range(int(episode_length)):
            action = np.tanh(rng.normal(size=action_dim)).astype(np.float32)
            next_state = state.copy()
            next_state[:action_dim] += 0.18 * action
            if obs_dim > action_dim:
                next_state[action_dim:] = 0.96 * state[action_dim:]
                next_state[action_dim] += 0.03 * float(action.mean())
            reward = -float(np.square(next_state[: min(3, obs_dim)]).sum()) - 0.01 * float(np.square(action).sum())
            done = t == int(episode_length) - 1
            labels = np.zeros(constraints, dtype=np.float32)
            if constraints > 0:
                labels[0] = float(np.linalg.norm(next_state[: min(2, obs_dim)]) > 1.0)
            if constraints > 1:
                labels[1] = float(next_state[0] > 0.75)
            if constraints > 2:
                labels[2] = float(done and reward < -0.5)
            if constraints > 3:
                labels[3] = float(np.abs(action).max() > 0.85)
            obs_rows.append(state.copy()); action_rows.append(action); reward_rows.append(reward)
            next_rows.append(next_state.copy()); done_rows.append(done); constraint_rows.append(labels)
            state = next_state
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.asarray(obs_rows, dtype=np.float32),
        actions=np.asarray(action_rows, dtype=np.float32),
        rewards=np.asarray(reward_rows, dtype=np.float32),
        next_observations=np.asarray(next_rows, dtype=np.float32),
        dones=np.asarray(done_rows, dtype=np.bool_),
        constraints=np.asarray(constraint_rows, dtype=np.float32),
    )
    return path


def write_synthetic_game_dataset(
    path: str | Path,
    *,
    episodes: int = 3,
    episode_length: int = 10,
    height: int = 16,
    width: int = 16,
    action_dim: int = 2,
    telemetry_dim: int = 4,
    seed: int = 25,
) -> Path:
    """Dependency-free RGB game-transition dataset with exact temporal chains.

    Frames contain a moving bright square; telemetry tracks normalized x/y position
    and velocity. It exists only to qualify video/cache plumbing, not intelligence.
    """
    rng = np.random.default_rng(seed)
    obs_rows=[]; next_rows=[]; action_rows=[]; reward_rows=[]; done_rows=[]
    tel_rows=[]; next_tel_rows=[]
    for _ in range(int(episodes)):
        pos = rng.uniform(.2, .8, size=2).astype(np.float32)
        vel = np.zeros(2, dtype=np.float32)
        def render(p):
            img=np.zeros((height,width,3),dtype=np.uint8)
            y=int(np.clip(round(float(p[1])*(height-1)),0,height-1))
            x=int(np.clip(round(float(p[0])*(width-1)),0,width-1))
            img[max(0,y-1):min(height,y+2),max(0,x-1):min(width,x+2),:]=255
            return img
        frame=render(pos)
        for t in range(int(episode_length)):
            action=np.tanh(rng.normal(size=action_dim)).astype(np.float32)
            vel[:min(2,action_dim)] = .75*vel[:min(2,action_dim)] + .08*action[:min(2,action_dim)]
            next_pos=np.clip(pos+vel,-.05,1.05).astype(np.float32)
            next_frame=render(next_pos)
            telemetry=np.concatenate([pos,vel])[:telemetry_dim].astype(np.float32)
            next_telemetry=np.concatenate([next_pos,vel])[:telemetry_dim].astype(np.float32)
            if telemetry.shape[0] < telemetry_dim:
                telemetry=np.pad(telemetry,(0,telemetry_dim-telemetry.shape[0]))
                next_telemetry=np.pad(next_telemetry,(0,telemetry_dim-next_telemetry.shape[0]))
            reward=-float(np.linalg.norm(next_pos-np.array([.8,.8],dtype=np.float32)))
            done=t==int(episode_length)-1
            obs_rows.append(frame); next_rows.append(next_frame); action_rows.append(action)
            reward_rows.append(reward); done_rows.append(done); tel_rows.append(telemetry); next_tel_rows.append(next_telemetry)
            pos=next_pos; frame=next_frame
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(
        path, observations=np.asarray(obs_rows,dtype=np.uint8), actions=np.asarray(action_rows,dtype=np.float32),
        rewards=np.asarray(reward_rows,dtype=np.float32), next_observations=np.asarray(next_rows,dtype=np.uint8),
        dones=np.asarray(done_rows,dtype=np.bool_), telemetry=np.asarray(tel_rows,dtype=np.float32),
        next_telemetry=np.asarray(next_tel_rows,dtype=np.float32),
    )
    return path
