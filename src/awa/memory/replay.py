from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
import torch


def _actions_tensor(actions, device):
    arr = np.asarray(actions)
    if arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer):
        return torch.as_tensor(arr, dtype=torch.long, device=device)
    return torch.as_tensor(arr, dtype=torch.float32, device=device)


@dataclass
class Transition:
    obs: np.ndarray
    action: int | np.ndarray
    reward: float
    next_obs: np.ndarray
    done: bool


@dataclass
class SequenceBatch:
    observations: torch.Tensor   # [B,T+1,D]
    actions: torch.Tensor        # [B,T]
    rewards: torch.Tensor        # [B,T,1]
    dones: torch.Tensor          # [B,T,1]
    sequence_ids: list[tuple[int, int]] | None = None
    importance_weights: torch.Tensor | None = None


class ReplayBuffer:
    """Legacy transition replay retained for compatibility."""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.data: list[Transition] = []
        self.pos = 0

    def __len__(self): return len(self.data)

    def add(self, transition: Transition):
        if len(self.data) < self.capacity: self.data.append(transition)
        else: self.data[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int, device):
        batch = random.sample(self.data, batch_size)
        obs = torch.as_tensor(np.stack([x.obs for x in batch]), dtype=torch.float32, device=device)
        nxt = torch.as_tensor(np.stack([x.next_obs for x in batch]), dtype=torch.float32, device=device)
        raw_actions = [x.action for x in batch]
        a = np.asarray(raw_actions)
        action = torch.as_tensor(a, dtype=torch.long if a.ndim == 1 and np.issubdtype(a.dtype, np.integer) else torch.float32, device=device)
        reward = torch.as_tensor([x.reward for x in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        done = torch.as_tensor([x.done for x in batch], dtype=torch.float32, device=device).unsqueeze(-1)
        return obs, action, reward, nxt, done


class EpisodeReplayBuffer:
    """Episode-aware replay that returns contiguous sequences without crossing terminals."""
    def __init__(self, capacity_transitions: int = 100_000):
        self.capacity = capacity_transitions
        self.episodes: list[list[Transition]] = []
        self.current: list[Transition] = []
        self.transitions = 0
        self._episode_serial = 0
        self._serials: list[int] = []

    def __len__(self): return self.transitions + len(self.current)

    def add(self, transition: Transition):
        self.current.append(transition)
        if transition.done:
            self.episodes.append(self.current)
            self._serials.append(self._episode_serial)
            self._episode_serial += 1
            self.transitions += len(self.current)
            self.current = []
            self._trim()

    def end_episode(self):
        if self.current:
            self.episodes.append(self.current)
            self._serials.append(self._episode_serial)
            self._episode_serial += 1
            self.transitions += len(self.current)
            self.current = []
            self._trim()

    def _trim(self):
        while self.transitions > self.capacity and self.episodes:
            old = self.episodes.pop(0)
            self._serials.pop(0)
            self.transitions -= len(old)

    def sequence_candidates(self, sequence_length: int):
        out = []
        for ei, ep in enumerate(self.episodes):
            for start in range(max(0, len(ep) - sequence_length + 1)):
                out.append((ei, start))
        return out

    def valid_sequences(self, sequence_length: int) -> int:
        return len(self.sequence_candidates(sequence_length))

    def can_sample(self, batch_size: int, sequence_length: int) -> bool:
        return self.valid_sequences(sequence_length) >= batch_size

    def _build_batch(self, picks, sequence_length, device, importance_weights=None):
        obs, actions, rewards, dones, ids = [], [], [], [], []
        for ei, start in picks:
            seq = self.episodes[ei][start:start + sequence_length]
            obs_seq = [seq[0].obs] + [tr.next_obs for tr in seq]
            obs.append(np.stack(obs_seq))
            actions.append([tr.action for tr in seq])
            rewards.append([[tr.reward] for tr in seq])
            dones.append([[float(tr.done)] for tr in seq])
            ids.append((self._serials[ei], start))
        iw = None if importance_weights is None else torch.as_tensor(importance_weights, dtype=torch.float32, device=device)
        return SequenceBatch(
            torch.as_tensor(np.stack(obs), dtype=torch.float32, device=device),
            _actions_tensor(actions, device),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device),
            ids, iw,
        )

    def sample_sequences(self, batch_size: int, sequence_length: int, device) -> SequenceBatch:
        candidates = self.sequence_candidates(sequence_length)
        if len(candidates) < batch_size:
            raise ValueError(f"Need {batch_size} sequences of length {sequence_length}; have {len(candidates)}")
        return self._build_batch(random.sample(candidates, batch_size), sequence_length, device)

    def state_dict(self):
        return {
            "capacity": self.capacity, "episodes": self.episodes, "current": self.current,
            "transitions": self.transitions, "episode_serial": self._episode_serial,
            "serials": self._serials, "type": type(self).__name__,
        }

    def load_state_dict(self, state):
        self.capacity = int(state["capacity"]); self.episodes = state["episodes"]; self.current = state["current"]
        self.transitions = int(state["transitions"]); self._episode_serial = int(state["episode_serial"])
        self._serials = list(state["serials"]); return self


class PrioritizedEpisodeReplayBuffer(EpisodeReplayBuffer):
    """Contiguous sequence replay with proportional prioritization.

    Priorities are keyed by stable `(episode_serial, start_index)` IDs. Sampling
    is proportional to priority**alpha and returns normalized importance weights.
    """
    def __init__(self, capacity_transitions: int = 100_000, alpha: float = 0.6,
                 beta: float = 0.4, epsilon: float = 1e-4):
        super().__init__(capacity_transitions)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.priorities: dict[tuple[int, int], float] = {}
        self.max_priority = 1.0

    def _trim(self):
        removed_serials = []
        while self.transitions > self.capacity and self.episodes:
            old = self.episodes.pop(0)
            serial = self._serials.pop(0)
            removed_serials.append(serial)
            self.transitions -= len(old)
        if removed_serials:
            removed = set(removed_serials)
            self.priorities = {k: v for k, v in self.priorities.items() if k[0] not in removed}

    def _stable_candidates(self, sequence_length: int):
        candidates = []
        stable_ids = []
        for ei, ep in enumerate(self.episodes):
            serial = self._serials[ei]
            for start in range(max(0, len(ep) - sequence_length + 1)):
                candidates.append((ei, start))
                stable_ids.append((serial, start))
        return candidates, stable_ids

    def sample_sequences(self, batch_size: int, sequence_length: int, device) -> SequenceBatch:
        candidates, ids = self._stable_candidates(sequence_length)
        if len(candidates) < batch_size:
            raise ValueError(f"Need {batch_size} sequences of length {sequence_length}; have {len(candidates)}")
        pri = np.asarray([self.priorities.get(i, self.max_priority) for i in ids], dtype=np.float64)
        probs = np.power(pri + self.epsilon, self.alpha)
        probs /= probs.sum()
        indices = np.random.choice(len(candidates), size=batch_size, replace=False, p=probs)
        picks = [candidates[i] for i in indices]
        picked_probs = probs[indices]
        weights = np.power(len(candidates) * picked_probs, -self.beta)
        weights /= weights.max() + 1e-12
        return self._build_batch(picks, sequence_length, device, weights)

    def update_priorities(self, sequence_ids: list[tuple[int, int]], errors):
        values = np.asarray(errors, dtype=np.float64).reshape(-1)
        if len(sequence_ids) != len(values):
            raise ValueError("sequence_ids and errors length mismatch")
        for sid, err in zip(sequence_ids, values):
            p = max(float(abs(err)) + self.epsilon, self.epsilon)
            self.priorities[sid] = p
            self.max_priority = max(self.max_priority, p)

    def state_dict(self):
        state = super().state_dict()
        state.update({"alpha": self.alpha, "beta": self.beta, "epsilon": self.epsilon,
                      "priorities": self.priorities, "max_priority": self.max_priority})
        return state

    def load_state_dict(self, state):
        super().load_state_dict(state)
        self.alpha = float(state.get("alpha", self.alpha)); self.beta = float(state.get("beta", self.beta))
        self.epsilon = float(state.get("epsilon", self.epsilon)); self.priorities = dict(state.get("priorities", {}))
        self.max_priority = float(state.get("max_priority", 1.0)); return self
