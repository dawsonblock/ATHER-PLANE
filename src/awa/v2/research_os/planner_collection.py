"""Collect real, reset-replayed ViZDoom evidence for the P1P decision gate.

This is deliberately costly: the environment oracle replays candidate branches
from a fresh seeded episode. A Doom save/load is not a valid counterfactual
because it leaves episode reward and clock state intact.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np
import torch

from .branch_replay import _observation_sha256, trace_on_env, validate_vizdoom_reset_replay_report
from .planner_diagnostic import FORMAT_RAW, HORIZONS, PROPOSALS, TESTS, build_planner_diagnostic_report


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_atomic(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def _key(seed: int, horizon: int, proposal: str) -> tuple[int, int, str]:
    return int(seed), int(horizon), str(proposal)


def maximum_branch_steps(seeds: int, candidates: int, episode_steps: int) -> int:
    """Pessimistic upper bound for all repeated real candidate replays."""
    groups = seeds * len(HORIZONS) * len(PROPOSALS) * candidates * 2 * (4 + max(HORIZONS))
    oracle = seeds * candidates * 2 * (episode_steps * (episode_steps - 1) // 2
                                       + 8 * episode_steps)
    return groups + oracle


def _action(values: Any) -> list[float]:
    a = np.asarray(values, dtype=np.float32).reshape(4)
    a[:2] = np.clip(a[:2], -1, 1)
    a[2:] = np.where(a[2:] >= 0, 1, -1)
    return [float(x) for x in a]


class PlannerBranchCollector:
    """Gather paired candidate and episode evidence without synthesizing returns.

    Injecting a fake environment is allowed only for tests: public CLI always
    demands an independently verified native ViZDoom backend and replay report.
    """

    def __init__(self, env_factory: Callable[[], Any], controller: Any, *,
                 seeds: list[int], world_checkpoint: str | Path,
                 actor_checkpoint: str | Path, replay_report: str | Path,
                 require_real_backend: bool = True, candidates: int = 16,
                 max_episode_steps: int = 525, gamma: float = .99):
        self.env_factory = env_factory
        self.controller = controller
        self.seeds = [int(seed) for seed in seeds]
        self.require_real_backend = bool(require_real_backend)
        if len(self.seeds) < 6 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("at least six distinct fixed seeds are required")
        if candidates < 16 or candidates % 2:
            raise ValueError("P1P requires at least 16 candidates, split evenly for mixed proposals")
        if not 1 <= max_episode_steps <= 525:
            raise ValueError("max_episode_steps must be between 1 and the native 525-step timeout")
        if not 0 < gamma <= 1:
            raise ValueError("gamma must be in (0, 1]")
        self.candidates = int(candidates)
        self.max_episode_steps = int(max_episode_steps)
        self.gamma = float(gamma)
        self.device = controller.device
        self.checkpoints = {"world": file_sha256(world_checkpoint), "actor": file_sha256(actor_checkpoint)}
        self.replay_path = Path(replay_report)
        replay = json.loads(self.replay_path.read_text(encoding="utf-8"))
        if self.require_real_backend and not validate_vizdoom_reset_replay_report(replay, self.seeds):
            raise ValueError("P1P requires a passing real-host replay report on the same seeds")
        self.replay_hash = file_sha256(self.replay_path)

    def _check_backend(self, env: Any) -> None:
        if self.require_real_backend:
            from awa.v2.game.vizdoom_qualification import assert_real_backend
            assert_real_backend(env)
        if getattr(getattr(env, "config", None), "scenario", None) is not None:
            from awa.v2.game.vizdoom_env import ViZDoomScenario
            if env.config.scenario != ViZDoomScenario.MY_WAY_HOME or env.config.track != "structured":
                raise ValueError("P1P requires structured my_way_home on every branch")

    def _rng(self, seed: int, horizon: int, proposal: str, step: int) -> np.random.Generator:
        digest = hashlib.sha256(f"P1P-v2:{seed}:{horizon}:{proposal}:{step}".encode()).digest()
        return np.random.default_rng(int.from_bytes(digest[:8], "big"))

    @torch.no_grad()
    def _belief_at(self, env: Any, seed: int, prefix: list[list[float]]) -> tuple[Any, torch.Tensor]:
        observation, _ = env.reset(seed=seed)
        state = self.controller.encoder.initial(1, self.device)
        previous = torch.zeros(1, 4, device=self.device)
        goal = torch.as_tensor(env.goal_vector(), dtype=torch.float32, device=self.device).reshape(1, -1)
        for index in range(len(prefix) + 1):
            obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
            state, belief = self.controller.encoder.observe(state, obs, previous, goal, index)
            if index < len(prefix):
                previous = torch.tensor([prefix[index]], dtype=torch.float32, device=self.device)
                observation, _, terminated, truncated, _ = env.step(prefix[index])
                if terminated or truncated:
                    raise RuntimeError("fixed prefix ended before candidate start")
        return observation, belief

    @torch.no_grad()
    def _population(self, belief: torch.Tensor, horizon: int, proposal: str,
                    rng: np.random.Generator) -> tuple[list[list[list[float]]], list[str]]:
        b = belief
        prior = []
        for _ in range(horizon):
            action = _action(self.controller.actor.actor.deterministic_action(b)[0].cpu().numpy())
            prior.append(action)
            b = self.controller.world.imagine_step(
                b, torch.tensor([action], dtype=torch.float32, device=self.device), deterministic=True)["belief"]
        actions = []
        origins = []
        for i in range(self.candidates):
            random_origin = proposal == "mixed" and i >= self.candidates // 2
            if random_origin:
                seq = [[float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), -1., -1.]
                       for _ in range(horizon)]
            else:
                seq = [[float(np.clip(a[0] + rng.normal(0, .35), -1, 1)),
                        float(np.clip(a[1] + rng.normal(0, .35), -1, 1)), a[2], a[3]] for a in prior]
                if i == 0:
                    seq = [a[:] for a in prior]
            actions.append(seq)
            origins.append("random" if random_origin else "actor")
        return actions, origins

    @torch.no_grad()
    def _model_scores(self, belief: torch.Tensor, actions: list[list[list[float]]],
                      arm: str) -> list[float]:
        count, horizon = len(actions), len(actions[0])
        seq = torch.tensor(actions, dtype=torch.float32, device=self.device)
        b = belief.expand(count, -1).clone()
        total = torch.zeros(count, device=self.device)
        alive = torch.ones(count, device=self.device)
        for t in range(horizon):
            out = self.controller.world.imagine_step(b, seq[:, t], deterministic=True)
            reward = out["reward"].reshape(-1)
            risk = out["risk"].reshape(-1) if arm == "P1P-D" else 0.
            total += alive * (reward - risk) * (self.gamma ** t)
            alive *= out["continuation"].reshape(-1)
            b = out["belief"]
        if arm in ("P1P-C", "P1P-D"):
            total += alive * self.controller.world.terminal_value(b).reshape(-1) * (self.gamma ** horizon)
        scores = [float(x) for x in total.cpu().tolist()]
        if not all(math.isfinite(x) for x in scores):
            raise RuntimeError("non-finite learned candidate scores")
        return scores

    def _real_scores(self, seed: int, prefix: list[list[float]],
                     candidates: list[list[list[float]]], expected_state: str,
                     replay_envs: tuple[Any, Any] | None = None) -> tuple[list[float], list[dict[str, Any]]]:
        scores = []
        proofs = []
        owned = replay_envs is None
        if owned:
            replay_envs = (self.env_factory(), self.env_factory())
        try:
            for env in replay_envs:
                self._check_backend(env)
            for actions in candidates:
                trials = [trace_on_env(env, seed, prefix, actions, self.require_real_backend)
                          for env in replay_envs]
                if trials[0] != trials[1] or trials[0].get("state_sha256") != expected_state:
                    raise RuntimeError("candidate branch replay differs from the measured starting state")
                if "error" in trials[0]:
                    raise RuntimeError(f"candidate branch failed: {trials[0]['error']}")
                scores.append(sum(self.gamma ** i * step["reward"]
                                  for i, step in enumerate(trials[0]["transitions"])))
                proofs.append({"state_sha256": expected_state,
                               "reward_trace": [step["reward"] for step in trials[0]["transitions"]],
                               "trace_sha256_by_replay": [hashlib.sha256(json.dumps(
                                   trial, sort_keys=True, allow_nan=False).encode()).hexdigest()
                                   for trial in trials]})
        finally:
            if owned:
                for env in replay_envs:
                    env.close()
        return scores, proofs

    def _episode(self, seed: int, horizon: int, proposal: str, arm: str) -> dict[str, Any]:
        env = self.env_factory()
        replay_envs = (self.env_factory(), self.env_factory()) if arm == "P1P-A" else None
        try:
            self._check_backend(env)
            observation, _ = env.reset(seed=seed)
            temporal = self.controller.encoder.initial(1, self.device)
            previous = torch.zeros(1, 4, device=self.device)
            goal = torch.as_tensor(env.goal_vector(), dtype=torch.float32, device=self.device).reshape(1, -1)
            prefix: list[list[float]] = []
            realized = 0.
            success = False
            for step in range(self.max_episode_steps):
                with torch.no_grad():
                    obs = torch.as_tensor(observation, dtype=torch.float32, device=self.device).reshape(1, -1)
                    temporal, belief = self.controller.encoder.observe(temporal, obs, previous, goal, step)
                    actions, _ = self._population(belief, horizon, proposal,
                                                   self._rng(seed, horizon, proposal, step))
                    if arm == "P1P-A":
                        scores, _ = self._real_scores(seed, prefix, actions, _observation_sha256(observation),
                                                      replay_envs=replay_envs)
                    else:
                        scores = self._model_scores(belief, actions, arm)
                selected = actions[int(np.argmax(scores))][0]
                previous = torch.tensor([selected], dtype=torch.float32, device=self.device)
                observation, reward, terminated, truncated, info = env.step(selected)
                prefix.append(selected)
                realized += float(reward)
                success = bool(info.get("success", False))
                if terminated or truncated:
                    break
            return {"seed": seed, "horizon": horizon, "proposal": proposal,
                    "success": success, "realized_return": realized, "steps": len(prefix),
                    "action_trace_sha256": hashlib.sha256(json.dumps(prefix).encode()).hexdigest()}
        finally:
            if replay_envs is not None:
                for replay_env in replay_envs:
                    replay_env.close()
            env.close()

    def _random_episode(self, seed: int) -> dict[str, Any]:
        env = self.env_factory()
        try:
            self._check_backend(env)
            env.reset(seed=seed)
            rng = self._rng(seed, 0, "random", 0)
            ret = 0.
            success = False
            for _ in range(self.max_episode_steps):
                action = [float(rng.uniform(-1, 1)), float(rng.uniform(-1, 1)), -1., -1.]
                _, reward, terminated, truncated, info = env.step(action)
                ret += float(reward)
                success = bool(info.get("success", False))
                if terminated or truncated:
                    break
            return {"seed": seed, "success": success, "realized_return": ret}
        finally:
            env.close()

    def _group(self, seed: int, horizon: int, proposal: str) -> dict[str, dict[str, Any]]:
        # A real, nonterminal prefix gives every arm the same held-out state.
        prefix = [[0., 1., -1., -1.]] * 4
        env = self.env_factory()
        try:
            self._check_backend(env)
            observation, belief = self._belief_at(env, seed, prefix)
            state_id = _observation_sha256(observation)
            population, origins = self._population(belief, horizon, proposal,
                                                    self._rng(seed, horizon, proposal, 0))
            now = time.perf_counter()
            realized, proofs = self._real_scores(seed, prefix, population, state_id)
            oracle_latency = (time.perf_counter() - now) * 1000.
            result = {}
            for arm in TESTS:
                now = time.perf_counter()
                scores = realized if arm == "P1P-A" else self._model_scores(belief, population, arm)
                latency = oracle_latency if arm == "P1P-A" else (time.perf_counter() - now) * 1000.
                result[arm] = {"seed": seed, "horizon": horizon, "proposal": proposal,
                               "state_id": state_id, "prefix_actions": prefix,
                               "predicted_scores": scores, "realized_returns": realized,
                               "action_sequences": population, "origins": origins,
                               "chosen_first_action": population[int(np.argmax(scores))][0],
                               "planning_latency_ms": latency}
                if arm == "P1P-A":
                    result[arm]["branch_proofs"] = proofs
            return result
        finally:
            env.close()

    def collect(self, output: str | Path) -> dict[str, Any]:
        """Resume via .partial; publish raw evidence only after every real run."""
        target = Path(output)
        partial = target.with_name(target.name + ".partial")
        empty = {"format": FORMAT_RAW, "scenario": "my_way_home", "seed_ids": self.seeds,
                 "horizons": list(HORIZONS), "checkpoint_sha256": self.checkpoints,
                 "branch_replay_report_sha256": self.replay_hash,
                 "collector": {"real_backend": self.require_real_backend, "candidate_replays": 2,
                               "max_episode_steps": self.max_episode_steps, "gamma": self.gamma,
                               "candidates": self.candidates},
                 "tests": [{"id": arm, **contract, "candidate_groups": [], "episode_results": []}
                           for arm, contract in TESTS.items()], "random_baseline": []}
        if target.exists() and not partial.exists():
            finished = json.loads(target.read_text(encoding="utf-8"))
            if any(finished.get(field) != empty[field] for field in
                   ("format", "scenario", "seed_ids", "horizons", "checkpoint_sha256",
                    "branch_replay_report_sha256", "collector")):
                raise ValueError("existing completed P1P receipt does not match frozen run")
            build_planner_diagnostic_report(finished)
            return finished
        raw = json.loads(partial.read_text()) if partial.exists() else empty
        for field in ("format", "scenario", "seed_ids", "horizons", "checkpoint_sha256",
                      "branch_replay_report_sha256", "collector"):
            if raw.get(field) != empty[field]:
                raise ValueError(f"partial P1P receipt does not match frozen {field}")
        if [row.get("id") for row in raw.get("tests", [])] != list(TESTS):
            raise ValueError("partial P1P test ladder is malformed")
        for seed in self.seeds:
            if seed not in [row["seed"] for row in raw["random_baseline"]]:
                raw["random_baseline"].append(self._random_episode(seed))
                _write_atomic(partial, raw)
            for horizon in HORIZONS:
                for proposal in PROPOSALS:
                    key = _key(seed, horizon, proposal)
                    groups = [{_key(g["seed"], g["horizon"], g["proposal"])
                               for g in row["candidate_groups"]} for row in raw["tests"]]
                    if any(key not in seen for seen in groups):
                        if any(key in seen for seen in groups):
                            raise ValueError("partial receipt contains an incomplete paired candidate group")
                        matched = self._group(seed, horizon, proposal)
                        for row in raw["tests"]:
                            row["candidate_groups"].append(matched[row["id"]])
                        _write_atomic(partial, raw)
                    for row in raw["tests"]:
                        if row["id"] == "P1P-A" and (horizon, proposal) != (8, "mixed"):
                            continue
                        if key not in {_key(e["seed"], e["horizon"], e["proposal"])
                                       for e in row["episode_results"]}:
                            outcome = self._episode(seed, horizon, proposal, row["id"])
                            row["episode_results"].append(outcome)
                            _write_atomic(partial, raw)
        oracle_rows = raw["tests"][0]["episode_results"]
        raw["tests"][0]["oracle_search_episode_results"] = [
            {"seed": seed, "success": next(row["success"] for row in oracle_rows
                                             if _key(row["seed"], row["horizon"], row["proposal"]) ==
                                             _key(seed, 8, "mixed")),
             "realized_return": next(row["realized_return"] for row in oracle_rows
                                     if _key(row["seed"], row["horizon"], row["proposal"]) ==
                                     _key(seed, 8, "mixed"))} for seed in self.seeds]
        # A BLOCKED behavioral result is legitimate evidence; only incomplete
        # or inconsistent collection fails to publish the raw measurement.
        build_planner_diagnostic_report(raw)
        _write_atomic(target, raw)
        partial.unlink(missing_ok=True)
        return raw
