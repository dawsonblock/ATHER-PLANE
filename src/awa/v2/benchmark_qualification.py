from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Any
import json
import time

import numpy as np
import torch

from awa.v2.arena import backend_budget_for_calls


@dataclass
class ClosedLoopEpisode:
    controller: str
    seed: int
    episode_return: float
    steps: int
    success: bool
    world_model_calls: int
    planning_latency_ms: float
    search_decisions: int
    total_decisions: int

    def to_dict(self): return asdict(self)


@dataclass
class ControllerScore:
    controller: str
    episodes: int
    mean_return: float
    std_return: float
    success_rate: float
    mean_steps: float
    mean_planning_latency_ms_per_decision: float
    world_model_calls_per_decision: float
    search_rate: float
    return_per_1k_world_calls: float | None
    paired_return_gain_vs_actor: float | None = None
    paired_win_rate_vs_actor: float | None = None

    def to_dict(self): return asdict(self)


@dataclass
class ClosedLoopQualificationReport:
    seeds: list[int]
    controllers: dict[str, ControllerScore]
    paired: dict[str, dict[str, float]]
    environment: dict[str, Any]

    def to_dict(self):
        return {
            "seeds": list(self.seeds),
            "controllers": {k: v.to_dict() for k, v in self.controllers.items()},
            "paired": self.paired,
            "environment": dict(self.environment),
        }


def _success_from_info(info: dict, done: bool) -> bool:
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    return bool(done and isinstance(info, dict) and info.get("is_success", False))


@torch.no_grad()
def run_closed_loop_episode(
    env,
    encode_observation: Callable[[Any], torch.Tensor],
    actor,
    *,
    controller: str = "actor",
    planner=None,
    requested_world_model_calls: int = 64,
    max_steps: int | None = None,
) -> ClosedLoopEpisode:
    obs = env.reset()
    done = False
    total = 0.0
    steps = 0
    calls = 0
    latency = 0.0
    searches = 0
    success = False
    cap = int(max_steps or getattr(env, "horizon", 1000))
    seed = int(getattr(env, "_benchmark_seed", -1))
    while not done and steps < cap:
        belief = encode_observation(obs)
        if belief.ndim == 1:
            belief = belief.unsqueeze(0)
        if controller == "actor" or planner is None:
            action = actor.deterministic_action(belief)
            if isinstance(action, tuple): action = action[0]
        else:
            internal_budget = backend_budget_for_calls(planner, int(requested_world_model_calls))
            t0 = time.perf_counter()
            result = planner.plan(belief, actor=actor, budget=internal_budget)
            latency += (time.perf_counter() - t0) * 1000.0
            action = result.action
            calls += int(result.world_model_calls)
            searches += 1
        obs, reward, done, info = env.step(action.squeeze(0).detach().cpu().numpy())
        total += float(reward)
        steps += 1
        success = success or _success_from_info(info, done)
    return ClosedLoopEpisode(
        controller=str(controller), seed=seed, episode_return=float(total), steps=int(steps),
        success=bool(success), world_model_calls=int(calls), planning_latency_ms=float(latency),
        search_decisions=int(searches), total_decisions=int(steps),
    )


def _aggregate(name: str, rows: list[ClosedLoopEpisode]) -> ControllerScore:
    returns = np.asarray([r.episode_return for r in rows], dtype=np.float64)
    decisions = max(1, sum(r.total_decisions for r in rows))
    calls = sum(r.world_model_calls for r in rows)
    eff = None if calls <= 0 else float(1000.0 * returns.sum() / calls)
    return ControllerScore(
        controller=name,
        episodes=len(rows),
        mean_return=float(returns.mean()) if len(returns) else float("nan"),
        std_return=float(returns.std(ddof=0)) if len(returns) else float("nan"),
        success_rate=float(np.mean([r.success for r in rows])) if rows else float("nan"),
        mean_steps=float(np.mean([r.steps for r in rows])) if rows else float("nan"),
        mean_planning_latency_ms_per_decision=float(sum(r.planning_latency_ms for r in rows) / decisions),
        world_model_calls_per_decision=float(calls / decisions),
        search_rate=float(sum(r.search_decisions for r in rows) / decisions),
        return_per_1k_world_calls=eff,
    )


def qualify_closed_loop_controllers(
    env_factory: Callable[[int], Any],
    encode_observation: Callable[[Any], torch.Tensor],
    actor,
    planner_specs: list[tuple[str, Any, int]],
    *,
    seeds: list[int] | tuple[int, ...] = (0, 1, 2, 3, 4),
    max_steps: int | None = None,
    environment_metadata: dict[str, Any] | None = None,
) -> tuple[ClosedLoopQualificationReport, list[ClosedLoopEpisode]]:
    """Compare actor and planners on identical seed sets in the real environment.

    Every controller receives a newly constructed environment with the same seed.
    This is a closed-loop benchmark rather than counterfactual one-state branching.
    Planner budgets are requested in approximate world-model transition calls.
    """
    seeds = [int(s) for s in seeds]
    all_rows: list[ClosedLoopEpisode] = []
    by_name: dict[str, list[ClosedLoopEpisode]] = {"actor": []}
    for name, _, _ in planner_specs:
        by_name.setdefault(str(name), [])

    controllers = [("actor", None, 0)] + [(str(n), p, int(b)) for n, p, b in planner_specs]
    for name, planner, calls in controllers:
        for seed in seeds:
            env = env_factory(seed)
            setattr(env, "_benchmark_seed", seed)
            try:
                row = run_closed_loop_episode(
                    env, encode_observation, actor, controller=name, planner=planner,
                    requested_world_model_calls=calls, max_steps=max_steps,
                )
            finally:
                close = getattr(env, "close", None)
                if close is not None: close()
            by_name[name].append(row); all_rows.append(row)

    summaries = {name: _aggregate(name, rows) for name, rows in by_name.items()}
    actor_map = {r.seed: r for r in by_name["actor"]}
    paired: dict[str, dict[str, float]] = {}
    for name, rows in by_name.items():
        if name == "actor": continue
        gains = [r.episode_return - actor_map[r.seed].episode_return for r in rows if r.seed in actor_map]
        paired[name] = {
            "mean_return_gain": float(np.mean(gains)) if gains else float("nan"),
            "median_return_gain": float(np.median(gains)) if gains else float("nan"),
            "win_rate": float(np.mean([g > 0 for g in gains])) if gains else float("nan"),
            "tie_rate": float(np.mean([abs(g) <= 1e-9 for g in gains])) if gains else float("nan"),
        }
        summaries[name].paired_return_gain_vs_actor = paired[name]["mean_return_gain"]
        summaries[name].paired_win_rate_vs_actor = paired[name]["win_rate"]
    report = ClosedLoopQualificationReport(seeds, summaries, paired, environment_metadata or {})
    return report, all_rows


def bootstrap_paired_return_ci(
    actor_returns: list[float], candidate_returns: list[float], *, samples: int = 2000, seed: int = 0,
) -> tuple[float, float, float]:
    a = np.asarray(actor_returns, dtype=np.float64)
    c = np.asarray(candidate_returns, dtype=np.float64)
    if a.shape != c.shape or a.ndim != 1 or len(a) == 0:
        raise ValueError("paired return arrays must be non-empty and equal length")
    diff = c - a
    rng = np.random.default_rng(seed)
    means = np.empty(int(samples), dtype=np.float64)
    for i in range(int(samples)):
        idx = rng.integers(0, len(diff), size=len(diff))
        means[i] = diff[idx].mean()
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(diff.mean()), float(lo), float(hi)


def write_closed_loop_report(report: ClosedLoopQualificationReport, rows: list[ClosedLoopEpisode], path: str | Path) -> Path:
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["episodes"] = [r.to_dict() for r in rows]
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target
