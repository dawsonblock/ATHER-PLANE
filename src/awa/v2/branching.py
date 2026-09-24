from __future__ import annotations

from dataclasses import dataclass, asdict
import copy
import time
import numpy as np
import torch


class ExactSnapshotError(RuntimeError):
    pass


def _reset_env(env, *, seed=None):
    """Normalize Gymnasium/Gym reset signatures to an observation."""
    if seed is None:
        out = env.reset()
    else:
        try:
            out = env.reset(seed=seed)
        except TypeError:
            out = env.reset()
    return out[0] if isinstance(out, tuple) and len(out) == 2 else out


def _step_env(env, action):
    """Normalize Gymnasium 5-tuples and legacy Gym 4-tuples."""
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


def capture_snapshot(env):
    if not hasattr(env, "state_dict") or not hasattr(env, "load_state_dict"):
        raise TypeError("exact branching requires state_dict/load_state_dict support")
    return copy.deepcopy(env.state_dict())


def restore_snapshot(env, snapshot):
    env.load_state_dict(copy.deepcopy(snapshot))


def verify_snapshot_roundtrip(env, action, atol=1e-7):
    """Prove a branch can be replayed identically from an exact environment state."""
    snap = capture_snapshot(env)
    o1, r1, d1, _ = _step_env(env, action)
    restore_snapshot(env, snap)
    o2, r2, d2, _ = _step_env(env, action)
    restore_snapshot(env, snap)
    ok = (
        np.allclose(np.asarray(o1), np.asarray(o2), rtol=1e-6, atol=atol)
        and abs(float(r1) - float(r2)) <= atol
        and bool(d1) == bool(d2)
    )
    if not ok:
        raise ExactSnapshotError("environment failed exact snapshot branch replay")
    return True


@dataclass
class BranchResult:
    name: str
    budget: int
    discounted_return: float
    steps: int
    world_model_calls: int
    latency_ms: float
    terminal: bool

    def to_dict(self):
        return asdict(self)


@torch.no_grad()
def rollout_real_branch(
    env,
    start_snapshot,
    encode_observation,
    actor,
    planner=None,
    budget: int = 0,
    horizon: int = 3,
    discount: float = 0.99,
    start_observation=None,
):
    """Roll out a counterfactual branch from an exact environment snapshot.

    ``start_observation`` makes the brancher compatible with external Gymnasium-style
    environments whose state snapshot does not expose an ``observation`` field or an
    internal ``_obs()`` helper.
    """
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    if not np.isfinite(discount) or discount < 0:
        raise ValueError("discount must be finite and >= 0")
    restore_snapshot(env, start_snapshot)
    obs = start_observation
    if obs is None:
        obs_fn = getattr(env, "_obs", None)
        obs = obs_fn() if callable(obs_fn) else None
    if obs is None:
        state = env.state_dict()
        obs = state.get("observation") if isinstance(state, dict) else None
    if obs is None:
        raise TypeError(
            "branch rollout needs start_observation, env._obs(), or snapshot['observation']"
        )

    total = 0.0
    disc = 1.0
    calls = 0
    done = False
    steps = 0
    t0 = time.perf_counter()
    try:
        for _ in range(int(horizon)):
            belief = encode_observation(obs)
            if belief.ndim == 1:
                belief = belief.unsqueeze(0)
            if planner is None:
                action = actor.deterministic_action(belief)
                if isinstance(action, tuple):
                    action = action[0]
            else:
                # Gradient planners explicitly re-enable their own optimization scope.
                result = planner.plan(belief, actor=actor, budget=budget)
                action = result.action
                calls += int(result.world_model_calls)
            env_action = action.squeeze(0).detach().cpu().numpy()
            obs, reward, done, _ = _step_env(env, env_action)
            total += disc * float(reward)
            steps += 1
            if done:
                break
            disc *= float(discount)
    finally:
        restore_snapshot(env, start_snapshot)
    latency = (time.perf_counter() - t0) * 1000.0
    return BranchResult(
        "actor" if planner is None else planner.name,
        int(budget),
        float(total),
        steps,
        calls,
        latency,
        bool(done),
    )


@dataclass
class BenefitRecord:
    features: list[float]
    actor_return: float
    planner: str
    budget: int
    planner_return: float
    gain: float
    normalized_cost: float
    world_model_calls: int
    latency_ms: float
    episode: int
    step: int

    def to_dict(self):
        return asdict(self)


class PlannerBenefitTable:
    def __init__(self, records, choices):
        self.records = list(records)
        self.choices = list(choices)

    def __len__(self):
        return len(self.records)

    def matrix(self):
        by_state = {}
        for r in self.records:
            by_state.setdefault((r.episode, r.step), []).append(r)
        xs, gains, costs, keys = [], [], [], []
        for key, rows in sorted(by_state.items()):
            m = {(r.planner, r.budget): r for r in rows}
            if not all(c in m for c in self.choices):
                continue
            xs.append(rows[0].features)
            gains.append([m[c].gain for c in self.choices])
            costs.append([m[c].normalized_cost for c in self.choices])
            keys.append(key)
        if not xs:
            raise ValueError("no complete planner-benefit rows")
        return (
            torch.tensor(xs, dtype=torch.float32),
            torch.tensor(gains, dtype=torch.float32),
            torch.tensor(costs, dtype=torch.float32),
            keys,
        )

    def summary(self):
        out = {}
        for c in self.choices:
            rows = [r for r in self.records if (r.planner, r.budget) == c]
            if not rows:
                continue
            out[f"{c[0]}:{c[1]}"] = {
                "samples": len(rows),
                "mean_gain": float(np.mean([r.gain for r in rows])),
                "win_rate": float(np.mean([r.gain > 0 for r in rows])),
                "mean_world_model_calls": float(np.mean([r.world_model_calls for r in rows])),
                "mean_latency_ms": float(np.mean([r.latency_ms for r in rows])),
                "gain_per_1k_calls": float(
                    1000
                    * np.sum([r.gain for r in rows])
                    / max(1, np.sum([r.world_model_calls for r in rows]))
                ),
            }
        return out


def default_voc_features(world, belief, actor_action, horizons=(1, 5, 20)):
    vals = []
    for h in horizons:
        vals.append(float(world.uncertainty.at_horizon(belief, int(h)).mean().detach()))
    vals.append(float(world.risk.aggregate_risk(belief, actor_action).mean().detach()))
    vals.append(float(actor_action.norm(dim=-1).mean().detach()))
    vals.append(float(belief.norm(dim=-1).mean().detach()))
    return vals


def _action_probe(env):
    action_dim = getattr(env, "action_dim", None)
    if action_dim is not None:
        return np.zeros(int(action_dim), dtype=np.float32)
    space = getattr(env, "action_space", None)
    shape = getattr(space, "shape", None)
    if shape is not None and int(np.prod(shape)) > 0:
        return np.zeros(shape, dtype=np.float32)
    raise TypeError("cannot infer action shape for snapshot qualification")


@torch.no_grad()
def collect_planner_benefits(
    env_factory,
    encode_observation,
    world,
    actor,
    planner_choices,
    episodes: int = 4,
    max_states: int = 128,
    branch_horizon: int = 3,
    discount: float = 0.99,
    cost_per_1k_calls: float = 0.01,
    seed: int = 0,
):
    """Measure every planner/budget from identical real environment snapshots."""
    records = []
    states = 0
    for ep in range(int(episodes)):
        env = env_factory(seed + ep)
        obs = _reset_env(env, seed=seed + ep)
        done = False
        step = 0
        try:
            verify_snapshot_roundtrip(env, _action_probe(env))
            while not done and states < int(max_states):
                belief = encode_observation(obs)
                if belief.ndim == 1:
                    belief = belief.unsqueeze(0)
                aa = actor.deterministic_action(belief)
                if isinstance(aa, tuple):
                    aa = aa[0]
                features = default_voc_features(world, belief, aa)
                snap = capture_snapshot(env)
                actor_result = rollout_real_branch(
                    env,
                    snap,
                    encode_observation,
                    actor,
                    None,
                    0,
                    branch_horizon,
                    discount,
                    start_observation=obs,
                )
                for planner, budget in planner_choices:
                    pr = rollout_real_branch(
                        env,
                        snap,
                        encode_observation,
                        actor,
                        planner,
                        budget,
                        branch_horizon,
                        discount,
                        start_observation=obs,
                    )
                    cost = float(cost_per_1k_calls) * (pr.world_model_calls / 1000.0)
                    records.append(
                        BenefitRecord(
                            features,
                            actor_result.discounted_return,
                            planner.name,
                            int(budget),
                            pr.discounted_return,
                            pr.discounted_return - actor_result.discounted_return,
                            cost,
                            pr.world_model_calls,
                            pr.latency_ms,
                            ep,
                            step,
                        )
                    )
                restore_snapshot(env, snap)
                obs, _, done, _ = _step_env(
                    env, aa.squeeze(0).detach().cpu().numpy()
                )
                states += 1
                step += 1
        finally:
            close = getattr(env, "close", None)
            if close is not None:
                close()
        if states >= int(max_states):
            break
    choices = [(p.name, int(b)) for p, b in planner_choices]
    return PlannerBenefitTable(records, choices)
