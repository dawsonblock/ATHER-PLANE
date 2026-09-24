"""Check that fixed ViZDoom states can be reproduced by reset and action replay.

Native ViZDoom save/load does not restore all reward/time counters. This probe
uses fresh episodes and compares observations and outcomes at each step; it is
an engineering prerequisite, not the P1P oracle-ladder measurement itself.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Callable

import numpy as np


DEFAULT_PROBES: tuple[dict[str, Any], ...] = (
    {"id": "initial", "prefix": [], "candidates": [
        [[0.0, 1.0, -1.0, -1.0]] * 8,
        [[0.8, 1.0, -1.0, -1.0]] * 8,
    ]},
    {"id": "after_four_steps", "prefix": [[0.0, 1.0, -1.0, -1.0]] * 4,
     "candidates": [
         [[-0.8, 1.0, -1.0, -1.0]] * 8,
         [[0.8, 1.0, -1.0, -1.0]] * 8,
     ]},
)


def _observation_sha256(observation: Any) -> str:
    array = np.ascontiguousarray(np.asarray(observation))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(json.dumps(array.shape).encode("utf-8"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _trace(env_factory: Callable[[], Any], seed: int, prefix: list[list[float]],
           actions: list[list[float]], require_real_backend: bool) -> dict[str, Any]:
    env = env_factory()
    try:
        observation, _ = env.reset(seed=seed)
        if require_real_backend:
            from awa.v2.game.vizdoom_qualification import assert_real_backend
            assert_real_backend(env)
        for action in prefix:
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                return {"error": "episode_ended_before_fixed_start"}
        state_sha = _observation_sha256(observation)
        transitions = []
        for action in actions:
            observation, reward, terminated, truncated, info = env.step(action)
            transitions.append({
                "observation_sha256": _observation_sha256(observation),
                "reward": float(reward), "terminated": bool(terminated),
                "truncated": bool(truncated), "success": bool(info.get("success", False)),
            })
            if terminated or truncated:
                break
        return {"state_sha256": state_sha, "transitions": transitions}
    finally:
        env.close()


def qualify_vizdoom_reset_replay(
    env_factory: Callable[[], Any], *, seeds: list[int],
    probes: list[dict[str, Any]] | None = None, repeats: int = 2,
    require_real_backend: bool = False,
) -> dict[str, Any]:
    """Require identical branch starts and complete traces for every probed seed.

    A PASS applies only to the listed prefixes, candidate actions, and seeds.
    The eventual P1P collector must repeat this check for its own fixed states.
    """
    if not seeds or len(set(seeds)) != len(seeds) or repeats < 2:
        raise ValueError("replay qualification needs distinct seeds and at least two repeats")
    chosen = list(probes if probes is not None else DEFAULT_PROBES)
    if not chosen or len({str(p.get("id")) for p in chosen}) != len(chosen):
        raise ValueError("replay probes need distinct IDs")
    rows = []
    for seed in seeds:
        for probe in chosen:
            prefix = probe.get("prefix") or []
            candidates = probe.get("candidates") or []
            if len(candidates) < 2 or any(not sequence for sequence in candidates):
                raise ValueError("each replay probe needs two nonempty candidate sequences")
            baseline_state = None
            for index, actions in enumerate(candidates):
                trials = [_trace(env_factory, seed, prefix, actions, require_real_backend)
                          for _ in range(repeats)]
                trial_hashes = [hashlib.sha256(json.dumps(
                    trial, sort_keys=True, allow_nan=False).encode()).hexdigest() for trial in trials]
                valid = (all("error" not in row for row in trials)
                         and all(row == trials[0] for row in trials[1:])
                         and (baseline_state is None or trials[0]["state_sha256"] == baseline_state))
                if baseline_state is None and "state_sha256" in trials[0]:
                    baseline_state = trials[0]["state_sha256"]
                rows.append({"seed": int(seed), "probe": str(probe["id"]),
                             "candidate": index, "replays": repeats,
                             "identical": bool(valid),
                             "state_sha256_by_replay": [trial.get("state_sha256") for trial in trials],
                             "trace_sha256_by_replay": trial_hashes,
                             "errors": [trial["error"] for trial in trials if "error" in trial]})
    qualified = all(row["identical"] for row in rows)
    return {"format": "awa-v2.38.6-vizdoom-reset-replay-v1",
            "status": "PASS" if qualified else "BLOCKED", "qualified": qualified,
            "real_backend_checked": bool(require_real_backend),
            "seeds": list(seeds), "probes": chosen, "results": rows,
            "claim_boundary": "This checks reproducibility only for the measured reset/action prefixes; "
                              "it does not establish an independent reward oracle or planner benefit."}


def validate_vizdoom_reset_replay_report(report: dict[str, Any], seeds: list[int]) -> bool:
    """Check complete repeated traces and starts before binding a P1P report."""
    if (report.get("format") != "awa-v2.38.6-vizdoom-reset-replay-v1"
        or report.get("status") != "PASS" or report.get("qualified") is not True
        or report.get("real_backend_checked") is not True
        or report.get("seeds") != seeds or len(seeds) < 6
        or any(not isinstance(seed, int) or isinstance(seed, bool) for seed in seeds)
        or len(set(seeds)) != len(seeds)):
        return False
    probes = report.get("probes")
    rows = report.get("results")
    if not isinstance(probes, list) or len(probes) < 2 or not isinstance(rows, list):
        return False
    expected = set()
    for probe in probes:
        if not isinstance(probe, dict) or not isinstance(probe.get("id"), str):
            return False
        candidates = probe.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            return False
        expected.update((seed, probe["id"], i) for seed in seeds for i in range(len(candidates)))
    if len(expected) != sum(len(p["candidates"]) for p in probes) * len(seeds):
        return False
    seen = set()
    starting_states = {}
    for row in rows:
        if not isinstance(row, dict):
            return False
        if (not isinstance(row.get("seed"), int) or not isinstance(row.get("probe"), str)
            or not isinstance(row.get("candidate"), int)):
            return False
        key = (row.get("seed"), row.get("probe"), row.get("candidate"))
        states = row.get("state_sha256_by_replay")
        traces = row.get("trace_sha256_by_replay")
        repeats = row.get("replays")
        if (key not in expected or key in seen or row.get("identical") is not True
            or not isinstance(repeats, int) or repeats < 2
            or not isinstance(states, list) or len(states) != repeats
            or not isinstance(traces, list) or len(traces) != repeats
            or any(not isinstance(x, str) or len(x) != 64 for x in states + traces)
            or len(set(states)) != 1 or len(set(traces)) != 1 or row.get("errors")):
            return False
        group = (row["seed"], row["probe"])
        if group in starting_states and starting_states[group] != states[0]:
            return False
        starting_states[group] = states[0]
        seen.add(key)
    return seen == expected
