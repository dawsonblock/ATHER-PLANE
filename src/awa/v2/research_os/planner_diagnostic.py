"""Evidence gate for the v2.38.6 ViZDoom planner branch comparison.

The executor does not synthesize planner results. This module validates a raw
receipt produced by the diagnostic run and computes paired ranking, horizon,
objective, and proposal-population summaries from its candidate-level records.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any


FORMAT_RAW = "awa-v2.38.6-planner-diagnostic-raw-v2"
FORMAT_REPORT = "awa-v2.38.6-planner-diagnostic-v2"
HORIZONS = (1, 2, 4, 8, 16, 32)
PROPOSALS = ("actor_seeded", "mixed")
TESTS = {
    "P1P-A": {"dynamics": "environment", "reward": "oracle", "terminal_value": "none", "risk": "none"},
    "P1P-B": {"dynamics": "learned", "reward": "learned", "terminal_value": "none", "risk": "none"},
    "P1P-C": {"dynamics": "learned", "reward": "learned", "terminal_value": "learned", "risk": "none"},
    "P1P-D": {"dynamics": "learned", "reward": "learned", "terminal_value": "learned", "risk": "learned"},
}


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in order[i:j]:
            ranks[k] = rank
        i = j
    return ranks


def _correlation(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    dx, dy = [v - mx for v in rx], [v - my for v in ry]
    den = math.sqrt(sum(v * v for v in dx) * sum(v * v for v in dy))
    return sum(a * b for a, b in zip(dx, dy)) / den if den else 0.0


def _candidate_group_metrics(group: dict[str, Any], *, top_k: int = 5) -> dict[str, Any]:
    predicted = [float(x) for x in group["predicted_scores"]]
    actual = [float(x) for x in group["realized_returns"]]
    actions = group["action_sequences"]
    origins = [str(x) for x in group["origins"]]
    if not (len(predicted) == len(actual) == len(actions) == len(origins)):
        raise ValueError("candidate score, return, action, and origin arrays must have equal length")
    if len(predicted) < top_k or not all(math.isfinite(x) for x in predicted + actual):
        raise ValueError("candidate groups need finite scores and at least top_k candidates")
    chosen = group.get("chosen_first_action")
    if not isinstance(chosen, list) or len(chosen) != 4 or not all(math.isfinite(float(x)) for x in chosen):
        raise ValueError("each candidate group must log the chosen four-dimensional first action")
    latency = float(group["planning_latency_ms"])
    if not math.isfinite(latency) or latency < 0:
        raise ValueError("planning latency must be a finite non-negative measurement")
    if not all(isinstance(seq, list) and seq for seq in actions):
        raise ValueError("each candidate action sequence must be a non-empty list")
    if not set(origins).issubset({"actor", "random", "elite"}):
        raise ValueError("candidate origins must be labeled actor, random, or elite")
    if not group.get("state_id"):
        raise ValueError("candidate group must bind its fixed ViZDoom state snapshot ID")
    if str(group["proposal"]) == "mixed" and not {"actor", "random"}.issubset(set(origins)):
        raise ValueError("mixed candidate population must include both actor and random-origin proposals")
    if str(group["proposal"]) == "actor_seeded" and "actor" not in set(origins):
        raise ValueError("actor_seeded candidate population must identify its actor-origin proposals")
    for seq in actions:
        if len(seq) != int(group["horizon"]):
            raise ValueError("candidate action sequence length must equal its declared horizon")
        if any(not isinstance(action, list) or len(action) != 4
               or not all(math.isfinite(float(x)) and -1.0001 <= float(x) <= 1.0001 for x in action)
               for action in seq):
            raise ValueError("ViZDoom candidate actions must use the four-dimensional action ABI")
    n = len(predicted)
    pred_top = set(sorted(range(n), key=predicted.__getitem__, reverse=True)[:top_k])
    actual_top = set(sorted(range(n), key=actual.__getitem__, reverse=True)[:top_k])
    origin_counts = Counter(origins)
    first_keys = []
    sequence_keys = set()
    for seq in actions:
        sequence_keys.add(json.dumps(seq, separators=(",", ":")))
        a = seq[0]
        first_keys.append((int(float(a[0]) > .25) - int(float(a[0]) < -.25),
                           int(float(a[1]) > .25) - int(float(a[1]) < -.25),
                           int(float(a[2]) > 0), int(float(a[3]) > 0)))
    action_counts = Counter(first_keys)
    action_entropy = -sum((c / n) * math.log(c / n) for c in action_counts.values() if c)
    max_entropy = math.log(max(1, min(n, 36)))
    return {
        "seed": int(group["seed"]), "horizon": int(group["horizon"]), "proposal": str(group["proposal"]),
        "candidate_count": n,
        "spearman_predicted_vs_realized": _correlation(predicted, actual),
        "top_k": top_k,
        "top_k_agreement": len(pred_top & actual_top) / float(top_k),
        "unique_action_sequences": len(sequence_keys),
        "unique_sequence_fraction": len(sequence_keys) / n,
        "first_action_entropy_nats": action_entropy,
        "normalized_first_action_entropy": action_entropy / max_entropy if max_entropy else 0.0,
        "origin_counts": dict(sorted(origin_counts.items())),
        "origin_fractions": {key: value / n for key, value in sorted(origin_counts.items())},
        "best_minus_median_predicted_score": max(predicted) - sorted(predicted)[n // 2],
        "chosen_first_action": group.get("chosen_first_action"),
        "planning_latency_ms": latency,
    }


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _paired_delta(base: list[dict[str, Any]], candidate: list[dict[str, Any]]) -> dict[str, Any]:
    def keyed(rows):
        return {(int(r["seed"]), int(r["horizon"]), str(r["proposal"])): r for r in rows}
    a, b = keyed(base), keyed(candidate)
    keys = sorted(set(a) & set(b))
    ret = [float(b[k]["realized_return"]) - float(a[k]["realized_return"]) for k in keys]
    suc = [float(bool(b[k]["success"])) - float(bool(a[k]["success"])) for k in keys]
    return {"paired_cells": len(keys), "mean_return_delta": _mean(ret), "mean_success_delta": _mean(suc),
            "return_delta_by_pair": ret, "success_delta_by_pair": suc}


def build_planner_diagnostic_report(raw: dict[str, Any], *, min_spearman: float = .30,
                                    min_top_k_agreement: float = .50,
                                    min_oracle_success_gain: float = .15,
                                    min_oracle_success_rate: float = .50,
                                    minimum_candidates: int = 16,
                                    minimum_qualified_horizon_for_p2: int = 8) -> dict[str, Any]:
    """Validate matched real branches and learned-model decisions."""
    if raw.get("format") != FORMAT_RAW:
        raise ValueError(f"raw planner evidence format must be {FORMAT_RAW}")
    if str(raw.get("scenario")) != "my_way_home":
        raise ValueError("P1P evidence must use my_way_home throughout")
    seeds = [int(x) for x in raw.get("seed_ids", [])]
    if len(seeds) < 6 or len(set(seeds)) != len(seeds):
        raise ValueError("P1P requires at least six distinct preregistered evaluation seeds")
    if tuple(int(x) for x in raw.get("horizons", [])) != HORIZONS:
        raise ValueError(f"P1P horizons must be exactly {list(HORIZONS)} in ascending order")
    checkpoint = raw.get("checkpoint_sha256") or {}
    if any(not isinstance(checkpoint.get(k), str) or len(checkpoint[k]) != 64 for k in ("world", "actor")):
        raise ValueError("raw receipt must bind the world and actor checkpoint SHA-256 hashes")
    replay_hash = raw.get("branch_replay_report_sha256")
    if not isinstance(replay_hash, str) or len(replay_hash) != 64:
        raise ValueError("raw receipt must bind the real-host reset-replay qualification SHA-256")
    collector = raw.get("collector") or {}
    if (collector.get("real_backend") is not True or collector.get("candidate_replays") != 2
        or not isinstance(collector.get("max_episode_steps"), int)
        or not 1 <= collector["max_episode_steps"] <= 525
        or not isinstance(collector.get("candidates"), int)
        or collector["candidates"] < minimum_candidates
        or not math.isfinite(float(collector.get("gamma", float("nan"))))
        or not 0 < float(collector["gamma"]) <= 1):
        raise ValueError("P1P requires complete native collector configuration and repeated candidate branches")
    tests = raw.get("tests")
    if (not isinstance(tests, list) or len(tests) != len(TESTS)
        or any(not isinstance(x, dict) for x in tests)
        or {str(x.get("id")) for x in tests} != set(TESTS)):
        raise ValueError("raw receipt must contain exactly P1P-A through P1P-D")
    by_id = {str(x["id"]): x for x in tests}
    group_summary: dict[str, list[dict[str, Any]]] = {}
    all_groups: dict[str, list[dict[str, Any]]] = {}
    episode_rows: dict[str, list[dict[str, Any]]] = {}
    reference_population: dict[tuple[int, int, str], tuple[str, str, tuple[float, ...], tuple[str, ...]]] = {}
    expected_groups = {(seed, h, p) for seed in seeds for h in HORIZONS for p in PROPOSALS}
    for test_id, contract in TESTS.items():
        row = by_id[test_id]
        for key, value in contract.items():
            if row.get(key) != value:
                raise ValueError(f"{test_id} has incorrect {key}; expected {value!r}")
        groups = row.get("candidate_groups") or []
        seen = set()
        computed = []
        for group in groups:
            if str(group.get("proposal")) not in PROPOSALS:
                raise ValueError(f"{test_id} has an unknown proposal type")
            key = (int(group["seed"]), int(group["horizon"]), str(group["proposal"]))
            if key not in expected_groups or key in seen:
                raise ValueError(f"{test_id} contains an unexpected or duplicate candidate group: {key}")
            if len(group.get("predicted_scores", [])) < minimum_candidates:
                raise ValueError(f"{test_id} candidate group has fewer than {minimum_candidates} candidates")
            if len(group["predicted_scores"]) != collector["candidates"]:
                raise ValueError("candidate population must match the frozen collector count")
            if test_id == "P1P-A" and group["predicted_scores"] != group["realized_returns"]:
                raise ValueError("P1P-A environment oracle must score actual replayed candidate returns")
            if test_id == "P1P-A":
                proofs = group.get("branch_proofs")
                if not isinstance(proofs, list) or len(proofs) != collector["candidates"]:
                    raise ValueError("P1P-A needs one repeated branch proof per candidate")
                for i, proof in enumerate(proofs):
                    if not isinstance(proof, dict):
                        raise ValueError("P1P-A branch proof must be an object")
                    hashes = proof.get("trace_sha256_by_replay")
                    rewards = proof.get("reward_trace")
                    if (proof.get("state_sha256") != group.get("state_id")
                        or not isinstance(hashes, list) or len(hashes) != 2
                        or any(not isinstance(h, str) or len(h) != 64 for h in hashes)
                        or hashes[0] != hashes[1]
                        or not isinstance(rewards, list) or not 1 <= len(rewards) <= group["horizon"]
                        or any(not isinstance(r, (float, int)) or not math.isfinite(r) for r in rewards)
                        or not math.isclose(sum(float(r) * collector["gamma"] ** t
                                                for t, r in enumerate(rewards)),
                                            float(group["realized_returns"][i]), rel_tol=1e-6, abs_tol=1e-6)):
                        raise ValueError("P1P-A branch proof does not reproduce its real return")
            sequence_hash = hashlib.sha256(json.dumps(group["action_sequences"], separators=(",", ":")).encode()).hexdigest()
            population = (str(group["state_id"]), sequence_hash,
                          tuple(float(value) for value in group["realized_returns"]),
                          tuple(str(origin) for origin in group["origins"]))
            if test_id == "P1P-A":
                reference_population[key] = population
            elif reference_population.get(key) != population:
                raise ValueError(f"{test_id} candidate group must reuse P1P-A's fixed state, actions, "
                                 "candidate origins, and realized environment returns")
            seen.add(key)
            computed.append(_candidate_group_metrics(group))
        if seen != expected_groups:
            missing = sorted(expected_groups - seen)[:4]
            raise ValueError(f"{test_id} candidate groups are incomplete; missing examples: {missing}")
        episodes = row.get("episode_results") or []
        episode_keys = {(int(e["seed"]), int(e["horizon"]), str(e["proposal"])) for e in episodes}
        required_episode_keys = ({(seed, 8, "mixed") for seed in seeds}
                                 if test_id == "P1P-A" else expected_groups)
        if len(episodes) != len(required_episode_keys) or episode_keys != required_episode_keys:
            raise ValueError(f"{test_id} requires exactly the preregistered closed-loop episodes")
        for e in episodes:
            if not isinstance(e.get("success"), bool) or not math.isfinite(float(e.get("realized_return"))):
                raise ValueError(f"{test_id} episode results need boolean success and finite realized_return")
        if row.get("oracle_reward_provenance") is not None:
            raise ValueError("imagined-state oracle reward claims are not supported by the v2 protocol")
        group_summary[test_id] = computed
        all_groups[test_id] = groups
        episode_rows[test_id] = episodes

    random_rows = raw.get("random_baseline") or []
    random_map = {int(x["seed"]): x for x in random_rows}
    if len(random_rows) != len(seeds) or set(random_map) != set(seeds):
        raise ValueError("random baseline must include the same fixed evaluation seeds as the branch comparison")
    oracle = (by_id["P1P-A"].get("oracle_search_episode_results") or [])
    oracle_map = {int(x["seed"]): x for x in oracle}
    if len(oracle) != len(seeds) or set(oracle_map) != set(seeds):
        raise ValueError("P1P-A needs one preregistered closed-loop oracle-search outcome per fixed seed")
    oracle_success = _mean([float(bool(oracle_map[s]["success"])) for s in seeds]) or 0.0
    random_success = _mean([float(bool(random_map[s]["success"])) for s in seeds]) or 0.0
    oracle_return = _mean([float(oracle_map[s]["realized_return"]) for s in seeds])
    random_return = _mean([float(random_map[s]["realized_return"]) for s in seeds])
    if not all(isinstance(oracle_map[s].get("success"), bool) and math.isfinite(float(oracle_map[s].get("realized_return"))) for s in seeds):
        raise ValueError("oracle-search episodes need boolean success and finite realized return")
    if not all(isinstance(random_map[s].get("success"), bool) and math.isfinite(float(random_map[s].get("realized_return"))) for s in seeds):
        raise ValueError("random-baseline episodes need boolean success and finite realized return")
    oracle_episodes = {(int(e["seed"]), int(e["horizon"]), str(e["proposal"])): e
                       for e in episode_rows["P1P-A"]}
    if any(oracle_map[seed]["success"] != oracle_episodes[(seed, 8, "mixed")]["success"]
           or float(oracle_map[seed]["realized_return"]) !=
           float(oracle_episodes[(seed, 8, "mixed")]["realized_return"])
           for seed in seeds):
        raise ValueError("oracle-search summary must equal the measured P1P-A H8 mixed episode")
    oracle_navigates = (oracle_success >= min_oracle_success_rate and
                        oracle_success - random_success >= min_oracle_success_gain)

    horizons = []
    for horizon in HORIZONS:
        metrics = [x for x in group_summary["P1P-B"] if x["horizon"] == horizon]
        rho = _mean([float(x["spearman_predicted_vs_realized"]) for x in metrics])
        top = _mean([float(x["top_k_agreement"]) for x in metrics])
        passes = bool(rho is not None and top is not None and rho >= min_spearman and top >= min_top_k_agreement)
        horizons.append({"horizon": horizon, "mean_spearman": rho, "mean_top_k_agreement": top,
                         "candidate_groups": len(metrics), "meets_decision_ranking_thresholds": passes})
    qualified_horizon = 0
    prefix_open = True
    for row in horizons:
        if prefix_open and row["meets_decision_ranking_thresholds"]:
            qualified_horizon = int(row["horizon"])
        else:
            prefix_open = False

    deltas = {
        "learned_terminal_value_vs_none": _paired_delta(episode_rows["P1P-B"], episode_rows["P1P-C"]),
        "learned_risk_vs_none": _paired_delta(episode_rows["P1P-C"], episode_rows["P1P-D"]),
    }
    proposal_effects = {}
    for test_id in ("P1P-B", "P1P-C", "P1P-D"):
        rows = episode_rows[test_id]
        by_hp = {(int(r["seed"]), int(r["horizon"])): {} for r in rows}
        for r in rows:
            by_hp[(int(r["seed"]), int(r["horizon"]))][str(r["proposal"])] = r
        paired = [v for v in by_hp.values() if set(v) == set(PROPOSALS)]
        gm = group_summary[test_id]
        paired_groups = {}
        for r in gm:
            paired_groups.setdefault((r["seed"], r["horizon"]), {})[r["proposal"]] = r
        pg = [v for v in paired_groups.values() if set(v) == set(PROPOSALS)]
        proposal_effects[test_id] = {
            "mixed_minus_actor_success": _mean([float(bool(v["mixed"]["success"])) - float(bool(v["actor_seeded"]["success"])) for v in paired]),
            "mixed_minus_actor_return": _mean([float(v["mixed"]["realized_return"]) - float(v["actor_seeded"]["realized_return"]) for v in paired]),
            "mixed_minus_actor_unique_sequence_fraction": _mean([v["mixed"]["unique_sequence_fraction"] - v["actor_seeded"]["unique_sequence_fraction"] for v in pg]),
            "mixed_minus_actor_action_entropy": _mean([v["mixed"]["normalized_first_action_entropy"] - v["actor_seeded"]["normalized_first_action_entropy"] for v in pg]),
            "mean_actor_candidate_origin_fractions": _mean_dict([v["actor_seeded"]["origin_fractions"] for v in pg]),
            "mean_mixed_candidate_origin_fractions": _mean_dict([v["mixed"]["origin_fractions"] for v in pg]),
        }
    question_answers = {
        "oracle_search_navigates": {"answered": True, "value": bool(oracle_navigates), "success_rate": oracle_success,
                                    "random_success_rate": random_success, "mean_return": oracle_return,
                                    "random_mean_return": random_return, "minimum_success_rate": min_oracle_success_rate,
                                    "minimum_success_gain": min_oracle_success_gain},
        "joint_model_reward_ranking_horizon": {"answered": True, "qualified_contiguous_horizon": qualified_horizon,
                                          "per_horizon": horizons, "minimum_spearman": min_spearman,
                                          "minimum_top_k_agreement": min_top_k_agreement},
        "value_and_risk_effects": {"answered": True, "paired_deltas": deltas},
        "actor_seed_search_restriction": {"answered": True, "proposal_effects": proposal_effects},
    }
    diagnosis_complete = all(x["answered"] for x in question_answers.values())
    p2_gate = (diagnosis_complete and oracle_navigates and collector["max_episode_steps"] == 525
               and qualified_horizon >= int(minimum_qualified_horizon_for_p2))
    serial = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return {
        "format": FORMAT_REPORT,
        "status": "PASS" if p2_gate else "BLOCKED",
        "diagnosis_complete": diagnosis_complete,
        "p2_authorized_by_planner_gate": bool(p2_gate),
        "scenario": "my_way_home",
        "seed_ids": seeds,
        "horizons": list(HORIZONS),
        "checkpoint_sha256": checkpoint,
        "identifiability": {"dynamics_and_reward": "joint_only", "unresolved_causes": ["dynamics", "reward"]},
        "raw_evidence_sha256": hashlib.sha256(serial).hexdigest(),
        "candidate_ranking": {test_id: _aggregate_groups(group_summary[test_id]) for test_id in TESTS},
        "question_answers": question_answers,
        "failure_localization": (_localize(question_answers) if collector["max_episode_steps"] == 525
                                 else ["shortened episodes cannot authorize full-length P2 navigation"]
                                 + _localize(question_answers)),
        "thresholds": {"minimum_candidates": minimum_candidates, "minimum_spearman": min_spearman,
                       "minimum_top_k_agreement": min_top_k_agreement,
                       "minimum_oracle_success_gain": min_oracle_success_gain,
                       "minimum_oracle_success_rate": min_oracle_success_rate,
                       "minimum_qualified_horizon_for_p2": int(minimum_qualified_horizon_for_p2)},
        "claim_boundary": (
            "P1P compares paired ViZDoom planner configurations on the supplied fixed seeds and action candidates. "
            "P1P-B tests learned dynamics and reward jointly against real ViZDoom branch returns; it cannot "
            "separate dynamics error from reward error. Results do not establish generalization beyond the "
            "declared scenario, seeds, checkpoint hashes, and horizons."
        ),
    }


def _mean_dict(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = sorted(set().union(*(r.keys() for r in rows))) if rows else []
    return {k: float(_mean([float(r.get(k, 0.0)) for r in rows]) or 0.0) for k in keys}


def _aggregate_groups(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_horizon: dict[int, list[dict[str, Any]]] = {}
    by_proposal: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_horizon.setdefault(int(row["horizon"]), []).append(row)
        by_proposal.setdefault(str(row["proposal"]), []).append(row)
    def agg(items):
        selected = Counter()
        origins = Counter()
        for r in items:
            action = r.get("chosen_first_action")
            if action is not None:
                signature = (int(float(action[0]) > .25) - int(float(action[0]) < -.25),
                             int(float(action[1]) > .25) - int(float(action[1]) < -.25),
                             int(float(action[2]) > 0), int(float(action[3]) > 0))
                selected["|".join(map(str, signature))] += 1
            for origin, fraction in (r.get("origin_fractions") or {}).items():
                origins[origin] += float(fraction)
        return {"groups": len(items),
                "mean_spearman": _mean([float(r["spearman_predicted_vs_realized"]) for r in items]),
                "mean_top_k_agreement": _mean([float(r["top_k_agreement"]) for r in items]),
                "mean_unique_sequence_fraction": _mean([float(r["unique_sequence_fraction"]) for r in items]),
                "mean_normalized_first_action_entropy": _mean([float(r["normalized_first_action_entropy"]) for r in items]),
                "mean_best_minus_median_score": _mean([float(r["best_minus_median_predicted_score"]) for r in items]),
                "mean_planning_latency_ms": _mean([float(r["planning_latency_ms"]) for r in items]),
                "chosen_first_action_distribution": {k: v / max(1, len(items)) for k, v in sorted(selected.items())},
                "mean_candidate_origin_fraction_by_source": {k: v / max(1, len(items)) for k, v in sorted(origins.items())}}
    return {"by_horizon": {str(k): agg(v) for k, v in sorted(by_horizon.items())},
            "by_proposal": {k: agg(v) for k, v in sorted(by_proposal.items())}}


def _localize(answers: dict[str, Any]) -> list[str]:
    notes = []
    if not answers["oracle_search_navigates"]["value"]:
        notes.append("oracle search does not clear the preregistered navigation check; fix search/objective setup before P2")
    horizon = answers["joint_model_reward_ranking_horizon"]["qualified_contiguous_horizon"]
    if horizon == 0:
        notes.append("learned dynamics and reward jointly fail candidate ranking; separate experiments are needed to identify the cause")
    elif horizon < 8:
        notes.append(f"learned ranking horizon {horizon} is shorter than the shipped P2 planner horizon 8")
    for key, label in (("learned_terminal_value_vs_none", "terminal value"),
                       ("learned_risk_vs_none", "risk penalty")):
        delta = answers["value_and_risk_effects"]["paired_deltas"][key]
        if delta["mean_success_delta"] is not None and delta["mean_success_delta"] < 0:
            notes.append(f"{label} reduces paired success in this diagnostic")
    effects = answers["actor_seed_search_restriction"]["proposal_effects"]
    if any(row["mixed_minus_actor_unique_sequence_fraction"] > .10 and
           (row["mixed_minus_actor_success"] or 0) > 0 for row in effects.values()):
        notes.append("mixed proposals improve diversity and paired success; actor seeding may constrain search")
    return notes or ["no tested component showed a blocking regression in the supplied paired evidence"]


def write_planner_diagnostic_report(raw_path: str | Path, output: str | Path, **thresholds) -> dict[str, Any]:
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("raw P1P evidence must be a JSON object")
    report = build_planner_diagnostic_report(raw, **thresholds)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report


def validate_reward_diagnostic_report(payload: dict[str, Any]) -> tuple[bool, str]:
    """P1D paired gate: reward scaling is useful only when task behavior improves."""
    if payload.get("format") != "awa-v2.38.6-reward-diagnostic-v1":
        return False, "unsupported_reward_diagnostic_format"
    if payload.get("scenario") != "my_way_home":
        return False, "reward_diagnostic_must_use_my_way_home"
    if (payload.get("backend") or {}).get("real_vizdoom") is not True:
        return False, "reward_diagnostic_requires_native_vizdoom_receipt"
    contract = payload.get("fixed_contract") or {}
    if (contract.get("scenario") != "my_way_home" or contract.get("track") != "structured"
        or not contract.get("source_dataset_sha256") or not contract.get("shared_nonreward_data_sha256")
        or not contract.get("optimizer") or not contract.get("checkpoint_selection")):
        return False, "paired_control_contract_incomplete"
    seeds = payload.get("training_seeds") or []
    eval_seeds = payload.get("evaluation_seed_ids") or []
    if len(seeds) != 3 or len(set(seeds)) != 3 or len(eval_seeds) < 6 or len(set(eval_seeds)) != len(eval_seeds):
        return False, "missing_three_training_seeds_or_six_fixed_evaluation_seeds"
    arms = payload.get("arms") or {}
    if set(arms) != {"raw_reward", "clipped_reward"}:
        return False, "both_raw_and_clipped_reward_arms_required"
    rows = {}
    if (arms["raw_reward"].get("reward_transform", {}).get("kind") != "identity"
        or arms["clipped_reward"].get("reward_transform", {}).get("kind") != "clip"
        or float(arms["clipped_reward"].get("reward_transform", {}).get("min", 0.0)) != -1.0
        or float(arms["clipped_reward"].get("reward_transform", {}).get("max", 0.0)) != 1.0):
        return False, "reward_arms_must_be_identity_vs_frozen_clip_minus1_plus1"
    for arm, item in arms.items():
        per_seed = item.get("per_training_seed") or []
        if len(per_seed) != len(seeds) or {int(x.get("seed", -1)) for x in per_seed} != {int(x) for x in seeds}:
            return False, f"{arm}_missing_training_seed_results"
        for seed_row in per_seed:
            if int(seed_row.get("model_initialization_seed", -1)) != int(seed_row["seed"]):
                return False, f"{arm}_initialization_seed_not_paired"
            hashes = seed_row.get("checkpoint_sha256") or {}
            if any(not isinstance(hashes.get(k), str) or len(hashes[k]) != 64 for k in ("world", "actor")):
                return False, f"{arm}_checkpoint_hashes_missing"
            episodes = seed_row.get("episodes") or []
            if len(episodes) != len(eval_seeds) or {int(x.get("seed", -1)) for x in episodes} != {int(x) for x in eval_seeds}:
                return False, f"{arm}_fixed_evaluation_seed_mismatch"
            if not all(isinstance(x.get("success"), bool) and math.isfinite(float(x.get("return"))) for x in episodes):
                return False, f"{arm}_episode_metrics_invalid"
            if not all(isinstance(x.get("action_entropy_nats"), (int, float))
                       and math.isfinite(float(x["action_entropy_nats"]))
                       and isinstance(x.get("action_distribution"), dict)
                       for x in episodes):
                return False, f"{arm}_action_distribution_or_entropy_missing"
        rows[arm] = per_seed
    component_stats = payload.get("reward_component_statistics") or {}
    required_components = {"terminal_success", "collision_damage", "living_step_cost", "progress_shaping", "timeout"}
    if not required_components.issubset(component_stats):
        return False, "reward_component_statistics_incomplete"
    deltas = []
    source_hashes = set()
    update_counts = set()
    for arm_rows in rows.values():
        for seed_row in arm_rows:
            source_hashes.add(str(seed_row.get("source_dataset_sha256", "")))
            report = seed_row.get("training_report") or {}
            update_counts.add((int(report.get("world_global_updates", -1)), int(report.get("actor_steps", -1))))
    if source_hashes != {str(contract.get("source_dataset_sha256"))}:
        return False, "arms_did_not_share_the_same_source_dataset"
    component_hashes = {str(seed_row.get("shared_nonreward_data_sha256", ""))
                        for arm_rows in rows.values() for seed_row in arm_rows}
    if component_hashes != {str(contract.get("shared_nonreward_data_sha256"))}:
        return False, "nonreward_dataset_lineage_is_not_shared_across_arms"
    if len(update_counts) != 1 or next(iter(update_counts))[0] <= 0:
        return False, "training_update_counts_are_not_matched"
    if list(contract.get("paired_model_initialization_seeds", [])) != [int(x) for x in seeds]:
        return False, "model_initialization_seed_contract_mismatch"
    if list(contract.get("paired_evaluation_seed_ids", [])) != [int(x) for x in eval_seeds]:
        return False, "evaluation_seed_contract_mismatch"
    deltas = []
    raw_by_seed = {int(row["seed"]): row for row in rows["raw_reward"]}
    clip_by_seed = {int(row["seed"]): row for row in rows["clipped_reward"]}
    for seed in seeds:
        raw_seed, clip_seed = raw_by_seed[int(seed)], clip_by_seed[int(seed)]
        raw_ep = {int(e["seed"]): e for e in raw_seed["episodes"]}
        clip_ep = {int(e["seed"]): e for e in clip_seed["episodes"]}
        if set(raw_ep) != set(clip_ep):
            return False, "paired_episode_seeds_do_not_match"
        deltas.append(_mean([float(bool(clip_ep[s]["success"])) - float(bool(raw_ep[s]["success"])) for s in eval_seeds]) or 0.0)
    raw_all = [e for seed_row in rows["raw_reward"] for e in seed_row["episodes"]]
    clip_all = [e for seed_row in rows["clipped_reward"] for e in seed_row["episodes"]]
    raw_success = _mean([float(bool(e["success"])) for e in raw_all]) or 0.0
    clip_success = _mean([float(bool(e["success"])) for e in clip_all]) or 0.0
    passed = clip_success > raw_success and sum(d > 0 for d in deltas) >= 2
    return passed, ("clipped_reward_improves_paired_navigation" if passed else "clipped_reward_did_not_clear_behavioral_gate")


def build_reward_diagnostic_report(raw: dict[str, Any]) -> dict[str, Any]:
    if raw.get("format") != "awa-v2.38.6-reward-diagnostic-raw-v1":
        raise ValueError("unsupported P1D raw evidence format")
    report = dict(raw)
    report["format"] = "awa-v2.38.6-reward-diagnostic-v1"
    raw_blob = json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["raw_evidence_sha256"] = hashlib.sha256(raw_blob).hexdigest()
    passed, detail = validate_reward_diagnostic_report(report)
    if not passed and detail != "clipped_reward_did_not_clear_behavioral_gate":
        raise ValueError(f"invalid P1D raw evidence: {detail}")
    def arm_summary(arm):
        records = [episode for seed_row in (report.get("arms", {}).get(arm, {}).get("per_training_seed") or [])
                   for episode in (seed_row.get("episodes") or [])]
        return {"episodes": len(records),
                "success_rate": _mean([float(bool(x["success"])) for x in records]),
                "mean_return": _mean([float(x["return"]) for x in records]),
                "mean_action_entropy_nats": _mean([float(x.get("action_entropy_nats", 0.0)) for x in records])}
    raw_seeds = {int(row["seed"]): row for row in report["arms"]["raw_reward"]["per_training_seed"]}
    clip_seeds = {int(row["seed"]): row for row in report["arms"]["clipped_reward"]["per_training_seed"]}
    paired_seed_rows = []
    for seed in sorted(raw_seeds):
        r = {int(x["seed"]): x for x in raw_seeds[seed]["episodes"]}
        c = {int(x["seed"]): x for x in clip_seeds[seed]["episodes"]}
        paired_seed_rows.append({"training_seed": seed,
                                 "raw_success_rate": _mean([float(bool(r[k]["success"])) for k in sorted(r)]),
                                 "clipped_success_rate": _mean([float(bool(c[k]["success"])) for k in sorted(c)]),
                                 "raw_mean_return": _mean([float(r[k]["return"]) for k in sorted(r)]),
                                 "clipped_mean_return": _mean([float(c[k]["return"]) for k in sorted(c)])})
    report["status"] = "PASS" if passed else "BLOCKED"
    report["behavioral_gate_passed"] = bool(passed)
    report["decision"] = detail
    report["outcome_summary"] = {"raw_reward": arm_summary("raw_reward"),
                                 "clipped_reward": arm_summary("clipped_reward"),
                                 "paired_training_seed_results": paired_seed_rows}
    report["claim_boundary"] = (
        "P1D compares raw and clipped reward training with the supplied paired training/evaluation seeds. "
        "A pass establishes only the declared my_way_home diagnostic result; it is not a general reward-design claim."
    )
    return report


def write_reward_diagnostic_report(raw_path: str | Path, output: str | Path) -> dict[str, Any]:
    raw = json.loads(Path(raw_path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("raw P1D evidence must be a JSON object")
    report = build_reward_diagnostic_report(raw)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    return report
