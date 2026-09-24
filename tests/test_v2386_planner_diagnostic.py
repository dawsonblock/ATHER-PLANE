from __future__ import annotations

import numpy as np

from awa.v2.research_os.branch_replay import DEFAULT_PROBES, qualify_vizdoom_reset_replay
from awa.v2.research_os.planner_diagnostic import (
    build_planner_diagnostic_report,
    build_reward_diagnostic_report,
)


def _p1p_raw():
    seeds = [101, 102, 103, 104, 105, 106]
    horizons = [1, 2, 4, 8, 16, 32]
    tests = []
    contracts = {
        "P1P-A": {"dynamics": "environment", "reward": "oracle", "terminal_value": "none", "risk": "none"},
        "P1P-B": {"dynamics": "learned", "reward": "learned", "terminal_value": "none", "risk": "none"},
        "P1P-C": {"dynamics": "learned", "reward": "learned", "terminal_value": "learned", "risk": "none"},
        "P1P-D": {"dynamics": "learned", "reward": "learned", "terminal_value": "learned", "risk": "learned"},
    }
    for test_id, contract in contracts.items():
        groups, episode_results = [], []
        for seed in seeds:
            for horizon in horizons:
                for proposal in ("actor_seeded", "mixed"):
                    actions = []
                    for index in range(16):
                        action = [float(index % 3 - 1), float((index // 3) % 3 - 1), 1.0, -1.0]
                        actions.append([action[:] for _ in range(horizon)])
                    groups.append({
                        "seed": seed, "horizon": horizon, "proposal": proposal,
                        "state_id": f"snapshot-{seed}-{horizon}-{proposal}",
                        "predicted_scores": list(map(float, range(16))),
                        "realized_returns": list(map(float, range(16))),
                        "action_sequences": actions,
                        "origins": (["actor"] * 16 if proposal == "actor_seeded" else ["actor"] * 8 + ["random"] * 8),
                        "chosen_first_action": [0.0, 1.0, 1.0, -1.0],
                        "planning_latency_ms": 2.5,
                    })
                    episode_results.append({"seed": seed, "horizon": horizon, "proposal": proposal,
                                            "success": True, "realized_return": 1.0})
        row = {"id": test_id, **contract, "candidate_groups": groups, "episode_results": episode_results}
        if test_id == "P1P-A":
            row["oracle_search_episode_results"] = [{"seed": seed, "success": True, "realized_return": 1.0}
                                                     for seed in seeds]
        tests.append(row)
    return {"format": "awa-v2.38.6-planner-diagnostic-raw-v2", "scenario": "my_way_home",
            "seed_ids": seeds, "horizons": horizons,
            "checkpoint_sha256": {"world": "b" * 64, "actor": "c" * 64},
            "branch_replay_report_sha256": "d" * 64,
            "tests": tests,
            "random_baseline": [{"seed": seed, "success": False, "realized_return": -1.0} for seed in seeds]}


def _p1p_replay_fixture(seeds):
    return {"format": "awa-v2.38.6-vizdoom-reset-replay-v1", "status": "PASS",
            "qualified": True, "real_backend_checked": True,
            "seeds": seeds, "probes": list(DEFAULT_PROBES),
            "results": [{"seed": seed, "probe": probe["id"], "candidate": index,
                         "replays": 2, "identical": True,
                         "state_sha256_by_replay": ["1" * 64] * 2,
                         "trace_sha256_by_replay": [str(index + 2) * 64] * 2,
                         "errors": []}
                        for seed in seeds for probe in DEFAULT_PROBES
                        for index in range(len(probe["candidates"]))]}


def test_planner_diagnostic_computes_oracle_ladder_and_paired_effects():
    report = build_planner_diagnostic_report(_p1p_raw())
    assert report["status"] == "PASS"
    assert report["diagnosis_complete"] is True
    assert report["question_answers"]["joint_model_reward_ranking_horizon"]["qualified_contiguous_horizon"] == 32
    assert report["identifiability"]["dynamics_and_reward"] == "joint_only"
    assert "learned_reward_vs_oracle_reward" not in report["question_answers"]["value_and_risk_effects"]["paired_deltas"]
    assert report["candidate_ranking"]["P1P-B"]["by_horizon"]["4"]["mean_spearman"] == 1.0
    assert report["candidate_ranking"]["P1P-B"]["by_proposal"]["mixed"]["chosen_first_action_distribution"]


def test_planner_diagnostic_rejects_legacy_imagined_reward_oracle():
    raw = _p1p_raw()
    raw["format"] = "awa-v2.38.6-planner-diagnostic-raw-v1"
    try:
        build_planner_diagnostic_report(raw)
    except ValueError as exc:
        assert "raw-v2" in str(exc)
    else:
        raise AssertionError("legacy evidence must not advance the corrected gate")
    raw["format"] = "awa-v2.38.6-planner-diagnostic-raw-v2"
    raw["tests"][1]["oracle_reward_provenance"] = {"uses_learned_reward_head": False}
    try:
        build_planner_diagnostic_report(raw)
    except ValueError as exc:
        assert "oracle reward claims" in str(exc)
    else:
        raise AssertionError("imagined-state oracle claims must be rejected")


def test_planner_diagnostic_rejects_incomplete_oracle_ladder():
    raw = _p1p_raw()
    raw["tests"] = raw["tests"][:-1]
    try:
        build_planner_diagnostic_report(raw)
    except ValueError as exc:
        assert "exactly P1P-A through P1P-D" in str(exc)
    else:
        raise AssertionError("incomplete P1P evidence must fail closed")


def test_planner_diagnostic_blocks_p2_when_ranking_horizon_is_below_eight():
    raw = _p1p_raw()
    for group in next(row for row in raw["tests"] if row["id"] == "P1P-B")["candidate_groups"]:
        if group["horizon"] >= 8:
            group["predicted_scores"] = list(reversed(group["predicted_scores"]))
    report = build_planner_diagnostic_report(raw)
    assert report["question_answers"]["joint_model_reward_ranking_horizon"]["qualified_contiguous_horizon"] == 4
    assert report["status"] == "BLOCKED"
    assert report["p2_authorized_by_planner_gate"] is False


def _p1d_raw(clipped_successes: int):
    train_seeds = [7101, 7102, 7103]
    eval_seeds = [9101, 9102, 9103, 9104, 9105, 9106]
    arms = {}
    for name, transform in (("raw_reward", {"kind": "identity"}),
                            ("clipped_reward", {"kind": "clip", "min": -1.0, "max": 1.0})):
        seed_rows = []
        for seed in train_seeds:
            succeeds = (clipped_successes if name == "clipped_reward" else 0)
            episodes = [{"seed": eval_seed, "success": i < succeeds, "return": float(i)}
                        for i, eval_seed in enumerate(eval_seeds)]
            seed_rows.append({"seed": seed, "source_dataset_sha256": "d" * 64,
                              "shared_nonreward_data_sha256": "e" * 64,
                              "model_initialization_seed": seed,
                              "checkpoint_sha256": {"world": "f" * 64, "actor": "a" * 64},
                              "training_report": {"world_global_updates": 20, "actor_steps": 20},
                              "episodes": [row | {"action_entropy_nats": 0.5, "action_distribution": {"forward": .5}}
                                           for row in episodes]})
        arms[name] = {"reward_transform": transform, "per_training_seed": seed_rows}
    return {"format": "awa-v2.38.6-reward-diagnostic-raw-v1", "scenario": "my_way_home",
            "backend": {"real_vizdoom": True},
            "training_seeds": train_seeds, "evaluation_seed_ids": eval_seeds,
            "fixed_contract": {"scenario": "my_way_home", "track": "structured",
                               "source_dataset_sha256": "d" * 64, "shared_nonreward_data_sha256": "e" * 64,
                               "optimizer": "fixed", "checkpoint_selection": "final",
                               "paired_model_initialization_seeds": train_seeds,
                               "paired_evaluation_seed_ids": eval_seeds},
            "reward_component_statistics": {key: {} for key in
                ("terminal_success", "collision_damage", "living_step_cost", "progress_shaping", "timeout")},
            "arms": arms}


def test_reward_diagnostic_advances_only_on_paired_success_gain():
    passing = build_reward_diagnostic_report(_p1d_raw(clipped_successes=2))
    failing = build_reward_diagnostic_report(_p1d_raw(clipped_successes=0))
    assert passing["status"] == "PASS"
    assert passing["outcome_summary"]["clipped_reward"]["success_rate"] > passing["outcome_summary"]["raw_reward"]["success_rate"]
    assert failing["status"] == "BLOCKED"


def test_reward_diagnostic_pairs_training_seeds_by_id_and_rejects_duplicate_episodes():
    raw = _p1d_raw(clipped_successes=2)
    raw["arms"]["clipped_reward"]["per_training_seed"].reverse()
    assert build_reward_diagnostic_report(raw)["status"] == "PASS"
    episodes = raw["arms"]["clipped_reward"]["per_training_seed"][0]["episodes"]
    episodes.append(dict(episodes[0]))
    try:
        build_reward_diagnostic_report(raw)
    except ValueError as exc:
        assert "fixed_evaluation_seed_mismatch" in str(exc)
    else:
        raise AssertionError("duplicate evaluation episodes must fail closed")


def test_planner_diagnostic_rejects_duplicate_test_id():
    raw = _p1p_raw()
    raw["tests"].append(raw["tests"][0])
    try:
        build_planner_diagnostic_report(raw)
    except ValueError as exc:
        assert "exactly P1P-A through P1P-D" in str(exc)
    else:
        raise AssertionError("duplicate oracle ladder entries must fail closed")


def test_planner_diagnostic_rejects_conflicting_real_branch_returns():
    raw = _p1p_raw()
    raw["tests"][1]["candidate_groups"][0]["realized_returns"][0] += 1.0
    try:
        build_planner_diagnostic_report(raw)
    except ValueError as exc:
        assert "realized environment returns" in str(exc)
    else:
        raise AssertionError("identical fixed branches cannot have different realized returns")


def test_reset_replay_preflight_detects_inconsistent_branch_rewards():
    class ReplayEnv:
        def __init__(self, reward_offset=0.0):
            self.reward_offset = reward_offset

        def reset(self, *, seed):
            self.seed = seed
            self.step_index = 0
            return np.asarray([seed, 0], dtype=np.float32), {}

        def step(self, action):
            self.step_index += 1
            return (np.asarray([self.seed, self.step_index], dtype=np.float32),
                    float(action[1]) + self.reward_offset, False, False, {"success": False})

        def close(self):
            pass

    probes = [{"id": "fixed_state", "prefix": [[0.0, 1.0, -1.0, -1.0]],
               "candidates": [[[0.0, 1.0, -1.0, -1.0]],
                              [[1.0, 0.0, -1.0, -1.0]]]}]
    assert qualify_vizdoom_reset_replay(ReplayEnv, seeds=[7, 8], probes=probes)["qualified"]

    created = 0

    def changing_env():
        nonlocal created
        created += 1
        return ReplayEnv(reward_offset=float(created))

    result = qualify_vizdoom_reset_replay(changing_env, seeds=[7, 8], probes=probes)
    assert result["status"] == "BLOCKED"
    assert any(not row["identical"] for row in result["results"])
