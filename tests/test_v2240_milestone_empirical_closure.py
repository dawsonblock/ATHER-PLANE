import json
from pathlib import Path

import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.empirical_closure import RunRecord
from awa.v2.milestone_analysis import (
    MilestonePromotionConfig,
    build_milestone_scorecard,
    evaluate_milestone_promotion,
)


def _records(*, final_candidate=0.70, split_penalty=None, early_bonus=0.04):
    rows = []
    milestones = (25_000, 100_000, 250_000)
    for system in ("active", "candidate"):
        for seed in (1, 2, 3, 4, 5):
            for split in ("heldout", "transfer"):
                for milestone in milestones:
                    base = 0.50 + 0.05 * (milestones.index(milestone)) + seed * 0.001
                    score = base
                    if system == "candidate":
                        if milestone == milestones[-1]:
                            score = final_candidate + seed * 0.001
                        else:
                            score = base + early_bonus
                        if split_penalty and split == split_penalty[0] and milestone == milestones[-1]:
                            score = base - abs(split_penalty[1])
                    metrics = {
                        "success_rate": score,
                        "episode_return": score * 2,
                        "constraint_violations": 0.0,
                        "planner_calls_per_episode": 3.0 - milestones.index(milestone) * 0.5,
                        "world_model_calls_per_episode": 12.0,
                        "inference_latency_ms": 2.0,
                        "wall_clock_seconds": 10.0,
                    }
                    rows.append(RunRecord(system, seed, "procedural", split, milestone, metrics))
    return rows


def test_v224_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_scorecard_reports_paired_final_and_learning_curve_gain():
    scorecard = build_milestone_scorecard(
        _records(), baseline="active", candidate="candidate", minimum_seeds=5, resamples=500
    )
    assert scorecard["status"] == "PASS"
    assert scorecard["aggregates"]["final_milestone"] == 250_000
    assert scorecard["aggregates"]["final_paired_gain"]["mean"] > 0
    assert scorecard["aggregates"]["curve_paired_gain"]["mean"] > 0
    assert set(scorecard["aggregates"]["split_final_gains"]) == {"heldout", "transfer"}
    assert len(scorecard["per_milestone"]) == 3
    assert scorecard["planner_dependence"] is not None


def test_missing_milestone_fails_closed():
    rows = _records()
    rows = [r for r in rows if not (r.system == "candidate" and r.seed == 3 and r.split == "transfer" and r.transitions == 100_000)]
    scorecard = build_milestone_scorecard(rows, baseline="active", candidate="candidate", minimum_seeds=5, resamples=500)
    assert scorecard["status"] == "FAIL"
    assert any("missing scorecard cells" in x for x in scorecard["failures"])
    receipt = evaluate_milestone_promotion(scorecard)
    assert receipt.status == "FAIL"


def test_early_wins_cannot_hide_final_regression():
    # Candidate dominates early milestones but is worse at the preregistered final milestone.
    scorecard = build_milestone_scorecard(
        _records(final_candidate=0.55, early_bonus=0.20),
        baseline="active", candidate="candidate", minimum_seeds=5, resamples=500,
    )
    assert scorecard["aggregates"]["curve_paired_gain"]["mean"] > 0
    assert scorecard["aggregates"]["final_paired_gain"]["mean"] < 0
    receipt = evaluate_milestone_promotion(scorecard)
    assert receipt.status == "FAIL"
    assert any("final mean paired gain" in x for x in receipt.failures)


def test_split_guardrail_blocks_transfer_regression():
    scorecard = build_milestone_scorecard(
        _records(final_candidate=0.78, split_penalty=("transfer", 0.08), early_bonus=0.08),
        baseline="active", candidate="candidate", minimum_seeds=5, resamples=500,
    )
    # Heldout is strong enough that the pooled final mean is still positive.
    assert scorecard["aggregates"]["final_paired_gain"]["mean"] > 0
    assert scorecard["aggregates"]["split_final_gains"]["transfer"] < -0.02
    receipt = evaluate_milestone_promotion(
        scorecard,
        MilestonePromotionConfig(maximum_split_final_regression=0.02, bootstrap_resamples=500),
    )
    assert receipt.status == "FAIL"
    assert any("split regression" in x for x in receipt.failures)


def test_single_milestone_remains_supported_for_small_smokes():
    rows = [r for r in _records() if r.transitions == 250_000]
    scorecard = build_milestone_scorecard(rows, baseline="active", candidate="candidate", minimum_seeds=5, resamples=500)
    assert scorecard["status"] == "PASS"
    assert scorecard["aggregates"]["curve_paired_gain"]["mean"] == pytest.approx(
        scorecard["aggregates"]["final_paired_gain"]["mean"]
    )
    assert evaluate_milestone_promotion(scorecard, MilestonePromotionConfig(bootstrap_resamples=500)).status == "PASS"
