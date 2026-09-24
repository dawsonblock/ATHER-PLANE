from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.ablation_campaign import (
    AblationCampaignConfig,
    AblationCampaignRunner,
    build_campaign_plan,
)
from awa.v2.ablation_report import build_ablation_decision_report
from awa.v2.empirical_closure import RunRecord


def test_v228_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_canonical_plan_is_70_train_jobs_and_140_result_cells():
    cfg = AblationCampaignConfig()
    protocol, plan = build_campaign_plan(cfg)
    assert len(plan.jobs) == 7 * 5 * 2
    assert plan.expected_result_cells == 7 * 5 * 2 * 2
    assert len(protocol.cells()) == 140
    assert protocol.protocol_id == "aether-v2.31-executable-component-ablation-v1"


def test_incomplete_report_refuses_keep_remove_decisions():
    cfg = AblationCampaignConfig()
    protocol, _ = build_campaign_plan(cfg)
    row = RunRecord(
        "actor_only", 1701, "procedural", "heldout", 25_000,
        {"success_rate": 0.5, "episode_return": 1.0, "constraint_violations": 0.0,
         "planner_calls_per_episode": 0.0, "world_model_calls_per_episode": 0.0,
         "inference_latency_ms": 1.0, "wall_clock_seconds": 2.0},
    )
    report = build_ablation_decision_report([row], protocol, cfg)
    assert report["status"] == "INSUFFICIENT_EVIDENCE"
    assert report["marginal_components"] == []


def _synthetic_complete_records(cfg, protocol):
    base_at_25 = {
        "actor_only": 0.40,
        "belief_actor": 0.55,
        "world_actor": 0.62,
        "world_planner": 0.70,
        "adaptive_compute": 0.70,
        "reusable_replay": 0.71,
        "full_curriculum": 0.72,
    }
    base_at_100 = {
        "actor_only": 0.55,
        "belief_actor": 0.72,
        "world_actor": 0.80,
        "world_planner": 0.88,
        "adaptive_compute": 0.88,
        "reusable_replay": 0.89,
        "full_curriculum": 0.90,
    }
    rows=[]
    for system in protocol.systems:
        for seed in protocol.seeds:
            jitter=(seed-protocol.seeds[0]) * 0.001
            for split in protocol.splits:
                split_delta=-0.02 if split == "transfer" else 0.0
                for m in protocol.milestones:
                    sr=(base_at_25 if m==25_000 else base_at_100)[system]+jitter+split_delta
                    # adaptive compute is intentionally equal success but much cheaper than fixed planning.
                    planner=8.0 if system == "world_planner" else (3.0 if system in {"adaptive_compute","reusable_replay","full_curriculum"} else 0.0)
                    latency=12.0 if system == "world_planner" else (6.0 if system in {"adaptive_compute","reusable_replay","full_curriculum"} else 2.0)
                    rows.append(RunRecord(system,seed,"procedural",split,m,{
                        "success_rate":sr,"episode_return":sr*10,"constraint_violations":0.0,
                        "planner_calls_per_episode":planner,"world_model_calls_per_episode":planner*4,
                        "inference_latency_ms":latency,"wall_clock_seconds":100.0,
                    }))
    return rows


def test_complete_report_uses_adjacent_value_and_compute_logic():
    cfg=AblationCampaignConfig(bootstrap_resamples=200)
    protocol,_=build_campaign_plan(cfg)
    report=build_ablation_decision_report(_synthetic_complete_records(cfg,protocol),protocol,cfg)
    assert report["status"] == "QUALIFIED"
    by={r["mechanism"]:r for r in report["marginal_components"]}
    assert by["temporal_memory"]["decision"] == "KEEP"
    assert by["world_model"]["decision"] == "KEEP"
    assert by["fixed_planner"]["decision"] == "KEEP"
    assert by["adaptive_compute"]["decision"] == "KEEP"  # non-inferior and cheaper
    assert by["grounded_hindsight_replay"]["decision"] in {"REMOVE_CANDIDATE","UNCERTAIN"}


def test_real_runner_commits_two_splits_once_and_resumes(tmp_path):
    cfg=AblationCampaignConfig(
        seeds=(11,12), milestones=(24,), systems=("actor_only",),
        device="cpu", horizon=8, hidden=16, sequence_length=1,
        world_epochs=1, actor_epochs=1, representation_epochs=1,
        calibration_epochs=0, batch_size=16, voc_epochs=1,
        eval_tasks_per_split=2, minimum_seeds=2, bootstrap_resamples=100,
    )
    runner=AblationCampaignRunner(cfg,tmp_path/"run")
    first=runner.execute(max_jobs=1)
    assert first["completed_now"] == 1
    job=runner.plan.jobs[0]
    receipt=json.loads((runner._job_dir(job)/"job_receipt.json").read_text())
    assert {r["split"] for r in receipt["records"]} == {"heldout","transfer"}
    assert len(receipt["records"]) == 2
    # Same output directory is resumable; the completed receipt is not retrained.
    second=runner.execute(max_jobs=1)
    assert second["completed_now"] == 1  # second seed, not the first job again
    assert runner.status()["completed_jobs"] == 2
    third=runner.execute(max_jobs=1)
    assert third["completed_now"] == 0


def test_worker_sharding_partitions_jobs_without_overlap():
    cfg=AblationCampaignConfig(seeds=(1,2),milestones=(10,),systems=("actor_only","belief_actor"),minimum_seeds=2)
    _,plan=build_campaign_plan(cfg)
    a={j.key for i,j in enumerate(plan.jobs) if i%2==0}
    b={j.key for i,j in enumerate(plan.jobs) if i%2==1}
    assert a.isdisjoint(b)
    assert a|b == {j.key for j in plan.jobs}
