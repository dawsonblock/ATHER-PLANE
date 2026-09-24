from __future__ import annotations

import torch
from torch import nn

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.ablation_campaign import AblationCampaignConfig, build_campaign_plan
from awa.v2.ablation_report import build_ablation_decision_report, build_compute_schedule_report
from awa.v2.compute_ledger import PhysicalComputeLedger, chunk_schedule
from awa.v2.empirical_closure import RunRecord
from awa.v2.planners import PolicySeededMPPI


class ZeroActor:
    def deterministic_action(self, b):
        return torch.zeros(b.shape[0], 1, device=b.device)


class QuadraticWorld(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy = nn.Parameter(torch.tensor(0.0))

    def imagine_step(self, b, a, deterministic=False):
        nxt = b + a
        reward = -(nxt - 1.0).pow(2)
        return {
            "belief": nxt,
            "reward": reward,
            "value": reward,
            "continuation": torch.ones_like(reward) * 0.99,
            "risk": torch.zeros(b.shape[0], device=b.device),
        }


def test_v229_version_is_closed():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_chunk_schedule_preserves_logical_candidate_budget():
    assert chunk_schedule(8, None) == (8,)
    assert chunk_schedule(8, 0) == (8,)
    assert chunk_schedule(8, 4) == (4, 4)
    assert chunk_schedule(8, 2) == (2, 2, 2, 2)
    assert chunk_schedule(8, 1) == (1, 1, 1, 1, 1, 1, 1, 1)
    assert sum(chunk_schedule(11, 4)) == 11


def test_mppi_reports_logical_work_separately_from_physical_schedule():
    torch.manual_seed(7)
    world = QuadraticWorld()
    belief = torch.zeros(1, 1)
    full = PolicySeededMPPI(world, ZeroActor(), [-1], [1], horizon=3, candidates=8)
    micro = PolicySeededMPPI(world, ZeroActor(), [-1], [1], horizon=3, candidates=8, world_batch_size=2)

    r_full = full.plan(belief, budget=8)
    torch.manual_seed(7)
    r_micro = micro.plan(belief, budget=8)

    # Search semantics are unchanged: 8 candidate trajectories x 3 model steps.
    assert r_full.world_model_calls == r_micro.world_model_calls == 24
    # Metadata also includes the 3 actor-prior model transitions.
    assert r_full.metadata["logical_world_model_transitions"] == 27
    assert r_micro.metadata["logical_world_model_transitions"] == 27
    # Full vectorization uses 3 prior forwards + 3 search forwards. Microbatch=2
    # requires four search calls per horizon step: 3 + (4 * 3) = 15.
    assert r_full.metadata["physical_world_model_forwards"] == 6
    assert r_micro.metadata["physical_world_model_forwards"] == 15
    assert r_full.metadata["max_batch_size"] == 8
    assert r_micro.metadata["max_batch_size"] == 2
    assert torch.allclose(r_full.action, r_micro.action, atol=1e-6)


def test_physical_compute_ledger_never_invents_energy():
    world = QuadraticWorld()
    planner = PolicySeededMPPI(world, ZeroActor(), [-1], [1], horizon=2, candidates=4)
    result = planner.plan(torch.zeros(1, 1), budget=4)
    ledger = PhysicalComputeLedger()
    ledger.add_planner_result(result, elapsed_seconds=0.01)
    out = ledger.to_dict()
    assert out["logical_world_model_transitions"] == result.metadata["logical_world_model_transitions"]
    assert out["physical_world_model_forwards"] == result.metadata["physical_world_model_forwards"]
    assert out["accelerator_energy_joules"] is None
    assert out["energy_measurement"] == "not_measured"


def _complete_compute_records(cfg, protocol):
    rows = []
    level = {s: 0.50 + i * 0.04 for i, s in enumerate(protocol.systems)}
    for system in protocol.systems:
        for seed in protocol.seeds:
            for split in protocol.splits:
                for milestone in protocol.milestones:
                    success = level[system]
                    physical = 100.0
                    logical = 800.0
                    # Same success and planner count as fixed planning but much better
                    # physical schedule for adaptive compute.
                    if system == "world_planner":
                        success = 0.70; physical = 100.0; logical = 800.0
                    if system == "adaptive_compute":
                        success = 0.70; physical = 50.0; logical = 800.0
                    rows.append(RunRecord(system, seed, "procedural", split, milestone, {
                        "success_rate": success,
                        "episode_return": success * 10,
                        "constraint_violations": 0.0,
                        "planner_calls_per_episode": 4.0 if system in {"world_planner", "adaptive_compute"} else 0.0,
                        "world_model_calls_per_episode": logical if system in {"world_planner", "adaptive_compute"} else 0.0,
                        "logical_world_model_transitions_per_episode": logical if system in {"world_planner", "adaptive_compute"} else 0.0,
                        "physical_world_model_forwards_per_episode": physical if system in {"world_planner", "adaptive_compute"} else 0.0,
                        "mean_world_model_batch_size": 8.0 if system == "world_planner" else (16.0 if system == "adaptive_compute" else 0.0),
                        "logical_transitions_per_forward": (logical / physical) if system in {"world_planner", "adaptive_compute"} else 0.0,
                        "inference_latency_ms": 5.0,
                        "wall_clock_seconds": 100.0,
                        "accelerator_wall_hours": 0.02,
                        "peak_cuda_memory_bytes": 1024.0,
                    }))
    return rows


def test_decision_report_can_credit_better_physical_schedule():
    cfg = AblationCampaignConfig(bootstrap_resamples=100)
    protocol, _ = build_campaign_plan(cfg)
    rows = _complete_compute_records(cfg, protocol)
    report = build_ablation_decision_report(rows, protocol, cfg)
    by = {x["mechanism"]: x for x in report["marginal_components"]}
    assert by["adaptive_compute"]["decision"] == "KEEP"
    assert by["adaptive_compute"]["physical_forward_reduction_fraction"] >= 0.49
    comp = build_compute_schedule_report(rows, protocol, cfg)
    assert comp["status"] == "COMPLETE"
    assert comp["systems"]
    assert all(row["accelerator_energy_joules"] is None for row in comp["systems"])


def test_campaign_config_binds_execution_schedule():
    a = AblationCampaignConfig(planner_world_batch_size=0)
    b = AblationCampaignConfig(planner_world_batch_size=8)
    assert a.sha256 != b.sha256
    pa, plana = build_campaign_plan(a)
    pb, planb = build_campaign_plan(b)
    assert pa.protocol_id == pb.protocol_id == "aether-v2.31-executable-component-ablation-v1"
    assert plana.config_sha256 != planb.config_sha256
