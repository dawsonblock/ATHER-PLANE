from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.evolving_campaign import CampaignCell, runner_from_config
from awa.v2.game import ACTION_DIM, GOAL_DIM, OBS_DIM, AdaptiveGamePolicy, EffortRuntimeTrace
from awa.v2.meta_exploration import DeclarativeExplorationPolicy
from awa.v2.native_campaign import (
    ExplorationPolicyCurriculumAdapter,
    NativeProceduralCampaignRunner,
    NativeProceduralRunnerConfig,
)
from awa.v2.reasoning.effort import AdaptiveReasoningEffortController, EffortLevel


def test_version_is_closed_across_packages():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_curriculum_allocation_is_deterministic_and_policy_sensitive():
    adapter = ExplorationPolicyCurriculumAdapter()
    base = DeclarativeExplorationPolicy("base", quality_weight=2.0, novelty_weight=0.0, transfer_weight=0.0)
    explore = DeclarativeExplorationPolicy("explore", quality_weight=0.0, novelty_weight=2.0, transfer_weight=2.0)
    a = adapter.allocate(base, range(1, 13), 24)
    b = adapter.allocate(base, range(1, 13), 24)
    c = adapter.allocate(explore, range(1, 13), 24)
    assert a == b
    assert sum(a.values()) == 24 == sum(c.values())
    assert c.get(10, 0) + c.get(11, 0) + c.get(12, 0) > a.get(10, 0) + a.get(11, 0) + a.get(12, 0)


def _fake_collect(tasks, output, *, episodes_per_task, horizon, policy):
    n = 48
    observations = np.zeros((n, OBS_DIM), dtype=np.float32)
    next_observations = np.zeros((n, OBS_DIM), dtype=np.float32)
    goals = np.zeros((n, GOAL_DIM), dtype=np.float32)
    actions = np.zeros((n, ACTION_DIM), dtype=np.float32)
    rewards = np.zeros(n, dtype=np.float32)
    dones = np.zeros(n, dtype=np.bool_)
    dones[15::16] = True
    constraints = np.zeros((n, 4), dtype=np.float32)
    np.savez_compressed(
        output,
        observations=observations,
        goals=goals,
        actions=actions,
        rewards=rewards,
        next_observations=next_observations,
        next_goals=goals.copy(),
        dones=dones,
        constraints=constraints,
    )
    return SimpleNamespace(transitions=n)


class _FakeReport:
    def to_dict(self):
        return {
            "actor_success_rate": 0.625,
            "actor_mean_return": 1.5,
            "actor_constraint_violations": 0.25,
            "actor_inference_latency_ms": 0.4,
        }


def _fake_train(dataset, tasks, out_dir, **kwargs):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "game_world.pt").write_bytes(b"world")
    (out / "game_actor.pt").write_bytes(b"actor")
    return _FakeReport()




def _fake_eval(world_checkpoint, actor_checkpoint, tasks, **kwargs):
    return {
        "success_rate": 0.625,
        "mean_return": 1.5,
        "constraint_violations": 0.25,
        "inference_latency_ms": 0.4,
    }


def test_native_runner_emits_grounded_content_addressed_evidence(tmp_path):
    cfg = NativeProceduralRunnerConfig(
        minimum_transitions=32,
        tasks_per_batch=4,
        eval_tasks=3,
        sequence_length=2,
        hidden=16,
        world_epochs=1,
        actor_epochs=1,
        calibration_epochs=0,
        batch_size=8,
        one_step_aux_epochs=0,
    )
    runner = NativeProceduralCampaignRunner(cfg, collect_fn=_fake_collect, train_fn=_fake_train, eval_fn=_fake_eval)
    cell = CampaignCell("active", 11, "procedural", "heldout", 32)
    policy = DeclarativeExplorationPolicy("active")
    evidence = runner.run(cell, policy, tmp_path / "cell")
    assert evidence.cell.key == cell.key
    assert evidence.evidence.grounded
    assert evidence.record.metrics["success_rate"] == pytest.approx(0.625)
    assert evidence.record.metrics["constraint_violations"] == pytest.approx(0.25)
    assert len(evidence.record.checkpoint_sha256) == 64
    assert len(evidence.record.config_sha256) == 64
    assert len(evidence.provenance.dataset_sha256) == 64
    assert len(evidence.provenance.scenario_ids) == 3
    assert (tmp_path / "cell" / "evidence.json").exists()
    assert (tmp_path / "cell" / "collection_receipt.json").exists()


def test_native_runner_refuses_under_budget_or_non_eval_split(tmp_path):
    cfg = NativeProceduralRunnerConfig(minimum_transitions=32)
    runner = NativeProceduralCampaignRunner(cfg, collect_fn=_fake_collect, train_fn=_fake_train, eval_fn=_fake_eval)
    policy = DeclarativeExplorationPolicy("p")
    with pytest.raises(ValueError, match="below native minimum"):
        runner.run(CampaignCell("p", 1, "procedural", "heldout", 16), policy, tmp_path / "low")
    with pytest.raises(ValueError, match="heldout or transfer"):
        runner.run(CampaignCell("p", 1, "procedural", "train", 32), policy, tmp_path / "train")


def test_generic_runner_factory_supports_native_procedural():
    cfg = {
        "runner": {
            "type": "native_procedural",
            "config": {"minimum_transitions": 32, "curriculum_stages": [1, 2, 12]},
        }
    }
    runner = runner_from_config(cfg)
    assert isinstance(runner, NativeProceduralCampaignRunner)
    assert runner.config.curriculum_stages == (1, 2, 12)


class _Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(()))
    def initial(self, batch, device):
        return None
    def observe(self, state, obs, prev, goal, step):
        return None, torch.ones((1, 4), device=self.p.device)
    def reconstruct(self, belief):
        return torch.zeros((1, OBS_DIM)), torch.zeros((1, GOAL_DIM))


class _Actor:
    def deterministic_action(self, belief):
        return torch.zeros((belief.shape[0], ACTION_DIM), device=belief.device)


class _World:
    class U:
        def at_horizon(self, belief, h):
            return torch.full((belief.shape[0],), 0.5, device=belief.device)
    class R:
        def aggregate_risk(self, belief, action):
            return torch.full((belief.shape[0],), 0.1, device=belief.device)
    uncertainty = U()
    risk = R()


class _VOC:
    choices = [("actor", 0), ("policy_mppi", 32)]
    def gains(self, features):
        return torch.tensor([[0.0, 1.0]], device=features.device)


class _Planner:
    def plan(self, belief, *, actor, budget):
        return SimpleNamespace(action=torch.ones((1, ACTION_DIM), device=belief.device), world_model_calls=17)


class _Env:
    task = SimpleNamespace(difficulty=0.8, novelty_class="transfer")
    def goal_vector(self):
        return np.zeros(GOAL_DIM, dtype=np.float32)


def _effort_controller(reliability_threshold=0.25):
    return AdaptiveReasoningEffortController(
        levels=[
            EffortLevel("reflex", "actor", 0, 0.0),
            EffortLevel("shallow", "policy_mppi", 32, 0.25),
        ],
        compute_lambda=0.0,
        latency_lambda=0.0,
        risk_lambda=0.0,
        risk_deliberation_bonus=0.0,
        minimum_model_reliability=reliability_threshold,
    )


def test_adaptive_game_policy_makes_effort_controller_load_bearing():
    encoder = _Encoder()
    policy = AdaptiveGamePolicy(
        encoder,
        SimpleNamespace(actor=_Actor()),
        _World(),
        planners={"policy_mppi": _Planner()},
        voc=_VOC(),
        effort_controller=_effort_controller(),
        model_reliability=1.0,
    )
    policy.reset_task(_Env.task)
    action, used = policy.act(_Env(), np.zeros(OBS_DIM, dtype=np.float32))
    assert used is True
    assert np.all(action == 1.0)
    assert isinstance(policy.last_effort_trace, EffortRuntimeTrace)
    assert policy.last_effort_trace.planner == "policy_mppi"
    assert policy.last_effort_trace.world_model_calls == 17


def test_low_model_reliability_masks_model_planner():
    encoder = _Encoder()
    policy = AdaptiveGamePolicy(
        encoder,
        SimpleNamespace(actor=_Actor()),
        _World(),
        planners={"policy_mppi": _Planner()},
        voc=_VOC(),
        effort_controller=_effort_controller(reliability_threshold=0.5),
        model_reliability=0.1,
    )
    policy.reset_task(_Env.task)
    action, used = policy.act(_Env(), np.zeros(OBS_DIM, dtype=np.float32))
    assert used is False
    assert np.all(action == 0.0)
    assert policy.last_effort_trace.planner == "actor"


def test_empty_history_returns_explicit_bootstrap_plan(tmp_path):
    from awa.v2.evolving_campaign import EvolvingCampaignOrchestrator, ValidationMatrix
    active = DeclarativeExplorationPolicy("active")
    validation = ValidationMatrix(
        seeds=(1, 2, 3, 4, 5), tasks=("procedural",), splits=("heldout",),
        milestones=(32,), minimum_seeds=5,
    )
    o = EvolvingCampaignOrchestrator(tmp_path / "meta", active_policy=active, validation=validation)
    report = o.plan_iteration(max_rounds=2)
    assert report["status"] == "NEEDS_BOOTSTRAP"
    assert report["bootstrap_plan"]["cell_count"] == 5


def test_bootstrap_populates_grounded_history_without_promotion(tmp_path):
    from awa.v2.evolving_campaign import EvolvingCampaignOrchestrator, ValidationMatrix
    from awa.v2.empirical_closure import RunRecord
    from awa.v2.evidence_integrity import RunProvenance
    from awa.v2.evolving_campaign import CampaignEvidence
    from awa.v2.meta_exploration import EvidenceClass

    active = DeclarativeExplorationPolicy("active")
    validation = ValidationMatrix(
        seeds=(1, 2, 3, 4, 5), tasks=("procedural",), splits=("heldout",),
        milestones=(32,), minimum_seeds=5,
    )
    o = EvolvingCampaignOrchestrator(tmp_path / "meta", active_policy=active, validation=validation)

    class BootstrapRunner:
        def run(self, cell, policy, work_dir):
            h = "a" * 64
            metrics = {
                "success_rate": .5, "episode_return": .5, "constraint_violations": 0.0,
                "planner_calls_per_episode": 0.0, "world_model_calls_per_episode": 0.0,
                "inference_latency_ms": 1.0, "wall_clock_seconds": 1.0,
                "normalized_compute_cost": .01, "novelty": .1,
                "information_gain": .1, "uncertainty_reduction": .1,
            }
            r = RunRecord(cell.policy_id, cell.seed, cell.task, cell.split, cell.transitions, metrics, h, h)
            rid = "|".join(map(str, cell.key))
            p = RunProvenance(rid, h, h, h, h, (f"scenario-{cell.seed}",))
            return CampaignEvidence(r, p, EvidenceClass.VALIDATED)

    receipt = o.bootstrap_active(BootstrapRunner())
    assert receipt["status"] == "PASS"
    assert receipt["created"] == 5
    assert o.active_policy.policy_id == "active"
    assert len(o.evidence_store.all()) == 5
