from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from awa.v2.branching import verify_snapshot_roundtrip, rollout_real_branch
from awa.v2.curriculum import ProceduralTaskFactory
from awa.v2.engram import EngramLite
from awa.v2.experience import RunningNovelty
from awa.v2.replay import LatentReplayExporter
from awa.v2.safety import SafeActionGuard, UnsafeRecoveryError
from awa.v2.skills import SkillContractModel
from awa.v2.training.reusable_engine import ReusableLearningConfig, ReusableLearningEngine
from awa.v2.training.rollout_pool import LocalRolloutWorkerPool
from awa.v2.training.staleness import RolloutEnvelope, StalenessPolicy
from awa.v2.transfer import OODCategory, TransferBenchmark, TransferEpisode
from awa.v2.datasets import OfflineTransitionDataset


def _promote_trainable_skill(engine: ReusableLearningEngine):
    task = engine.factory.make(1, 0.2, 0)
    states = np.asarray([[0.0, 0.0], [0.2, 0.0], [0.4, 0.0]], np.float32)
    actions = np.asarray([[0.2, 0.0], [0.2, 0.0]], np.float32)
    records = []
    for i in range(2):
        records.append(
            engine.record_step(
                task,
                state=states[i],
                goal=[0.4, 0.0],
                action=actions[i],
                predicted_next=states[i + 1],
                actual_next=states[i + 1],
                task_reward=0.5,
                uncertainty=0.1,
                risk=0.0,
                success=i == 1,
            ).record
        )
    result = engine.finish_episode(
        task,
        records,
        success=True,
        reward=1.0,
        states=states,
        actions=actions,
    )
    name = result["promoted_skills"][0]
    model, _ = engine.train_skill_policy(
        name,
        [-1, -1],
        [1, 1],
        hidden=8,
        epochs=2,
        batch_size=2,
        lr=1e-2,
        seed=3,
    )
    contract = SkillContractModel(2, goal_dim=2, hidden=8)
    with torch.no_grad():
        for p in contract.parameters():
            p.zero_()
        contract.net[-1].bias.copy_(torch.tensor([4.0, -4.0, -4.0]))
    engine.executable_skills.contracts.attach(name, contract)
    return name, states[0], records[0].goal, model.act(states[0], records[0].goal).clone()


def test_v280_checkpoint_restores_executable_skill_contract_and_numpy_metadata(tmp_path):
    engine = ReusableLearningEngine(ReusableLearningConfig(skill_min_examples=1, seed=3))
    name, state, goal, before = _promote_trainable_skill(engine)
    task = engine.factory.make(1, 0.2, 99)
    engine.record_step(
        task,
        state=[0, 0],
        goal=[1, 0],
        action=[0, 0],
        predicted_next=[0, 0],
        actual_next=[0, 0],
        task_reward=0,
        uncertainty=0,
        risk=0,
        success=False,
        metadata={"array": np.asarray([1, 2]), "scalar": np.float32(1.5)},
    )
    path = engine.save(tmp_path / "engine.json")
    json.loads(path.read_text())
    restored = ReusableLearningEngine.load(path)
    assert name in restored.executable_skills.policies
    assert restored.executable_skills.contracts.get(name) is not None
    after = restored.executable_skills.act(name, state, goal)
    assert torch.allclose(before, after, atol=1e-7, rtol=1e-7)
    decision = restored.executable_skills.contracts.check(name, state, goal)
    assert decision.can_start and not decision.failed


def test_v280_adaptive_tasks_are_fresh_and_counter_resumes(tmp_path):
    engine = ReusableLearningEngine(ReusableLearningConfig(seed=5))
    a = [t.task_id for t in engine.generate_adaptive_tasks(3)]
    b = [t.task_id for t in engine.generate_adaptive_tasks(3)]
    assert a != b and not (set(a) & set(b))
    restored = ReusableLearningEngine.load(engine.save(tmp_path / "state.json"))
    c = [t.task_id for t in restored.generate_adaptive_tasks(3)]
    assert not (set(a + b) & set(c))


def test_v280_novelty_and_engram_tolerate_mixed_vector_dimensions():
    novelty = RunningNovelty(8)
    assert novelty.observe([1.0, 0.0]) == 1.0
    assert novelty.observe([1.0, 0.0, 0.0]) == 1.0
    memory = EngramLite(capacity=8, buckets=2, seed=1)
    memory.put("x", [1.0, 0.0], {"dim": 2})
    memory.put("x", [1.0, 0.0, 0.0], {"dim": 3})
    # Even a bucket collision must never dot-product incompatible vectors.
    for q in ([1.0, 0.0], [1.0, 0.0, 0.0]):
        for _, entry in memory.query(q, k=8, pattern="x"):
            assert entry.vector.shape == np.asarray(q).reshape(-1).shape


def test_v280_future_policy_rollouts_are_rejected():
    policy = StalenessPolicy(max_lag=4, half_life=2.0)
    assert policy.weight(10, RolloutEnvelope(None, 11)) == 0.0
    assert policy.accept(10, RolloutEnvelope(None, 11)) is False
    assert policy.weight(10, RolloutEnvelope(None, 10)) == pytest.approx(1.0)


class _Gym5SnapshotEnv:
    def __init__(self, seed=0):
        self.x = 0
        self.closed = False
        self.action_dim = 1

    def reset(self, seed=None):
        self.x = 0
        return np.asarray([0.0], np.float32), {"seed": seed}

    def step(self, action):
        self.x += 1
        return np.asarray([float(self.x)], np.float32), 1.0, self.x >= 2, False, {}

    def state_dict(self):
        return {"x": self.x, "observation": np.asarray([float(self.x)], np.float32)}

    def load_state_dict(self, state):
        self.x = int(state["x"])

    def close(self):
        self.closed = True


class _Actor:
    def deterministic_action(self, belief):
        return torch.zeros((belief.shape[0], 1), dtype=torch.float32)


def test_v280_snapshot_branching_supports_gymnasium_5_tuple():
    env = _Gym5SnapshotEnv()
    obs, _ = env.reset()
    assert verify_snapshot_roundtrip(env, np.zeros(1, np.float32))
    snap = env.state_dict()
    result = rollout_real_branch(
        env,
        snap,
        lambda x: torch.as_tensor(x, dtype=torch.float32),
        _Actor(),
        horizon=2,
        start_observation=obs,
    )
    assert result.steps == 2 and result.terminal is True
    assert env.x == 0


def test_v280_rollout_pool_closes_gymnasium_environments():
    created = []

    def factory(worker_id, seed):
        env = _Gym5SnapshotEnv(seed)
        created.append(env)
        return env

    pool = LocalRolloutWorkerPool(workers=2, base_seed=2)
    rows = pool.collect(factory, lambda obs: np.zeros(1, np.float32), episodes_per_worker=2, max_steps=3)
    assert len(rows) == 4
    assert created and all(env.closed for env in created)


class _AbsRisk:
    def aggregate_risk(self, belief, action):
        return torch.abs(action).max(dim=-1, keepdim=True).values


def test_v280_guard_reports_recovery_risk_and_strict_mode_fails_closed():
    guard = SafeActionGuard(_AbsRisk(), [-1], [1], risk_limit=0.5, recovery_action=[0.25])
    out = guard.validate(torch.zeros(1, 1), torch.tensor([[0.9]]))
    assert out.accepted is False
    assert out.risk == pytest.approx(0.25)
    assert out.proposed_risk == pytest.approx(0.9)
    assert out.recovery_safe is True

    strict = SafeActionGuard(
        _AbsRisk(), [-1], [1], risk_limit=0.5, recovery_action=[1.0], strict_recovery=True
    )
    with pytest.raises(UnsafeRecoveryError):
        strict.validate(torch.zeros(1, 1), torch.tensor([[0.9]]))


def test_v280_world_model_export_excludes_hindsight_rows(tmp_path):
    engine = ReusableLearningEngine(ReusableLearningConfig(seed=10))
    task = engine.factory.make(3, 0.4, 0)
    records = []
    achieved = []
    actions = []
    for i in range(3):
        s = np.asarray([i * 0.2], np.float32)
        a = np.asarray([0.2], np.float32)
        records.append(
            engine.record_step(
                task,
                state=s,
                goal=[1.0],
                action=a,
                predicted_next=s + 0.2,
                actual_next=s + 0.2,
                task_reward=0.0,
                uncertainty=0.1,
                risk=0.0,
                success=False,
                metadata={"done": i == 2},
            ).record
        )
        achieved.append(s + 0.2)
        actions.append(a)
    engine.augment_with_hindsight(records, np.asarray(achieved), np.asarray(actions), [1.0])
    path = LatentReplayExporter.export(engine.replay.items, tmp_path / "world.npz")
    ds = OfflineTransitionDataset(path)
    assert len(ds) == len(records)


def test_v280_transfer_episodes_to_target_is_exposure_count_not_zero_based_index():
    rows = [
        TransferEpisode(OODCategory.SEEN, "task", i, True, 1.0)
        for i in range(3)
    ]
    stats = TransferBenchmark(target_success=0.8, window=1).evaluate(rows).categories["seen"]
    assert stats.episodes_to_target == 1.0


def test_v280_replay_export_regroups_interleaved_episode_rows(tmp_path):
    from awa.v2.experience import ExperienceRecord

    def rec(ep, step, s, done=False):
        state = np.asarray([s], np.float32)
        return ExperienceRecord(
            state=state,
            goal=np.asarray([1.0], np.float32),
            action=np.asarray([0.1], np.float32),
            predicted_next=state + 0.1,
            actual_next=state + 0.1,
            reward=0.0,
            success=False,
            prediction_error=0.0,
            novelty=0.0,
            uncertainty=0.0,
            risk=0.0,
            metadata={"episode_id": ep, "step": step, "done": done},
        )

    rows = [
        rec("a", 0, 0.0),
        rec("b", 0, 10.0),
        rec("a", 1, 0.1, True),
        rec("b", 1, 10.1, True),
    ]
    path = LatentReplayExporter.export(rows, tmp_path / "ordered.npz")
    raw = np.load(path, allow_pickle=False)
    assert raw["observations"][:, 0].tolist() == pytest.approx([0.0, 0.1, 10.0, 10.1])
    assert raw["dones"].tolist() == [False, True, False, True]


def test_v280_prediction_error_rejects_shape_broadcast_and_teacher_pool_ignores_nan():
    from awa.v2.experience import ExperienceAnalyzer
    from awa.v2.distill import TeacherPool, TeacherCandidate

    with pytest.raises(ValueError):
        ExperienceAnalyzer.prediction_error(np.zeros((2, 1)), np.zeros((2,)))
    good = TeacherCandidate("good", np.asarray([0.1], np.float32), value=1.0)
    bad = TeacherCandidate("nan", np.asarray([0.0], np.float32), value=float("nan"))
    assert TeacherPool().select([bad, good]).teacher == "good"
