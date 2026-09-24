import numpy as np
import torch

from awa.environments.continuous_point import ContinuousPointEnv
from awa.v2.curriculum import (
    ProceduralTaskFactory,
    TaskVerifierRegistry,
    ConceptCoverageTracker,
    TaskOutcome,
)
from awa.v2.goals import HindsightGoalRelabeler
from awa.v2.training import (
    TrainingMode,
    IntrinsicRewardComposer,
    LocalRolloutWorkerPool,
    ReusableLearningEngine,
    ReusableLearningConfig,
)
from awa.v2.skills import SkillContractModel, SkillContractTrainer
from awa.v2.transfer import OODCategory, TransferEpisode, TransferBenchmark, build_transfer_suite
from awa.v2.replay import LatentReplayExporter
from awa.v2.datasets import OfflineTransitionDataset


def test_task_verifier_registry_engine_distance_threshold():
    factory = ProceduralTaskFactory(1)
    task = factory.make(1, .2, 0)
    reg = TaskVerifierRegistry()
    assert reg.verify(task, {"success": True, "reward": 2.0}).success

    from dataclasses import replace
    d = replace(task, verifier_name="distance_goal", goal={"target": [1.0, 0.0], "tolerance": .2})
    out = reg.verify(d, {"position": [.9, 0.0]})
    assert out.success and out.metadata["distance"] < .2

    t = replace(task, verifier_name="threshold", goal={"metric": "score", "op": ">=", "threshold": .7})
    assert reg.verify(t, {"score": .8}).success
    assert not reg.verify(t, {"score": .6}).success


def test_hindsight_goal_relabeling_reuses_future_achievements():
    her = HindsightGoalRelabeler(k_future=2, tolerance=.01, seed=2)
    achieved = np.asarray([[0.0], [.5], [1.0]], np.float32)
    actions = np.asarray([[1.0], [1.0], [0.0]], np.float32)
    rows = her.relabel_episode(achieved, actions, desired_goal=np.asarray([2.0], np.float32))
    assert len(rows) == 5
    assert any(r.success for r in rows)
    assert all(r.source_future_index >= r.transition_index for r in rows)


def test_intrinsic_reward_only_changes_exploration_reward():
    c = IntrinsicRewardComposer(.1, .1, .1, .5)
    exploit = c.compose(1.0, novelty=1, information=1, surprise=1, mode=TrainingMode.EXPLOIT)
    explore = c.compose(1.0, novelty=1, information=1, surprise=1, mode=TrainingMode.EXPLORE)
    qualify = c.compose(1.0, novelty=1, information=1, surprise=1, mode=TrainingMode.QUALIFY)
    assert exploit.total == 1.0 and qualify.total == 1.0
    assert explore.total > 1.0


def test_skill_contract_model_trains_and_emits_three_decisions():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(256, 2)).astype(np.float32)
    y = np.stack([x[:, 0] > 0, x[:, 1] > .5, x[:, 0] < -.75], axis=1).astype(np.float32)
    model = SkillContractModel(2, hidden=24)
    before = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(x)), torch.from_numpy(y)).item()
    report = SkillContractTrainer(model, lr=8e-3).fit(x, y, epochs=12, batch_size=64, seed=3)
    after = torch.nn.functional.binary_cross_entropy_with_logits(model(torch.from_numpy(x)), torch.from_numpy(y)).item()
    decision = model.decision([1.0, 1.0])
    assert report["steps"] > 0 and after < before
    assert 0 <= decision.precondition_probability <= 1
    assert 0 <= decision.termination_probability <= 1
    assert 0 <= decision.failure_probability <= 1


def test_transfer_suite_has_all_explicit_ood_categories():
    suite = build_transfer_suite(ProceduralTaskFactory(4), count_per_category=2)
    assert set(suite) == set(OODCategory)
    assert all(len(v) == 2 for v in suite.values())
    assert suite[OODCategory.VISUAL][0].environment["texture_seed"] != suite[OODCategory.SEEN][0].environment["texture_seed"]
    assert suite[OODCategory.DYNAMICS][0].novelty_class == OODCategory.DYNAMICS.value


def test_transfer_benchmark_tracks_adaptation_and_planner_dependency():
    rows = []
    for c in OODCategory:
        for i in range(8):
            rows.append(TransferEpisode(c, f"{c.value}-task", i, i >= 2, float(i >= 2), .8 - .08 * i, .5 - .04 * i))
    report = TransferBenchmark(target_success=.8, window=3).evaluate(rows)
    assert len(report.categories) == 6
    for stats in report.categories.values():
        assert stats.final_window_success == 1.0
        assert stats.planner_dependency_end < stats.planner_dependency_start
        assert stats.episodes_to_target is not None


def test_local_rollout_pool_returns_versioned_episode_envelopes():
    pool = LocalRolloutWorkerPool(workers=2, base_seed=5)
    rows = pool.collect(
        lambda worker_id, seed: ContinuousPointEnv(seed=seed),
        lambda obs: np.zeros(2, np.float32),
        episodes_per_worker=2,
        max_steps=4,
        policy_version=11,
    )
    assert len(rows) == 4
    assert all(r.policy_version == 11 for r in rows)
    assert all(1 <= len(r.payload) <= 4 for r in rows)
    assert len({r.episode_id for r in rows}) == 4


def test_concept_coverage_identifies_weak_concept():
    f = ProceduralTaskFactory(6)
    easy = f.make(1, .1, 0)
    hard = f.make(6, .7, 0)
    tracker = ConceptCoverageTracker()
    for _ in range(5):
        tracker.record(easy, TaskOutcome(easy.task_id, True, 1.0))
        tracker.record(hard, TaskOutcome(hard.task_id, False, 0.0))
    weak = tracker.weakest(3)
    assert weak and weak[0].recent_success == 0.0


def test_reusable_engine_is_closed_loop_resumable_and_exportable(tmp_path):
    engine = ReusableLearningEngine(ReusableLearningConfig(replay_capacity=32, skill_min_examples=1, seed=7))
    task = engine.factory.make(3, .4, 0)
    records = []
    states = []
    actions = []
    for i in range(4):
        state = np.asarray([i * .1, 0], np.float32)
        action = np.asarray([.1, 0], np.float32)
        step = engine.record_step(
            task,
            state=state,
            goal=[1, 0],
            action=action,
            predicted_next=state + [.09, 0],
            actual_next=state + [.1, 0],
            task_reward=.1,
            uncertainty=.1,
            risk=.0,
            success=i == 3,
            metadata={"done": i == 3},
        )
        records.append(step.record); states.append(state); actions.append(action)
    engine.finish_episode(task, records, success=True, reward=1, states=np.asarray(states + [[.4, 0]], np.float32), actions=np.asarray(actions))
    path = engine.save(tmp_path / "state.json")
    restored = ReusableLearningEngine.load(path)
    assert restored.summary()["replay"] == engine.summary()["replay"]
    assert restored.summary()["episodes"] == 1
    assert restored.summary()["skills"] >= 1

    dataset_path = LatentReplayExporter.export(restored.replay.items, tmp_path / "replay.npz")
    dataset = OfflineTransitionDataset(dataset_path)
    assert len(dataset) == 4
    assert dataset.manifest.observation_shape == (2,)


def test_reusable_engine_qualification_mode_does_not_add_intrinsic_reward():
    engine = ReusableLearningEngine(ReusableLearningConfig(seed=8))
    task = engine.factory.make(1, .2, 0)
    out = engine.record_step(
        task,
        state=[0, 0], goal=[1, 0], action=[.1, 0], predicted_next=[0, 0], actual_next=[1, 0],
        task_reward=.25, uncertainty=.9, risk=.1, information_value=1.0, success=True, mode=TrainingMode.QUALIFY,
    )
    assert out.shaped_reward == .25


def test_adaptive_task_generator_targets_weak_concept():
    from awa.v2.curriculum import AdaptiveTaskGenerator
    f = ProceduralTaskFactory(9)
    tracker = ConceptCoverageTracker()
    combat = f.make(6, .6, 0)
    movement = f.make(1, .2, 0)
    for _ in range(8):
        tracker.record(combat, TaskOutcome(combat.task_id, False, 0.0))
        tracker.record(movement, TaskOutcome(movement.task_id, True, 1.0))
    tasks = AdaptiveTaskGenerator(f, tracker).generate(3)
    assert tasks
    assert any("combat" in t.concepts or "damage" in t.concepts for t in tasks)
    assert all(t.novelty_class.startswith("adaptive:") for t in tasks)


def test_hindsight_augmentation_is_inserted_back_into_structural_replay():
    engine = ReusableLearningEngine(ReusableLearningConfig(replay_capacity=64, seed=10))
    task = engine.factory.make(3, .4, 0)
    records = []
    achieved = []
    actions = []
    for i in range(3):
        s = np.asarray([i * .2], np.float32)
        a = np.asarray([.2], np.float32)
        step = engine.record_step(
            task,
            state=s, goal=[1.0], action=a,
            predicted_next=s + .15, actual_next=s + .2,
            task_reward=0.0, uncertainty=.1, risk=.0, success=False,
        )
        records.append(step.record); achieved.append(s + .2); actions.append(a)
    before = len(engine.replay)
    rows = engine.augment_with_hindsight(records, np.asarray(achieved), np.asarray(actions), np.asarray([1.0], np.float32))
    assert rows and len(engine.replay) == before + len(rows)
    assert all((r.metadata or {}).get("hindsight") for r in rows)


def test_skill_execution_manager_uses_learned_contract_gate():
    from awa.v2.skills import SkillSpec, SkillRegistry, SkillContractRegistry, SkillExecutionManager
    registry = SkillRegistry()
    assert registry.register(SkillSpec("move", preconditions=frozenset({"ready"}), success_rate=1.0, confidence=1.0))
    model = SkillContractModel(1, hidden=8)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        # Last layer [precondition, termination, failure]: high precondition, low failure.
        model.net[-1].bias.copy_(torch.tensor([5.0, -5.0, -5.0]))
    contracts = SkillContractRegistry(); contracts.attach("move", model)
    manager = SkillExecutionManager(registry, contracts)
    available = manager.available({"ready"}, [0.0])
    assert [s.name for s in available] == ["move"]
    assert manager.available(set(), [0.0]) == []


def test_distilled_skill_policy_is_executable_and_bounded():
    from awa.v2.skills import DistilledContinuousSkill, SkillPolicyTrainer
    rng = np.random.default_rng(12)
    states = rng.uniform(-1, 1, size=(256, 2)).astype(np.float32)
    actions = np.stack([np.clip(.5 * states[:, 0], -1, 1), np.clip(-.5 * states[:, 1], -1, 1)], axis=1).astype(np.float32)
    model = DistilledContinuousSkill(2, 2, [-1, -1], [1, 1], hidden=32)
    before = torch.square(model(states) - torch.from_numpy(actions)).mean().item()
    report = SkillPolicyTrainer(model, lr=5e-3).fit(states, actions, epochs=12, batch_size=64, seed=12)
    after = torch.square(model(states) - torch.from_numpy(actions)).mean().item()
    out = model.act([1.0, -1.0]).cpu().numpy()
    assert report["steps"] > 0 and after < before
    assert np.all(out <= 1.0 + 1e-6) and np.all(out >= -1.0 - 1e-6)


def test_reusable_engine_can_train_promoted_skill_policy():
    engine = ReusableLearningEngine(ReusableLearningConfig(skill_min_examples=1, seed=13))
    task = engine.factory.make(1, .2, 0)
    states = np.asarray([[0.0, 0.0], [.25, 0.0], [.5, 0.0]], np.float32)
    actions = np.asarray([[.25, 0.0], [.25, 0.0]], np.float32)
    records = []
    for i in range(2):
        records.append(engine.record_step(
            task,
            state=states[i], goal=[.5, 0], action=actions[i],
            predicted_next=states[i + 1], actual_next=states[i + 1],
            task_reward=.5, uncertainty=.05, risk=.0, success=i == 1,
        ).record)
    result = engine.finish_episode(task, records, success=True, reward=1.0, states=states, actions=actions)
    assert result["promoted_skills"]
    name = result["promoted_skills"][0]
    model, report = engine.train_skill_policy(name, [-1, -1], [1, 1], hidden=16, epochs=4, batch_size=2, lr=1e-2, seed=13)
    assert report["examples"] == 2
    assert name in engine.executable_skills.policies
    assert model.act(states[0], records[0].goal).shape == (1, 2)
