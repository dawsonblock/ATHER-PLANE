from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import torch

from awa.v2.branching import capture_snapshot
from awa.v2.curriculum import ProceduralTaskFactory
from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.game import (
    ACTION_DIM,
    GOAL_DIM,
    OBS_DIM,
    AdaptiveGamePolicy,
    AdaptationMode,
    GameBeliefEncoder,
    LogicalArenaTeacher,
    ProceduralArenaEnv,
    ReusableGameLoop,
    collect_game_dataset,
    evaluate_game_policy_transfer,
    train_game_stack,
)
from awa.v2.game.continual import export_prioritized_game_replay
from awa.v2.transfer import build_transfer_suite
from awa.v2.game.counterfactual import add_counterfactual_branches
from awa.v2.training import ReusableLearningEngine, ReusableLearningConfig


def _roll(env, policy, horizon=140):
    obs = env.reset(seed=env.task.seed)
    info = {"success": False}
    for _ in range(horizon):
        out = policy.act(env, obs) if hasattr(policy, "act") else policy(obs)
        action = out[0] if isinstance(out, tuple) else out
        obs, _, done, info = env.step(action)
        if done:
            break
    return info


def test_stage_is_not_policy_input_and_goal_is_explicit():
    f = ProceduralTaskFactory(210)
    t1, t9 = f.make(1, .2, 0), f.make(9, .2, 0)
    e1, e9 = ProceduralArenaEnv(t1, horizon=30), ProceduralArenaEnv(t9, horizon=30)
    o1, o9 = e1.reset(seed=777), e9.reset(seed=777)
    assert o1.shape == o9.shape == (OBS_DIM,)
    assert o1[7] == 0.0 and o9[7] == 0.0  # objective progress, not stage/12
    assert e1.goal_vector().shape == e9.goal_vector().shape == (GOAL_DIM,)
    assert not np.array_equal(e1.goal_vector(), e9.goal_vector())


def test_stage8_requires_tactical_cover_before_goal_completion():
    task = ProceduralTaskFactory(211).make(8, .3, 0)
    env = ProceduralArenaEnv(task, horizon=30)
    env.reset(seed=task.seed)
    assert env.active_objective == "take_cover"
    env.player = env.goal.copy()
    _, _, _, info = env.step(np.zeros(ACTION_DIM, np.float32))
    assert not info["success"]
    assert env.active_objective == "take_cover"


def test_reward_shaping_tracks_active_key_subgoal_not_final_exit():
    f = ProceduralTaskFactory(212)
    task = f.make(9, .2, 0)
    cfg = dict(task.environment); cfg["obstacle_density"] = 0.0
    task = replace(task, environment=cfg)
    env = ProceduralArenaEnv(task, horizon=40)
    env.reset(seed=task.seed)
    env.player = np.asarray([0.0, 0.0], np.float32)
    env.velocity[:] = 0
    env.key_pos = np.asarray([0.50, 0.0], np.float32)
    env.goal = np.asarray([-0.80, 0.0], np.float32)
    env.door_pos = np.asarray([-0.55, 0.5], np.float32)
    env.obstacles = []
    snap = env.state_dict()
    _, toward, _, _ = env.step(np.asarray([1.0, 0.0, 0.0, 0.0], np.float32))
    env.load_state_dict(snap)
    _, away, _, _ = env.step(np.asarray([-1.0, 0.0, 0.0, 0.0], np.float32))
    assert env.program.objectives[0] == "collect_key"
    assert toward > away


def test_generated_critical_points_are_clear_and_reachable():
    f = ProceduralTaskFactory(213)
    for i in range(40):
        task = f.make(12, .85, i)
        env = ProceduralArenaEnv(task, horizon=50)
        env.reset(seed=task.seed + 17)
        for point in (env.player, env.goal, env.object_pos, env.key_pos, env.door_pos, env.enemy_pos):
            assert env._point_clear(point, margin=0.0, include_door=False)
            assert env._grid_reachable(env.player, point)


def test_goal_conditioned_dataset_and_belief_history(tmp_path: Path):
    f = ProceduralTaskFactory(214)
    tasks = [f.make(s, .25, s) for s in (1, 3, 9)]
    collect_game_dataset(tasks, tmp_path / "game.npz", episodes_per_task=1, horizon=70)
    ds = OfflineTransitionDataset(tmp_path / "game.npz")
    assert ds.manifest.goal_shape == (GOAL_DIM,)
    assert ds.arrays["goals"].shape[1:] == (GOAL_DIM,)
    encoder = GameBeliefEncoder(latent_dim=16, local_dim=16, global_dim=16, goal_latent_dim=8)
    obs = ds.arrays["observations"][0]
    goal = ds.arrays["goals"][0]
    s1 = encoder.initial(1, "cpu"); s2 = encoder.initial(1, "cpu")
    _, b1 = encoder.observe(s1, obs, np.zeros(ACTION_DIM, np.float32), goal, 0)
    _, b2 = encoder.observe(s2, obs, np.asarray([1, -1, .5, 0], np.float32), goal, 0)
    assert b1.shape[-1] == encoder.belief_dim > OBS_DIM
    assert not torch.allclose(b1, b2)


def test_task_and_compositional_ood_change_executable_objectives():
    suite = build_transfer_suite(ProceduralTaskFactory(215), count_per_category=1, difficulty=.5)
    comp = suite["compositional_ood"][0]
    task = suite["task_ood"][0]
    seen = suite["seen"][0]
    assert tuple(comp.goal["objectives"]) != tuple(seen.goal["objectives"])
    assert tuple(task.goal["objectives"]) != tuple(seen.goal["objectives"])
    assert len(comp.goal["objectives"]) > 1 and len(task.goal["objectives"]) > 1
    assert _roll(ProceduralArenaEnv(comp, horizon=140), LogicalArenaTeacher(), 140)["success"]
    assert _roll(ProceduralArenaEnv(task, horizon=140), LogicalArenaTeacher(), 140)["success"]


def test_counterfactual_branch_restores_exact_real_state():
    f = ProceduralTaskFactory(216); task = f.make(6, .3, 0)
    env = ProceduralArenaEnv(task, horizon=60); obs = env.reset(seed=task.seed)
    engine = ReusableLearningEngine(ReusableLearningConfig(seed=216))
    snap = capture_snapshot(env)
    rows = add_counterfactual_branches(
        engine, env, task, obs, env.goal_vector(), np.asarray([.5, .1, .9, 0], np.float32),
        count=2, episode_id="probe", step=0,
    )
    assert len(rows) == 2
    after = env.state_dict()
    assert snap.keys() == after.keys()
    for key in snap:
        a, b = snap[key], after[key]
        if isinstance(a, np.ndarray): assert np.array_equal(a, b), key
        else: assert a == b, key


def test_structural_replay_exports_sgd_weights_and_trains(tmp_path: Path):
    f = ProceduralTaskFactory(217); tasks = [f.make(s, .25, s) for s in (1, 3, 9)]
    loop = ReusableGameLoop(horizon=70)
    loop.run_tasks(tasks, episodes_per_task=1, policy=LogicalArenaTeacher(), counterfactual_branches=1)
    export_prioritized_game_replay(loop.engine, tmp_path / "replay.npz")
    ds = OfflineTransitionDataset(tmp_path / "replay.npz")
    assert "sample_weights" in ds.arrays
    assert "constraints" in ds.arrays and ds.manifest.constraints_shape == (4,)
    assert np.isfinite(ds.arrays["sample_weights"]).all()
    assert np.isclose(ds.arrays["sample_weights"].mean(), 1.0, atol=1e-5)
    report = train_game_stack(
        tmp_path / "replay.npz", tasks, tmp_path / "train", sequence_length=1,
        horizons=(1,), world_epochs=1, actor_epochs=1, calibration_epochs=1,
        batch_size=16, hidden=24, seed=217, one_step_aux_epochs=0,
    )
    assert report.transitions == len(ds)
    assert report.belief_dim > OBS_DIM
    assert np.isfinite(report.value_rmse)


def test_in_context_task_boundary_resets_but_exposures_can_persist():
    encoder = GameBeliefEncoder(latent_dim=8, local_dim=8, global_dim=8, goal_latent_dim=4)
    policy = AdaptiveGamePolicy(encoder, SimpleNamespace(actor=None), world=None)
    f = ProceduralTaskFactory(218); t1, t2 = f.make(1, .2, 0), f.make(2, .2, 1)
    policy.begin_task(t1, preserve_context=True)
    policy._temporal.local.fill_(7.0)
    policy.begin_task(t2, preserve_context=True)
    assert torch.count_nonzero(policy._temporal.local) == 0


def test_transfer_modes_are_explicit_and_prediction_error_is_measured():
    class Policy:
        def __init__(self): self.begins = 0; self.resets = 0
        def begin_task(self, task, preserve_context=False): self.begins += 1
        def reset_task(self, task=None): self.resets += 1
        def act(self, env, obs): return LogicalArenaTeacher().act(env, obs)
        def predict_next(self, env, obs, action): return np.asarray(obs, np.float32)  # persistence predictor
        def end_task(self, task=None): pass
    p = Policy()
    report = evaluate_game_policy_transfer(
        p, base_seed=219, count_per_category=1, exposures=2, horizon=100,
        adaptation_mode=AdaptationMode.ZERO_SHOT,
    )
    assert p.begins == 6 and p.resets == 12
    assert set(report.categories) == {"seen","visual_ood","layout_ood","dynamics_ood","compositional_ood","task_ood"}
    assert any(v.mean_prediction_error > 0 for v in report.categories.values())
