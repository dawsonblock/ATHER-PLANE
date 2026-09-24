from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from awa.v2.curriculum import ProceduralTaskFactory
from awa.v2.branching import verify_snapshot_roundtrip
from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.game import (
    ProceduralArenaEnv,
    LogicalArenaTeacher,
    collect_game_dataset,
    ReusableGameLoop,
    train_game_stack,
    evaluate_game_policy_transfer,
    OBS_DIM,
    ACTION_DIM,
)


def test_procedural_arena_fixed_contract_and_snapshot():
    task=ProceduralTaskFactory(29).make(12,.6,0)
    env=ProceduralArenaEnv(task,horizon=40)
    obs=env.reset(seed=task.seed)
    assert obs.shape==(OBS_DIM,)
    assert env.action_dim==ACTION_DIM
    assert env.render_rgb(32).shape==(32,32,3)
    assert verify_snapshot_roundtrip(env,np.zeros(ACTION_DIM,np.float32))


def test_hidden_goal_memory_stage_masks_goal_after_reveal():
    task=ProceduralTaskFactory(29).make(4,.3,0)
    env=ProceduralArenaEnv(task,horizon=30)
    obs=env.reset(seed=task.seed)
    assert obs[10]==1.0
    for _ in range(5):
        obs,_,done,_=env.step(np.zeros(ACTION_DIM,np.float32))
        assert not done
    assert obs[10]==0.0
    assert np.allclose(obs[8:10],0.0)


def test_logical_teacher_solves_basic_and_compositional_tasks():
    f=ProceduralTaskFactory(29); teacher=LogicalArenaTeacher()
    for stage in (1,2,3,4,6,9,12):
        task=f.make(stage,.35,stage)
        env=ProceduralArenaEnv(task,horizon=100)
        obs=env.reset(seed=task.seed)
        info={"success":False}
        for _ in range(100):
            obs,_,done,info=env.step(teacher.act(env,obs))
            if done: break
        assert info["success"], f"teacher failed stage {stage}"


def test_game_dataset_is_valid_and_chain_consistent(tmp_path: Path):
    f=ProceduralTaskFactory(29)
    tasks=[f.make(s,.3,s) for s in (1,2,3)]
    report=collect_game_dataset(tasks,tmp_path/"game.npz",episodes_per_task=1,horizon=60)
    ds=OfflineTransitionDataset(tmp_path/"game.npz")
    assert report.transitions==len(ds)>0
    assert ds.manifest.observation_shape==(OBS_DIM,)
    assert ds.manifest.action_shape==(ACTION_DIM,)
    assert ds.manifest.constraints_shape==(4,)
    raw=np.load(tmp_path/"game.npz")
    done=raw["dones"]
    for i in range(len(done)-1):
        if not done[i]:
            assert np.allclose(raw["next_observations"][i],raw["observations"][i+1])


def test_reusable_game_loop_checkpoint_roundtrip(tmp_path: Path):
    f=ProceduralTaskFactory(29)
    tasks=[f.make(s,.3,s) for s in (1,3,4)]
    loop=ReusableGameLoop(horizon=70)
    report=loop.run_tasks(tasks,episodes_per_task=1,policy=LogicalArenaTeacher())
    assert report.episodes==3 and report.steps>0 and report.replay_size>=report.steps
    path=loop.save(tmp_path/"loop.json")
    restored=ReusableGameLoop.load(path,horizon=70)
    assert restored.engine.total_episodes==loop.engine.total_episodes
    assert len(restored.engine.replay)==len(loop.engine.replay)


def test_game_training_stack_runs_end_to_end(tmp_path: Path):
    f=ProceduralTaskFactory(31)
    tasks=[f.make(s,.25,s) for s in (1,2,3)]
    collect_game_dataset(tasks,tmp_path/"game.npz",episodes_per_task=1,horizon=50)
    report=train_game_stack(tmp_path/"game.npz",tasks,tmp_path/"run",sequence_length=3,horizons=(1,2),world_epochs=1,actor_epochs=1,calibration_epochs=1,batch_size=16,hidden=32,seed=31)
    assert report.transitions>0 and report.sequences>0
    assert set(report.horizon_rmse)=={"1","2"}
    assert (tmp_path/"run"/"game_world.pt").exists()
    assert (tmp_path/"run"/"game_actor.pt").exists()
    assert json.loads((tmp_path/"run"/"game_training_report.json").read_text())["transitions"]==report.transitions


def test_game_transfer_suite_reports_all_categories():
    teacher=LogicalArenaTeacher()
    report=evaluate_game_policy_transfer(teacher,base_seed=41,count_per_category=1,exposures=2,horizon=80)
    assert set(report.categories)=={"seen","visual_ood","layout_ood","dynamics_ood","compositional_ood","task_ood"}
    assert 0.0 <= report.macro_final_success <= 1.0


def test_dynamics_shift_changes_motion_response():
    f=ProceduralTaskFactory(43)
    base=f.make(1,.2,0)
    shifted=f.make(10,.8,0)
    e1=ProceduralArenaEnv(base,horizon=10); e2=ProceduralArenaEnv(shifted,horizon=10)
    e1.reset(seed=100); e2.reset(seed=100)
    # Align player state to isolate dynamics rather than layout/goal.
    e2.player=e1.player.copy(); e2.velocity=e1.velocity.copy()
    p1=e1.player.copy(); p2=e2.player.copy()
    e1.step(np.asarray([1,0,0,0],np.float32)); e2.step(np.asarray([1,0,0,0],np.float32))
    d1=float(np.linalg.norm(e1.player-p1)); d2=float(np.linalg.norm(e2.player-p2))
    assert not np.isclose(d1,d2)


def test_visual_ood_seeds_change_pixels_not_structured_game_state():
    from dataclasses import replace
    f=ProceduralTaskFactory(55)
    base=f.make(8,.4,0)
    env_a=dict(base.environment); env_b=dict(base.environment)
    env_b["texture_seed"]=env_a["texture_seed"]+12345
    env_b["lighting_seed"]=env_a["lighting_seed"]+54321
    a=ProceduralArenaEnv(base,horizon=20); b=ProceduralArenaEnv(replace(base,task_id="visual-b",environment=env_b),horizon=20)
    oa=a.reset(seed=999); ob=b.reset(seed=999)
    # Same simulator state/physics, changed rendering only.
    assert np.allclose(oa,ob)
    assert not np.array_equal(a.render_rgb(32),b.render_rgb(32))


def test_dynamics_parameters_are_hidden_from_structured_observation():
    f=ProceduralTaskFactory(57)
    task=f.make(10,.9,0)
    env=ProceduralArenaEnv(task,horizon=20)
    obs=env.reset(seed=task.seed)
    assert np.allclose(obs[29:32],0.0)
