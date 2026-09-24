from __future__ import annotations
import json
import numpy as np
import pytest
import torch

from awa.v2.modular_world import SparseMoETrunk,ModularWorldModel,make_compute_matched_pair
from awa.v2.experiment_ledger import ExperimentLedger
from awa.v2.scaling import ControlledScalingRunner,run_controlled_scaling_experiment
from awa.v2.game import collect_game_dataset,LogicalArenaTeacher,GameBeliefEncoder,GameBeliefSequenceDataset
from awa.v2.datasets import OfflineTransitionDataset
from awa.v2.game.belief import GameBeliefWorldTrainer
from awa.v2.world import MultimodalWorldModel
from awa.v2.curriculum import ProceduralTaskFactory


def _dataset(tmp_path):
    f=ProceduralTaskFactory(2120)
    tasks=[f.make(1,.1,0,'v212'),f.make(3,.15,1,'v212')]
    path=tmp_path/'game.npz'
    collect_game_dataset(tasks,path,episodes_per_task=1,horizon=26,policy=LogicalArenaTeacher())
    return path,tasks


def test_sparse_moe_executes_sparse_routes_and_tracks_load():
    torch.manual_seed(1)
    m=SparseMoETrunk(10,8,experts=4,top_k=1,expert_hidden=7)
    x=torch.randn(13,10); y=m(x)
    assert y.shape==(13,8)
    assert int(m.routing_counts.sum())==13
    assert int((m.routing_counts>0).sum())<=4
    assert m.active_macs()<m.total_macs()


def test_compute_matched_moe_is_within_five_percent_active_macs():
    _,moe,mono_p,moe_p=make_compute_matched_pair(112,4,hidden=64,experts=4,top_k=1,horizons=(1,2))
    ratio=moe_p.trunk_active_macs/mono_p.trunk_active_macs
    assert .95<=ratio<=1.05
    assert moe_p.total_parameters>mono_p.total_parameters
    assert moe_p.active_parameters_approx<moe_p.total_parameters
    assert isinstance(moe,ModularWorldModel)


def test_modular_world_preserves_world_model_interface():
    torch.manual_seed(2)
    m=ModularWorldModel(12,4,hidden=16,components=3,horizons=(1,2),experts=4,top_k=1)
    b=torch.randn(5,12); a=torch.randn(5,4); n=torch.randn(5,12)
    out=m.imagine_step(b,a,deterministic=True)
    assert out['belief'].shape==(5,12)
    assert out['reward'].shape==(5,1)
    assert torch.isfinite(m.nll_loss(b,a,n))
    assert m.terminal_value(b).shape==(5,)


def test_experiment_ledger_resumes_completed_phase_and_binds_config(tmp_path):
    root=tmp_path/'exp'; calls=[]
    led=ExperimentLedger(root,{'a':1},experiment_id='x')
    first=led.run('one',lambda:(calls.append(1) or {'x':2}))
    again=ExperimentLedger(root,{'a':1},experiment_id='x').run('one',lambda:(calls.append(2) or {'x':3}))
    assert first==again=={'x':2} and calls==[1]
    with pytest.raises(ValueError): ExperimentLedger(root,{'a':2},experiment_id='x')


def test_world_trainer_precision_validation_on_cpu():
    enc=GameBeliefEncoder(latent_dim=8,local_dim=8,global_dim=8,goal_latent_dim=4)
    w=MultimodalWorldModel(enc.belief_dim,4,hidden=16,components=2,horizons=(1,2))
    GameBeliefWorldTrainer(enc,w,precision='fp32')
    with pytest.raises(ValueError): GameBeliefWorldTrainer(enc,w,precision='fp16')
    with pytest.raises(ValueError): GameBeliefWorldTrainer(enc,w,precision='wat')


def test_controlled_runner_compares_matched_pair_on_same_sequences(tmp_path):
    path,_=_dataset(tmp_path); ds=OfflineTransitionDataset(path)
    r=ControlledScalingRunner(ds,device='cpu',precision='fp32',seed=12,horizons=(1,2))
    rows=r.model_pair(sequence_length=2,hidden=16,epochs=1,batch_size=16,experts=4,top_k=1)
    assert [x.model_kind for x in rows]==['monolithic','moe']
    assert rows[0].train_sequences==rows[1].train_sequences
    assert abs(rows[0].trunk_active_macs-rows[1].trunk_active_macs)/rows[0].trunk_active_macs<.06
    assert all(np.isfinite(x.final_loss) for x in rows)


def test_data_and_context_scaling_curves_emit_real_points(tmp_path):
    path,_=_dataset(tmp_path); ds=OfflineTransitionDataset(path)
    r=ControlledScalingRunner(ds,device='cpu',precision='fp32',seed=13,horizons=(1,2))
    data=r.data_curve(fractions=(.5,1.0),sequence_length=2,hidden=12,epochs=1,batch_size=16)
    ctx=r.context_curve(specs=((2,2,1),(3,4,2)),hidden=12,epochs=1,batch_size=16)
    assert data[0].train_sequences<data[1].train_sequences
    assert [x.context['sequence_length'] for x in ctx]==[2,3]
    assert all(x.axis=='context' for x in ctx)


def test_resumable_full_scaling_experiment_writes_scorecard(tmp_path):
    path,_=_dataset(tmp_path)
    cfg={
        'seed':14,
        'runtime':{'device':'cpu','precision':'fp32'},
        'model':{'hidden':12,'components':2,'horizons':[1,2],'experts':3,'top_k':1},
        'training':{'epochs':1,'batch_size':16,'sequence_length':2,'lr':7e-4},
        'data':{'fractions':[.5,1.0],'model_kind':'monolithic'},
        'scaling':{'hidden_sizes':[12]},
        'context':{'specs':[[2,2,1],[3,4,2]],'model_kind':'monolithic'},
    }
    report,state=run_controlled_scaling_experiment(path,tmp_path/'run',cfg)
    assert report.format=='awa-v2.12-controlled-scaling-v1'
    assert len(report.points)==2+2+2+2
    assert all(v['status']=='completed' for v in state['phases'].values())
    saved=json.loads((tmp_path/'run'/'scaling_report.json').read_text())
    assert len(saved['points'])==len(report.points)
    # exact resume: no duplicate points and same immutable config hash
    report2,state2=run_controlled_scaling_experiment(path,tmp_path/'run',cfg)
    assert len(report2.points)==len(report.points)
    assert state2['config_hash']==state['config_hash']
