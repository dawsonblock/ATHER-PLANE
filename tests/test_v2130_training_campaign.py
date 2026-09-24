from pathlib import Path
import json
import numpy as np
import pytest

from awa.v2.campaign import (
    CampaignStage, PromotionPolicy, CheckpointRegistry, GameTrainingCampaign,
    detect_resource_profile, estimate_campaign_compute, campaign_plan,
)
from awa.v2.replay import ShardedReplayStore, mix_replay_arrays
import yaml


def _arrays(n=6, episode_offset=0):
    obs=np.arange(n*3,dtype=np.float32).reshape(n,3)
    nxt=obs+1
    dones=np.zeros(n,dtype=np.bool_); dones[-1]=True
    return {
        'observations':obs,
        'actions':np.zeros((n,2),dtype=np.float32),
        'rewards':np.arange(n,dtype=np.float32),
        'next_observations':nxt,
        'dones':dones,
        'episode_ids':np.full(n,episode_offset,dtype=np.int64),
    }


def test_resource_profiles_are_conservative_and_injected_vram_is_deterministic():
    p12=detect_resource_profile('cuda',vram_gb=12,system_ram_gb=64)
    p24=detect_resource_profile('cuda',vram_gb=24,system_ram_gb=128)
    assert p12.name=='consumer-12gb' and p12.max_batch_size==64 and p12.recommended_ensemble_members==1
    assert p24.name=='consumer-24gb' and p24.max_batch_size==128 and p24.recommended_ensemble_members==3
    assert p24.detected_system_ram_gb==128


def test_campaign_stage_and_compute_estimate():
    stages=[
        CampaignStage('a',100,(1,2),.2,4,32,2,3,1,16),
        CampaignStage('b',250,(1,2,3),.3,8,48,3,4,1,32),
    ]
    est=estimate_campaign_compute(stages,samples_per_second=1000)
    assert est.final_target_transitions==250
    assert est.approximate_world_updates>0 and est.approximate_actor_updates>0
    assert est.estimated_gpu_hours is not None and est.estimated_gpu_hours>0


def test_replay_store_rejects_schema_drift_and_materializes(tmp_path):
    store=ShardedReplayStore(tmp_path/'replay',shard_size=3)
    store.append(_arrays(7))
    assert len(store)==7 and store.verify()==[]
    out=store.materialize(tmp_path/'all.npz')
    with np.load(out,allow_pickle=False) as z:
        assert len(z['observations'])==7
    with pytest.raises(ValueError):
        store.append({'observations':np.zeros((1,3),dtype=np.float32),'actions':np.zeros((1,2),dtype=np.float32)})


def test_replay_mixer_respects_capacity_and_remaps_episode_ids():
    a=_arrays(5,0); b=_arrays(7,0)
    mixed,rep=mix_replay_arrays({'actor':a,'teacher':b},{'actor':.25,'teacher':.75},10,seed=7)
    assert rep.produced==10
    assert len(mixed['replay_source'])==10
    assert set(np.unique(mixed['replay_source']))=={0,1}
    # Each source's episode id 0 must map to a different global episode id.
    ids=mixed['episode_ids']; src=mixed['replay_source']
    assert set(ids[src==0]).isdisjoint(set(ids[src==1]))


def test_promotion_policy_rejects_nonfinite_and_regression():
    policy=PromotionPolicy(max_actor_success_regression=.1,max_planner_success_regression=.1)
    base={'actor_success_rate':.8,'planner_success_rate':.9}
    cand={'actor_success_rate':.5,'planner_success_rate':.9,'actor_mean_return':1.,'planner_mean_return':2.,'horizon_rmse':{'1':.1},'risk_brier':.1}
    ok,reasons=policy.assess(cand,base)
    assert not ok and any('actor success regressed' in x for x in reasons)
    bad=dict(cand); bad['actor_mean_return']=float('nan')
    ok,reasons=policy.assess(bad,None)
    assert not ok and any('non-finite' in x for x in reasons)


def test_checkpoint_registry_promote_verify_and_rollback(tmp_path):
    reg=CheckpointRegistry(tmp_path/'ckpt')
    for idx in (1,2):
        w=tmp_path/f'w{idx}.pt'; a=tmp_path/f'a{idx}.pt'
        w.write_bytes(f'world{idx}'.encode()); a.write_bytes(f'actor{idx}'.encode())
        reg.promote(f's{idx}',w,a,{'actor_success_rate':idx/10,'planner_success_rate':idx/10})
    assert reg.current()['stage']=='s2' and reg.verify()==[]
    rolled=reg.rollback()
    assert rolled['stage']=='s1'


def test_campaign_plan_uses_config_without_running_training():
    cfg={
        'runtime':{'device':'cuda','vram_gb':12,'system_ram_gb':64},
        'compute':{'measured_samples_per_second':1000},
        'stages':[{'name':'debug','target_transitions':100,'curriculum_stages':[1,2],
                   'sequence_length':4,'hidden':16,'world_epochs':1,'actor_epochs':1,
                   'calibration_epochs':0,'batch_size':16}],
    }
    plan=campaign_plan(cfg)
    assert plan['resource_profile']['name']=='consumer-12gb'
    assert plan['compute_estimate']['final_target_transitions']==100
    assert plan['compute_estimate']['estimated_gpu_hours'] is not None


def test_tiny_campaign_runs_and_resumes(tmp_path):
    cfg={
        'seed':213,
        'runtime':{'device':'cpu','precision':'fp32','strict_resources':False},
        'replay':{'shard_size':20},
        'promotion':{'max_actor_success_regression':1.0,'max_planner_success_regression':1.0,'max_risk_brier':1.0},
        'stop_on_failed_promotion':True,
        'stages':[
            {'name':'tiny-a','target_transitions':30,'curriculum_stages':[1], 'difficulty':.1,
             'sequence_length':1,'hidden':16,'world_epochs':1,'actor_epochs':1,'calibration_epochs':0,
             'batch_size':16,'episodes_per_task':1,'tasks_per_stage':1,'horizon':20,'teacher_fraction':1.0},
            {'name':'tiny-b','target_transitions':60,'curriculum_stages':[1,3], 'difficulty':.12,
             'sequence_length':1,'hidden':16,'world_epochs':1,'actor_epochs':1,'calibration_epochs':0,
             'batch_size':16,'episodes_per_task':1,'tasks_per_stage':1,'horizon':20,'teacher_fraction':1.0},
        ],
    }
    c=GameTrainingCampaign(cfg,tmp_path/'campaign')
    r1=c.run()
    assert r1['replay_transitions']>=60
    assert r1['replay_integrity_failures']==[] and r1['checkpoint_integrity_failures']==[]
    assert r1['stable_checkpoint'] is not None
    state_before=json.loads((tmp_path/'campaign'/'experiment_state.json').read_text())
    assert state_before['config']['_resolved_runtime']['name']=='cpu'
    c2=GameTrainingCampaign(cfg,tmp_path/'campaign')
    r2=c2.run()
    state_after=json.loads((tmp_path/'campaign'/'experiment_state.json').read_text())
    assert r2['replay_transitions']==r1['replay_transitions']
    assert state_before['phases']==state_after['phases']


def test_shipped_campaign_config_plans_with_comment_only_compute_section():
    cfg=yaml.safe_load(Path("configs/v2_13_training_campaign.yaml").read_text())
    plan=campaign_plan(cfg)
    assert plan["compute_estimate"]["final_target_transitions"]==1_000_000
    assert plan["compute_estimate"]["estimated_gpu_hours"] is None
