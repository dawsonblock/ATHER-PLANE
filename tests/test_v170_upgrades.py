import numpy as np
import torch

from awa.planning.uncertainty import UncertaintyEnsemble
from awa.training.continuous_critic import TwinQCritic, TargetTwinQCritic, RunningTargetScale
from awa.dynamics.stateful_sequence import CachedSequenceAdapter
from awa.training.trainer import build_components, train, evaluate
from awa.config import validate_config
from awa.planning.auto_calibrate import collect_transition_uncertainty, fit_calibrator_from_errors
from awa.evaluation.planning_benefit import collect_one_step_planning_benefit, fit_arbitrator_dataset
from awa.evaluation.env_qualification import qualify_environment


def continuous_cfg(gate=False, calibrated=True, q=True):
    return validate_config({
        'seed':5,'device':'cpu',
        'env':{'kind':'continuous_point','action_type':'continuous','horizon':10,'obs_dim':4,'action_dim':2,'action_low':[-1,-1],'action_high':[1,1]},
        'model':{'obs_dim':4,'latent_dim':8,'deterministic_dim':16,'stochastic_dim':4,'hidden_dim':24,'action_dim':2,'dynamics':'gru','slow_enabled':False,'uncertainty_heads':3},
        'planner':{'enabled':True,'horizon':2,'candidates':8,'elites':2,'iterations':1,'min_std':.05,'uncertainty_limit':1.5,'calibrated_uncertainty':calibrated,'learned_gate':gate},
        'training':{'steps':38,'batch_size':2,'sequence_length':3,'burn_in':0,'overshoot_horizon':2,'overshoot_coef':.05,'replay_capacity':500,'lr':3e-4,'actor_lr':3e-4,'value_lr':3e-4,'grad_clip':10.,'free_nats':.1,'imagination_horizon':3,'discount':.99,'lambda':.95,'entropy_coef':.001,'checkpoint_every':1000,'log_every':20,'auto_calibrate_uncertainty':False,
                    'continuous_q':{'enabled':q,'hidden_dim':24,'lr':3e-4,'target_tau':.02,'actor_coef':.1}},
    })


def test_action_conditioned_uncertainty_changes_input():
    u=UncertaintyEnsemble(6,4,heads=3,hidden_dim=12,action_dim=2)
    b=torch.randn(5,6); a0=torch.zeros(5,2); a1=torch.ones(5,2)
    p0,_=u(b,a0); p1,_=u(b,a1)
    assert p0.shape==(5,4) and not torch.allclose(p0,p1)


def test_twin_q_critic_and_target_scale():
    q=TwinQCritic(10,2,32,[-1,-2],[1,2]); s=torch.randn(8,10); a=torch.randn(8,2).clamp(-1,1)
    q1,q2=q(s,a); target=torch.randn(8); scale=RunningTargetScale(); scale.update(target)
    loss=scale.scaled_huber(q1,target)+scale.scaled_huber(q2,target); loss.backward()
    assert q1.shape==(8,) and q2.shape==(8,) and scale.std.item()>0
    tq=TargetTwinQCritic(q); before=[p.clone() for p in tq.net.parameters()]; tq.update(q,.5)
    assert len(before)==len(list(tq.net.parameters()))


def test_continuous_q_build_and_training(tmp_path):
    cfg=continuous_cfg(q=True); c=build_components(cfg,torch.device('cpu'))
    assert c.q_critic is not None and c.target_q is not None and c.q_scale is not None
    c=train(cfg,torch.device('cpu'),tmp_path); r=evaluate(cfg,c,torch.device('cpu'),episodes=2)
    assert np.isfinite(r['mean_reward'])


def test_cached_sequence_adapter_matches_recompute_last_token():
    backend=torch.nn.GRU(input_size=3,hidden_size=3,batch_first=True)
    class Wrap(torch.nn.Module):
        def __init__(self): super().__init__(); self.g=backend
        def forward(self,x): return self.g(x)[0]
    w=CachedSequenceAdapter(Wrap(),max_context=4); cache=None; xs=[]
    for _ in range(5):
        t=torch.randn(2,3); xs.append(t); y,cache=w.step(t,cache)
        ctx=torch.stack(xs[-4:],dim=1); expected=w.backend(ctx)[:,-1]
        assert torch.allclose(y,expected,atol=1e-6)
    assert cache.context.shape[1]==4 and cache.tokens_seen==5


def test_auto_calibration_collection_and_fit():
    cfg=continuous_cfg(calibrated=True,q=False); c=build_components(cfg,torch.device('cpu'))
    u,e=collect_transition_uncertainty(cfg,c,torch.device('cpu'),episodes=2,max_samples=12)
    assert len(u)==len(e)>0 and np.isfinite(u).all() and np.isfinite(e).all()
    report=fit_calibrator_from_errors(c.calibrator,u,e,steps=20,lr=.05)
    assert report.samples==len(u) and 0<=report.failure_rate<=1


def test_planning_benefit_collection_and_gate_fit():
    cfg=continuous_cfg(gate=True,calibrated=True,q=False); c=build_components(cfg,torch.device('cpu'))
    data=collect_one_step_planning_benefit(cfg,c,torch.device('cpu'),episodes=1,max_samples=6)
    assert len(data)>0 and data.diagnostics.shape[1]==5
    report=fit_arbitrator_dataset(c.arbitrator,data,epochs=3,lr=.01)
    assert report['loss']>=0


def test_environment_qualification_continuous_point():
    r=qualify_environment(continuous_cfg(),steps=5)
    assert r['finite'] and r['action_type']=='continuous' and r['obs_dim']==4
