import numpy as np
import torch
from awa.actions import ActionSpec, action_spec_from_config
from awa.planning.actor import TanhGaussianActor
from awa.planning.mpc import ContinuousCEMPlanner
from awa.environments.continuous_point import ContinuousPointEnv
from awa.memory.replay import EpisodeReplayBuffer, Transition
from awa.model import WorldModel
from awa.training.trainer import build_components, train, evaluate
from awa.config import validate_config


def tiny_cfg():
    return validate_config({
        "seed": 3, "device": "cpu",
        "env": {"kind":"continuous_point","action_type":"continuous","horizon":10,"obs_dim":4,"action_dim":2,
                "action_low":[-1,-1],"action_high":[1,1]},
        "model": {"obs_dim":4,"latent_dim":8,"deterministic_dim":16,"stochastic_dim":4,"dynamics":"gru","action_dim":2,
                  "hidden_dim":24,"slow_enabled":False,"uncertainty_heads":2},
        "planner": {"enabled":True,"horizon":3,"candidates":12,"elites":4,"iterations":2,"min_std":0.05,"uncertainty_limit":1.5},
        "training": {"steps":35,"batch_size":2,"sequence_length":3,"burn_in":0,"overshoot_horizon":2,"overshoot_coef":0.05,
                     "replay_capacity":500,"prioritized_replay":False,"lr":3e-4,"actor_lr":3e-4,"value_lr":3e-4,"grad_clip":10.0,
                     "free_nats":0.1,"imagination_horizon":3,"discount":0.99,"lambda":0.95,"entropy_coef":0.001,
                     "checkpoint_every":1000,"log_every":20,"epsilon_start":0.2,"epsilon_final":0.05},
        "acceptance": {"success_improvement_pct":5,"sample_efficiency_pct":15,"robustness_pct":20,"compute_reduction_pct":20,"max_critical_regression_pct":3}
    })


def test_action_spec_continuous_bounds():
    spec=ActionSpec.continuous([-2,-1],[2,3]); x=spec.random(64,torch.device('cpu'))
    assert x.shape==(64,2) and torch.all(x[:,0]>=-2) and torch.all(x[:,0]<=2)


def test_tanh_gaussian_actor_is_bounded_and_differentiable():
    actor=TanhGaussianActor(10,2,24,[-2,-1],[2,3]); state=torch.randn(5,10)
    action,logp,entropy=actor.sample_with_log_prob(state); loss=-(logp+0.01*entropy).mean(); loss.backward()
    assert action.shape==(5,2); assert torch.all(action[:,0]>=-2) and torch.all(action[:,0]<=2)
    assert all(p.grad is not None for p in actor.parameters() if p.requires_grad)


def test_continuous_cem_respects_bounds():
    wm=WorldModel(4,8,16,4,2,24,"gru"); b=wm.initial_belief(1,torch.device('cpu'))
    p=ContinuousCEMPlanner(wm,2,[-1,-.5],[1,.5],horizon=2,candidates=8,elites=2,iterations=1)
    model_action,env_action,_=p.plan(b)
    assert model_action.shape==(1,2); assert env_action.shape==(2,); assert np.all(env_action >= [-1,-.5]) and np.all(env_action <= [1,.5])


def test_continuous_replay_sequence_shape():
    r=EpisodeReplayBuffer(100)
    for i in range(5):
        r.add(Transition(np.ones(4,dtype=np.float32)*i,np.array([.1,-.2],dtype=np.float32),0.1,np.ones(4,dtype=np.float32)*(i+1),i==4))
    b=r.sample_sequences(1,3,torch.device('cpu'))
    assert b.actions.shape==(1,3,2) and b.actions.dtype==torch.float32


def test_continuous_point_state_restore():
    e=ContinuousPointEnv(horizon=5,seed=9); o=e.reset(); e.step([.2,-.1]); st=e.state_dict(); a=e.step([.3,.4]); e.load_state_dict(st); b=e.step([.3,.4])
    assert np.allclose(a[0],b[0]) and a[1:3]==b[1:3]


def test_build_continuous_components():
    c=build_components(tiny_cfg(),torch.device('cpu'))
    assert c.action_spec.kind=='continuous'; assert c.actor.action_type=='continuous'; assert c.planner.action_type=='continuous'


def test_continuous_end_to_end_training(tmp_path):
    cfg=tiny_cfg(); c=train(cfg,torch.device('cpu'),tmp_path); result=evaluate(cfg,c,torch.device('cpu'),episodes=2)
    assert result['action_type']=='continuous' and np.isfinite(result['mean_reward'])


def test_invalid_continuous_bounds_rejected():
    cfg=tiny_cfg(); cfg['env']['action_low']=[0,0]; cfg['env']['action_high']=[0,1]
    import pytest
    from awa.config import ConfigError
    with pytest.raises(ConfigError): validate_config(cfg)


def test_continuous_engine_action_is_bounded():
    cfg=tiny_cfg(); c=build_components(cfg,torch.device('cpu')); b=c.world_model.initial_belief(1,torch.device('cpu'))
    model_action,env_action,meta=c.engine.act(b)
    assert model_action.shape==(1,2); assert env_action.shape==(2,)
    assert np.all(env_action>=-1) and np.all(env_action<=1); assert meta['source'] in {'actor','planner'}
