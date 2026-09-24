import copy
from pathlib import Path
import torch

from awa.dynamics.stateful_sequence import CachedSequenceAdapter, SequenceCache
from awa.training.continuous_critic import QuantileTwinQCritic, TargetTwinQCritic, quantile_huber_loss
from awa.config import load_config, validate_config, ConfigError
from awa.training.trainer import build_components
from awa.evaluation.planning_benefit import collect_planning_benefit
from awa.evaluation.env_qualification import qualify_environment_multiseed


class NativeToy(torch.nn.Module):
    def forward(self, sequence):
        return torch.cumsum(sequence, dim=1)
    def initial_cache(self, batch_size=None, device=None, dtype=None):
        return torch.zeros(batch_size or 1, 3, device=device, dtype=dtype or torch.float32)
    def step_cached(self, token, state):
        x=token[:,0]
        state=state+x
        return state,state


def test_native_sequence_cache_contract():
    a=CachedSequenceAdapter(NativeToy(),max_context=2,prefer_native=True)
    cache=a.initial_cache(batch_size=1,device=torch.device('cpu'),dtype=torch.float32)
    out,cache=a.step(torch.ones(1,3),cache)
    out,cache=a.step(torch.full((1,3),2.0),cache)
    assert torch.allclose(out,torch.full((1,3),3.0))
    assert cache.tokens_seen==2
    assert cache.metadata['backend']=='native'
    assert cache.context is None


def test_quantile_twin_q_shapes_and_target_copy():
    q=QuantileTwinQCritic(12,2,hidden_dim=24,quantiles=8,low=[-1,-1],high=[1,1])
    b=torch.randn(5,12); a=torch.randn(5,2).clamp(-1,1)
    q1,q2=q(b,a)
    assert q1.shape==(5,8) and q2.shape==(5,8)
    assert q.minimum(b,a).shape==(5,)
    tq=TargetTwinQCritic(q)
    assert tq.conservative_quantiles(b,a).shape==(5,8)


def test_quantile_huber_is_finite_and_differentiable():
    pred=torch.randn(4,8,requires_grad=True); target=torch.randn(4,8)
    taus=(torch.arange(8,dtype=torch.float32)+0.5)/8
    loss=quantile_huber_loss(pred,target,taus)
    loss.backward()
    assert torch.isfinite(loss)
    assert pred.grad is not None and torch.isfinite(pred.grad).all()


def test_v18_config_builds_distributional_critic():
    root=Path(__file__).parents[1]
    cfg=load_config(root/'configs'/'v1_8_continuous.yaml')
    comps=build_components(cfg,torch.device('cpu'))
    assert getattr(comps.q_critic,'distributional',False)
    assert comps.q_critic.num_quantiles==32
    assert comps.arbitrator is not None


def test_invalid_quantile_config_rejected():
    root=Path(__file__).parents[1]
    cfg=load_config(root/'configs'/'v1_7_continuous.yaml')
    bad=copy.deepcopy(cfg)
    bad['training']['continuous_q']['critic_type']='quantile_twin_q'
    bad['training']['continuous_q']['quantiles']=2
    try:
        validate_config(bad)
        assert False, 'expected ConfigError'
    except ConfigError:
        pass


def _small_planning_cfg():
    root=Path(__file__).parents[1]
    cfg=load_config(root/'configs'/'v1_8_continuous.yaml')
    cfg['device']='cpu'
    cfg['model'].update({'latent_dim':8,'deterministic_dim':16,'stochastic_dim':4,'hidden_dim':24,'slow_enabled':False,'uncertainty_heads':2})
    cfg['planner'].update({'horizon':2,'candidates':4,'elites':2,'iterations':1,'adaptive':False})
    cfg['training']['continuous_q']['hidden_dim']=24
    cfg['training']['continuous_q']['quantiles']=8
    return cfg


def test_multistep_planning_benefit_collection():
    cfg=_small_planning_cfg(); comps=build_components(cfg,torch.device('cpu'))
    data=collect_planning_benefit(cfg,comps,torch.device('cpu'),episodes=1,max_samples=3,branch_horizon=2)
    assert len(data)>0
    assert data.branch_horizon==2
    assert data.diagnostics.shape[1]==5
    assert torch.isfinite(data.planner_return).all()
    assert torch.isfinite(data.actor_return).all()


def test_multiseed_environment_qualification_reports_all_seeds():
    cfg=_small_planning_cfg()
    report=qualify_environment_multiseed(cfg,seeds=[2,3,4],steps=4)
    assert report['passed']==3
    assert report['failed']==0
    assert len(report['reports'])==3
    assert all(r['finite'] for r in report['reports'])
