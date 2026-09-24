import torch
import numpy as np

from awa.v2.temporal import LocalGlobalTemporalCore
from awa.v2.world import MultimodalWorldModel, MonotoneHorizonUncertainty, RiskConstraintModel
from awa.v2.voc import ValueOfComputation
from awa.v2.memory import LongContextBuffer, EpisodicStore, EpisodicItem
from awa.v2.hierarchy import Subgoal, SubgoalPlanner
from awa.v2.agent import AetherV2Agent


def test_local_global_temporal_core_shapes_and_updates():
    core=LocalGlobalTemporalCore(16,2,local_dim=16,global_dim=16,event_slots=4,global_stride=2,heads=4)
    st=core.initial(3,torch.device('cpu'))
    st2=core(st,torch.randn(3,16),torch.randn(3,2),0)
    assert st2.vector.shape==(3,32)
    assert st2.event_buffer.shape==(3,4,16)
    assert not torch.equal(st.global_ctx,st2.global_ctx)


def test_multimodal_world_distribution_and_sample_shapes():
    m=MultimodalWorldModel(24,3,hidden=32,components=4)
    b=torch.randn(5,24); a=torch.randn(5,3)
    d,_=m.distribution(b,a)
    assert d.means.shape==(5,4,24)
    assert d.scales.min().item()>0
    assert d.sample(7).shape==(5,7,24)
    assert torch.allclose(d.probs.sum(-1),torch.ones(5),atol=1e-5)


def test_world_nll_is_finite_and_differentiable():
    m=MultimodalWorldModel(12,2,hidden=32,components=3)
    b=torch.randn(8,12); a=torch.randn(8,2); nxt=torch.randn(8,12)
    loss=m.nll_loss(b,a,nxt)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None for p in m.parameters())


def test_horizon_uncertainty_monotone():
    u=MonotoneHorizonUncertainty(10,horizons=(1,5,20),hidden=16)
    vals=u(torch.randn(6,10))
    assert torch.all(vals[:,1] >= vals[:,0])
    assert torch.all(vals[:,2] >= vals[:,1])


def test_risk_is_distinct_probability_signal():
    r=RiskConstraintModel(10,2,hidden=16,constraints=3)
    p=r.probabilities(torch.randn(4,10),torch.randn(4,2))
    assert p.shape==(4,3)
    assert torch.all((p>=0)&(p<=1))


def test_value_of_computation_selects_best_net_value():
    voc=ValueOfComputation(4,[('actor',0),('mppi',32),('icem',64)],hidden=8)
    with torch.no_grad():
        for p in voc.net.parameters(): p.zero_()
        voc.net[-1].bias.copy_(torch.tensor([0.0,2.0,1.0]))
        voc.log_cost_scale.fill_(-10.0)
    choice,_=voc.choose(torch.zeros(1,4))
    assert choice[0].planner=='mppi'
    assert choice[0].budget==32


def test_long_context_is_bounded():
    mem=LongContextBuffer(max_events=3)
    for i in range(5): mem.append(np.array([i,0],dtype=np.float32))
    assert len(mem)==3
    assert mem.recent_latents(3)[0,0]==2


def test_episodic_goal_conditioned_retrieval():
    store=EpisodicStore(10)
    store.add(EpisodicItem(np.array([1.,0.]),np.array([1.,0.]),1.0,{'id':'a'}))
    store.add(EpisodicItem(np.array([0.,1.]),np.array([0.,1.]),1.0,{'id':'b'}))
    hit=store.query(np.array([1.,0.]),np.array([1.,0.]),k=1)[0][1]
    assert hit.metadata['id']=='a'


def test_subgoal_planner_penalizes_risky_unreachable_shortcut():
    goals=[
        Subgoal('unsafe_shortcut',frozenset({'start'}),frozenset({'goal'}),cost=.1,reachability=.2,risk=.9),
        Subgoal('step1',frozenset({'start'}),frozenset({'mid'}),cost=1.,reachability=1.,risk=0.),
        Subgoal('step2',frozenset({'mid'}),frozenset({'goal'}),cost=1.,reachability=1.,risk=0.),
    ]
    plan,cost=SubgoalPlanner(goals).plan({'start'},{'goal'})
    assert [x.name for x in plan]==['step1','step2']


def test_v2_agent_end_to_end_shapes():
    agent=AetherV2Agent(12,2,'continuous',latent_dim=16,local_dim=16,global_dim=16,hidden=32,low=[-1,-1],high=[1,1])
    st=agent.initial_state(2,torch.device('cpu'))
    st,b=agent.observe(st,torch.randn(2,12),torch.zeros(2,2),0)
    assert b.shape==(2,32)
    a=agent.actor.deterministic_action(b)
    out=agent.world.imagine_step(b,a,deterministic=True)
    assert out['belief'].shape==b.shape
    assert out['risk'].shape==(2,)

from awa.v2.decision import InformationValue

def test_information_value_rewards_information_and_penalizes_risk():
    a=InformationValue.utility(1.0,1.0,0.1,0.1,beta=.5)
    b=InformationValue.utility(1.0,0.0,0.4,0.1,beta=.5)
    assert a>b

from awa.v2.decision import SelectiveDecisionController
from awa.v2.voc import ComputeChoice
from awa.v2.interfaces import PlanResult

class _Actor:
    def deterministic_action(self,b): return torch.zeros(b.shape[0],2)
class _U:
    def at_horizon(self,b,h): return torch.zeros(b.shape[0])
class _R:
    def aggregate_risk(self,b,a): return torch.zeros(b.shape[0])
class _World:
    uncertainty=_U(); risk=_R()
class _VOC:
    def __init__(self,positive): self.positive=positive
    def choose(self,f,c=None):
        ch=ComputeChoice('mppi' if self.positive else 'actor',8,1.0,0.1,0.9 if self.positive else -0.1)
        return [ch],torch.tensor([[ch.voc]])
class _Planner:
    def plan(self,b,actor=None,budget=None): return PlanResult(torch.ones(1,2),2.0,'mppi',10,{})

def test_selective_controller_skips_search_when_voc_negative():
    c=SelectiveDecisionController(_Actor(),_World(),_VOC(False),{'mppi':_Planner()})
    d=c.act(torch.zeros(1,4),torch.zeros(1,3))
    assert d.source=='actor' and d.budget==0

def test_selective_controller_uses_search_when_voc_positive_and_safe():
    c=SelectiveDecisionController(_Actor(),_World(),_VOC(True),{'mppi':_Planner()})
    d=c.act(torch.zeros(1,4),torch.zeros(1,3))
    assert d.source=='mppi' and d.budget==8
