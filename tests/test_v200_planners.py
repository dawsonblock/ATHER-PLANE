import torch
from torch import nn
from awa.v2.planners import PolicySeededMPPI, PolicySeededICEM, GradientPlanner
from awa.v2.arena import compare_planners

class ZeroActor:
    def deterministic_action(self,b): return torch.zeros(b.shape[0],1,device=b.device)

class QuadraticWorld(nn.Module):
    # belief is 1D position; actions move position; target +1
    def __init__(self):
        super().__init__(); self.dummy=nn.Parameter(torch.tensor(0.0))
    def imagine_step(self,b,a,deterministic=False):
        nxt=b+a
        reward=-(nxt-1.0).pow(2)
        value=-(nxt-1.0).pow(2)
        return {'belief':nxt,'reward':reward,'value':value,'continuation':torch.ones_like(reward)*0.99,'risk':torch.zeros(b.shape[0],device=b.device)}

def test_policy_seeded_mppi_returns_bounded_action_and_calls():
    w=QuadraticWorld(); p=PolicySeededMPPI(w,ZeroActor(),[-1],[1],horizon=3,candidates=128,noise_std=.5)
    r=p.plan(torch.zeros(1,1),budget=64)
    assert r.action.shape==(1,1)
    assert -1 <= r.action.item() <= 1
    assert r.world_model_calls==64*3


def test_icem_improves_over_zero_actor_on_quadratic_world():
    torch.manual_seed(2)
    w=QuadraticWorld(); p=PolicySeededICEM(w,ZeroActor(),[-1],[1],horizon=2,candidates=128,elites=16,iterations=4)
    r=p.plan(torch.zeros(1,1),budget=128)
    assert r.action.item() > 0.15
    assert r.score > -2.0


def test_gradient_planner_moves_toward_target():
    w=QuadraticWorld(); p=GradientPlanner(w,[-1],[1],horizon=2,steps=12,lr=.15)
    r=p.plan(torch.zeros(1,1),actor=ZeroActor(),budget=12)
    assert r.action.item() > 0.1


def test_planner_arena_records_equal_budget():
    w=QuadraticWorld(); actor=ZeroActor()
    planners=[PolicySeededMPPI(w,actor,[-1],[1],2,32),PolicySeededICEM(w,actor,[-1],[1],2,32,8,2)]
    rows=compare_planners(planners,torch.zeros(1,1),budgets=(64,),actor=actor,equal_world_model_calls=True)
    assert len(rows)==2
    assert all(r.requested_world_model_calls==64 for r in rows)
    assert all(r.world_model_calls <= 64 for r in rows)
    assert all(r.world_model_calls >= 32 for r in rows)
    assert all(r.latency_ms>=0 for r in rows)

from awa.v2.planners import BeamPlanner

class DiscreteWorld(nn.Module):
    def imagine_step(self,b,a,deterministic=False):
        # action 1 increases state toward target 2; action 0 decreases.
        delta=(a[:,1:2]-a[:,0:1])
        nxt=b+delta
        reward=-(nxt-2).abs()
        return {'belief':nxt,'reward':reward,'value':reward,'continuation':torch.ones_like(reward),'risk':torch.zeros(b.shape[0])}

def test_beam_planner_discrete_prefers_positive_action():
    p=BeamPlanner(DiscreteWorld(),2,horizon=2,beam_width=4)
    r=p.plan(torch.zeros(1,1),budget=4)
    assert int(r.action.argmax(-1).item())==1


def test_gradient_planner_does_not_accumulate_world_parameter_grads():
    w=QuadraticWorld(); p=GradientPlanner(w,[-1],[1],horizon=2,steps=3,lr=.1)
    p.plan(torch.zeros(1,1),actor=ZeroActor(),budget=3)
    assert w.dummy.grad is None
