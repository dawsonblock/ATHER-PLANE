import numpy as np
import torch

from awa.environments.continuous_point import ContinuousPointEnv
from awa.v2.actor_baseline import LatentTransitionDataset, OfflineActorCriticBaseline
from awa.v2.arena import backend_budget_for_calls, compare_planners
from awa.v2.branching import capture_snapshot, restore_snapshot, verify_snapshot_roundtrip, collect_planner_benefits
from awa.v2.planner_qualification import PointOracleWorld, PointHeuristicActor, identity_encoder
from awa.v2.planners import CEMPlanner, MPPIPlanner, PolicySeededMPPI, PolicySeededICEM, GradientPlanner
from awa.v2.safety import SafeActionGuard
from awa.v2.voc import ValueOfComputation, fit_value_of_computation
from awa.v2.world import RiskConstraintModel


def stack():
    w=PointOracleWorld(); a=PointHeuristicActor(.55); low=[-1,-1]; high=[1,1]
    return w,a,[
        CEMPlanner(w,low,high,horizon=4,candidates=16,elites=4,iterations=2),
        MPPIPlanner(w,low,high,horizon=4,candidates=16),
        PolicySeededMPPI(w,a,low,high,horizon=4,candidates=16),
        PolicySeededICEM(w,a,low,high,horizon=4,candidates=16,elites=4,iterations=2),
        GradientPlanner(w,low,high,horizon=4,steps=4,lr=.1),
    ]


def test_vanilla_planners_are_distinct_from_policy_seeded_controls():
    w,a,planners=stack(); b=torch.tensor([[0.,0.,.8,.2]])
    torch.manual_seed(3)
    results=[p.plan(b,actor=a,budget=8) for p in planners[:4]]
    names=[r.planner for r in results]
    assert names==['cem','mppi','policy_mppi','policy_icem']
    assert results[0].metadata['policy_seeded'] is False
    assert results[1].metadata['policy_seeded'] is False
    assert results[2].metadata['policy_seeded'] is True
    assert results[3].metadata['policy_seeded'] is True


def test_equal_world_call_budget_translation_keeps_backends_bounded():
    w,a,planners=stack(); b=torch.tensor([[0.,0.,.8,.2]])
    rows=compare_planners(planners,b,budgets=(64,),actor=a,equal_world_model_calls=True)
    assert len(rows)==5
    assert all(r.requested_world_model_calls==64 for r in rows)
    assert all(r.world_model_calls<=64 for r in rows)
    assert all(r.world_model_calls>0 for r in rows)


def test_exact_snapshot_roundtrip_restores_environment():
    env=ContinuousPointEnv(seed=12); env.reset(); snap=capture_snapshot(env)
    assert verify_snapshot_roundtrip(env,np.array([.2,-.1],dtype=np.float32))
    env.step(np.array([1,1],dtype=np.float32)); restore_snapshot(env,snap)
    s=env.state_dict(); assert np.allclose(s['position'],snap['position']); assert s['t']==snap['t']


def test_real_branch_collection_builds_complete_voc_matrix():
    w,a,planners=stack(); chosen=[]
    for p in planners[:3]: chosen.append((p,backend_budget_for_calls(p,48)))
    table=collect_planner_benefits(lambda seed:ContinuousPointEnv(seed=seed),identity_encoder,w,a,chosen,
                                   episodes=2,max_states=6,branch_horizon=2,seed=4)
    x,g,c,keys=table.matrix()
    assert x.shape==(6,6)
    assert g.shape==(6,3) and c.shape==(6,3) and len(keys)==6
    assert torch.isfinite(g).all() and torch.isfinite(c).all()
    assert len(table.summary())==3


def test_voc_supervised_fit_learns_cost_adjusted_choice():
    torch.manual_seed(5)
    x=torch.randn(96,6)
    # choice 0 wins for negative x0, choice 1 for positive x0; choice 2 costs too much.
    gains=torch.stack([-x[:,0],x[:,0]+.2,torch.ones_like(x[:,0])*.5],-1)
    costs=torch.tensor([0.,.05,1.0]).view(1,-1).expand_as(gains)
    model=ValueOfComputation(6,[('actor',0),('mppi',32),('icem',128)],hidden=64)
    rep=fit_value_of_computation(model,x,gains,costs,epochs=300,lr=5e-3,cost_weight=.75)
    assert rep.choice_accuracy>.85
    assert rep.gain_mae<.2


def test_fail_closed_guard_replaces_high_risk_action():
    risk=RiskConstraintModel(4,2,hidden=8,constraints=1)
    with torch.no_grad():
        for p in risk.parameters(): p.zero_()
        risk.net[-1].bias.fill_(10.)
    guard=SafeActionGuard(risk,[-1,-1],[1,1],risk_limit=.5,recovery_action=[0,0])
    out=guard.validate(torch.zeros(1,4),torch.ones(1,2))
    assert out.accepted is False
    assert torch.allclose(out.action,torch.zeros_like(out.action))
    assert out.risk>.5


def test_offline_actor_baseline_trains_and_preserves_action_bounds():
    rng=np.random.default_rng(8); n=64
    s=rng.uniform(-1,1,size=(n,4)).astype(np.float32)
    a=np.clip((s[:,2:]-s[:,:2])/.4,-1,1).astype(np.float32)
    ns=s.copy(); ns[:,:2]+=0.15*a
    r=(np.linalg.norm(s[:,2:]-s[:,:2],axis=1)-np.linalg.norm(ns[:,2:]-ns[:,:2],axis=1)).astype(np.float32)
    d=np.zeros(n,dtype=np.float32)
    ds=LatentTransitionDataset(s,a,r,ns,d)
    trainer=OfflineActorCriticBaseline(4,2,[-1,-1],[1,1],hidden=32,lr=1e-3)
    rep=trainer.fit(ds,epochs=3,batch_size=32)
    out=trainer.action(s[:4])
    assert np.isfinite(rep.actor_loss) and np.isfinite(rep.critic_loss)
    assert out.shape==(4,2)
    assert torch.all(out<=1.00001) and torch.all(out>=-1.00001)


def test_gradient_planner_works_inside_no_grad_collection_context():
    w,a,planners=stack(); p=planners[-1]
    with torch.no_grad():
        r=p.plan(torch.tensor([[0.,0.,.8,.2]]),actor=a,budget=3)
    assert r.world_model_calls==12
    assert torch.isfinite(r.action).all()
