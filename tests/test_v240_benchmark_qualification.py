from pathlib import Path
import numpy as np
import torch
from torch import nn

from awa.environments.continuous_point import ContinuousPointEnv
from awa.v2.actor_baseline import OfflineActorCriticBaseline
from awa.v2.benchmark_qualification import (
    qualify_closed_loop_controllers, bootstrap_paired_return_ci,
)
from awa.v2.checkpoints import (
    load_world_checkpoint, save_actor_checkpoint, load_actor_checkpoint,
    assert_identity_encoder_compatible,
)
from awa.v2.planner_qualification import PointOracleWorld, PointHeuristicActor, identity_encoder
from awa.v2.planners import CEMPlanner, MPPIPlanner, PolicySeededMPPI
from awa.v2.representation import IdentityBackbone, module_fingerprint, build_pretrained_representation
from awa.v2.world import MultimodalWorldModel


def test_identity_representation_is_stable_and_non_normalizing():
    enc = build_pretrained_representation('identity', latent_dim=4, feature_dim=4)
    x = torch.tensor([[1., 2., 3., 4.]])
    features = enc.encode_features(x)
    assert torch.allclose(features, x)
    assert module_fingerprint(enc.backbone) == module_fingerprint(IdentityBackbone(4))


def test_world_checkpoint_roundtrip_preserves_predictions(tmp_path: Path):
    torch.manual_seed(7)
    feature_dim=4; latent_dim=6; action_dim=2; hidden=16; components=3; horizons=(1,2,4)
    projector=nn.Sequential(nn.Linear(feature_dim,latent_dim),nn.LayerNorm(latent_dim))
    world=MultimodalWorldModel(latent_dim,action_dim,hidden=hidden,components=components,horizons=horizons)
    path=tmp_path/'world.pt'
    torch.save({
        'format':'awa-v2.2-world-checkpoint-v1','dataset_sha256':'abc',
        'encoder_fingerprint':module_fingerprint(IdentityBackbone(feature_dim)),
        'model_state':world.state_dict(),'projector_state':projector.state_dict(),
        'feature_dim':feature_dim,'latent_dim':latent_dim,'action_dim':action_dim,
        'hidden':hidden,'components':components,'horizons':horizons,
    },path)
    runtime=load_world_checkpoint(path)
    assert_identity_encoder_compatible(runtime)
    obs=torch.tensor([[.1,.2,.3,.4]])
    belief=projector(obs); loaded=runtime.encode_observation(obs)
    assert torch.allclose(belief,loaded,atol=1e-6)
    action=torch.tensor([[.2,-.1]])
    a=world.imagine_step(belief,action,deterministic=True)
    b=runtime.world.imagine_step(loaded,action,deterministic=True)
    assert torch.allclose(a['belief'],b['belief'],atol=1e-6)
    assert torch.allclose(a['reward'],b['reward'],atol=1e-6)


def test_actor_checkpoint_roundtrip(tmp_path: Path):
    torch.manual_seed(11)
    tr=OfflineActorCriticBaseline(4,2,[-1,-1],[1,1],hidden=32,lr=1e-3)
    probe=torch.randn(5,4)
    before=tr.actor.deterministic_action(probe)
    path=save_actor_checkpoint(tr,tmp_path/'actor.pt',world_checkpoint_sha256='deadbeef',dataset_sha256='cafe')
    loaded=load_actor_checkpoint(path)
    after=loaded.actor.deterministic_action(probe)
    assert loaded.state_dim==4 and loaded.action_dim==2 and loaded.hidden==32
    assert torch.allclose(before,after,atol=1e-7)


def test_closed_loop_multiseed_report_contains_paired_actor_deltas():
    world=PointOracleWorld(); actor=PointHeuristicActor(.55); low=[-1,-1]; high=[1,1]
    specs=[
        ('cem',CEMPlanner(world,low,high,horizon=4,candidates=16,elites=4,iterations=2),64),
        ('mppi',MPPIPlanner(world,low,high,horizon=4,candidates=16),64),
        ('policy_mppi',PolicySeededMPPI(world,actor,low,high,horizon=4,candidates=16),64),
    ]
    report,rows=qualify_closed_loop_controllers(lambda seed:ContinuousPointEnv(seed=seed),identity_encoder,actor,specs,
                                                  seeds=[1,2,3],max_steps=20)
    assert len(rows)==12
    assert set(report.controllers)=={'actor','cem','mppi','policy_mppi'}
    assert set(report.paired)=={'cem','mppi','policy_mppi'}
    assert report.controllers['actor'].world_model_calls_per_decision==0.0
    assert report.controllers['cem'].world_model_calls_per_decision>0
    assert 0.0 <= report.controllers['policy_mppi'].success_rate <= 1.0


def test_equal_requested_budget_bounds_closed_loop_world_calls_per_decision():
    world=PointOracleWorld(); actor=PointHeuristicActor(.55); low=[-1,-1]; high=[1,1]
    planner=CEMPlanner(world,low,high,horizon=4,candidates=16,elites=4,iterations=2)
    report,_=qualify_closed_loop_controllers(lambda seed:ContinuousPointEnv(seed=seed),identity_encoder,actor,
                                              [('cem',planner,64)],seeds=[0],max_steps=5)
    # CEM backend budget is translated so each decision uses no more than the requested call envelope.
    assert report.controllers['cem'].world_model_calls_per_decision <= 64.0
    assert report.controllers['cem'].world_model_calls_per_decision > 0.0


def test_bootstrap_paired_ci_is_deterministic_and_centered_on_gain():
    actor=[1.,2.,3.,4.,5.]; candidate=[2.,3.,4.,5.,6.]
    mean,lo,hi=bootstrap_paired_return_ci(actor,candidate,samples=300,seed=9)
    assert abs(mean-1.0)<1e-9
    assert lo<=mean<=hi
    assert lo==hi==1.0


def test_loader_infers_legacy_nondefault_risk_geometry(tmp_path: Path):
    from awa.v2.world import RiskConstraintModel
    feature_dim=4; latent_dim=6; action_dim=2; hidden=16; components=2; horizons=(1,3)
    projector=nn.Sequential(nn.Linear(feature_dim,latent_dim),nn.LayerNorm(latent_dim))
    world=MultimodalWorldModel(latent_dim,action_dim,hidden=hidden,components=components,horizons=horizons)
    world.risk=RiskConstraintModel(latent_dim,action_dim,hidden=11,constraints=3)
    path=tmp_path/'legacy.pt'
    torch.save({
        'format':'awa-v2.2-world-checkpoint-v1','dataset_sha256':'legacy',
        'encoder_fingerprint':module_fingerprint(IdentityBackbone(feature_dim)),
        'model_state':world.state_dict(),'projector_state':projector.state_dict(),
        'feature_dim':feature_dim,'latent_dim':latent_dim,'action_dim':action_dim,
        'hidden':hidden,'components':components,'horizons':horizons,
    },path)
    runtime=load_world_checkpoint(path)
    assert runtime.metadata.risk_hidden==11
    assert runtime.metadata.risk_constraints==3
    assert runtime.world.risk.net[0].out_features==11
