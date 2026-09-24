from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from awa.v2 import __version__ as V2_VERSION
from awa.v2.agent import AetherV2Agent
from awa.v2.planners import CEMPlanner, MPPIPlanner, PolicySeededMPPI, PolicySeededICEM, GradientPlanner
from awa.v2.arena import compare_planners
from awa.v2.voc import ValueOfComputation
from awa.v2.datasets import OfflineTransitionDataset, write_synthetic_dataset, write_synthetic_sequence_dataset, write_synthetic_game_dataset, precompute_representation_cache
from awa.v2.representation import (
    FrozenBackboneProjector,
    CachedRepresentation,
    RepresentationCache,
    module_fingerprint,
)


def _components():
    torch.manual_seed(7)
    agent = AetherV2Agent(
        12, 2, "continuous", latent_dim=32, local_dim=32, global_dim=32,
        hidden=64, low=[-1, -1], high=[1, 1]
    )
    state = agent.initial_state(1, torch.device("cpu"))
    obs = torch.randn(1, 12)
    prev = torch.zeros(1, 2)
    state, belief = agent.observe(state, obs, prev, 0)
    return agent, belief


def smoke_main():
    agent, belief = _components()
    out = agent.world.imagine_step(belief, agent.actor.deterministic_action(belief), True)
    planner = PolicySeededMPPI(agent.world, agent.actor, [-1, -1], [1, 1], horizon=4, candidates=16)
    result = planner.plan(belief, budget=8)
    print(json.dumps({
        "status": "ok",
        "version": V2_VERSION,
        "belief_dim": belief.shape[-1],
        "future_components": agent.world.components,
        "planner": result.planner,
        "planner_calls": result.world_model_calls,
        "risk": float(out["risk"].mean().detach()),
    }, indent=2))


def arena_main():
    p = argparse.ArgumentParser()
    p.add_argument("--world-calls", type=int, default=64, help="Approximate equal world-model transition-call budget per planner")
    args = p.parse_args()
    agent, belief = _components()
    planners = [
        CEMPlanner(agent.world, [-1, -1], [1, 1], 4, 16, 4, 2),
        MPPIPlanner(agent.world, [-1, -1], [1, 1], 4, 16),
        PolicySeededMPPI(agent.world, agent.actor, [-1, -1], [1, 1], 4, 16),
        PolicySeededICEM(agent.world, agent.actor, [-1, -1], [1, 1], 4, 16, 4, 2),
        GradientPlanner(agent.world, [-1, -1], [1, 1], 4, 8, .05),
    ]
    rows = compare_planners(planners, belief, budgets=(args.world_calls,), actor=agent.actor, equal_world_model_calls=True)
    print(json.dumps([r.__dict__ for r in rows], indent=2))


def voc_demo_main():
    model = ValueOfComputation(6, [("actor", 0), ("mppi", 32), ("mppi", 128), ("icem", 64)])
    features = torch.zeros(1, 6)
    choices, _ = model.choose(features)
    print(json.dumps(choices[0].__dict__, indent=2))


class _ToyFrozenBackbone(nn.Module):
    def __init__(self, obs_dim: int = 12, feature_dim: int = 24):
        super().__init__()
        self.linear = nn.Linear(obs_dim, feature_dim, bias=False)
        torch.manual_seed(91)
        nn.init.normal_(self.linear.weight, std=0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x.float())


def cache_main():
    p = argparse.ArgumentParser(description="Precompute deterministic frozen-backbone representations for an NPZ dataset")
    p.add_argument("dataset", nargs="?", default=None, help="NPZ dataset path; omit to run a synthetic cache smoke test")
    p.add_argument("--cache-dir", default=".awa-representation-cache")
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--provider", choices=("toy", "identity", "hf"), default="toy")
    p.add_argument("--feature-dim", type=int, default=None, help="required only for custom/state-vector workflows when not inferable")
    p.add_argument("--model-name-or-path", default=None, help="local Hugging Face model path for --provider hf")
    p.add_argument("--allow-download", action="store_true", help="allow Transformers to resolve non-local model files")
    args = p.parse_args()
    temporary = None
    if args.dataset is None:
        temporary = tempfile.TemporaryDirectory()
        dataset_path = write_synthetic_dataset(Path(temporary.name) / "synthetic.npz", n=24)
    else:
        dataset_path = Path(args.dataset)
    dataset = OfflineTransitionDataset(dataset_path)
    obs_dim = int(np.prod(dataset.manifest.observation_shape))
    if args.provider == "toy":
        backbone = _ToyFrozenBackbone(obs_dim, feature_dim=int(args.feature_dim or 24))
        encoder = FrozenBackboneProjector(backbone, int(args.feature_dim or 24), args.latent_dim)
    elif args.provider == "identity":
        from awa.v2.representation import build_pretrained_representation
        encoder = build_pretrained_representation("identity", args.latent_dim, feature_dim=obs_dim)
        backbone = encoder.backbone
    else:
        from awa.v2.representation import build_pretrained_representation
        if not args.model_name_or_path:
            raise ValueError("--provider hf requires --model-name-or-path")
        encoder = build_pretrained_representation(
            "hf", args.latent_dim, model_name_or_path=args.model_name_or_path,
            local_files_only=not args.allow_download,
        )
        backbone = encoder.backbone
    cache = RepresentationCache(args.cache_dir, namespace=dataset.manifest.sha256[:16])
    fp = module_fingerprint(backbone)
    index_path = Path(args.cache_dir) / dataset.manifest.sha256[:16] / "index.json"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index = precompute_representation_cache(dataset, encoder, cache, index_path=index_path)
    result = {
        "status": "ok",
        "version": V2_VERSION,
        "provider": args.provider,
        "dataset_sha256": dataset.manifest.sha256,
        "transitions": len(dataset),
        "encoder_fingerprint": fp,
        "cache": cache.stats(),
        "writes": index["writes"],
        "hits": index["hits"],
        "index": str(index_path),
    }
    print(json.dumps(result, indent=2))
    if temporary is not None:
        temporary.cleanup()


def representation_check_main():
    torch.manual_seed(4)
    backbone = _ToyFrozenBackbone(12, 24)
    encoder = FrozenBackboneProjector(backbone, 24, 16)
    x = torch.randn(4, 12)
    y = encoder(x)
    frozen = all(not p.requires_grad for p in backbone.parameters())
    trainable_projection = any(p.requires_grad for p in encoder.projection.parameters())
    print(json.dumps({
        "status": "ok",
        "version": V2_VERSION,
        "shape": list(y.shape),
        "backbone_frozen": frozen,
        "projection_trainable": trainable_projection,
        "backbone_fingerprint": module_fingerprint(backbone),
    }, indent=2))


def _synthetic_world_stack(tmpdir: str, latent_dim: int = 12, sequence_length: int = 5):
    from awa.v2.world import MultimodalWorldModel
    from awa.v2.world_training import CachedFeatureSequenceDataset

    dataset_path = write_synthetic_sequence_dataset(
        Path(tmpdir) / "world.npz", episodes=4, episode_length=8, obs_dim=12, action_dim=2
    )
    dataset = OfflineTransitionDataset(dataset_path)
    backbone = _ToyFrozenBackbone(12, feature_dim=24)
    encoder = FrozenBackboneProjector(backbone, 24, latent_dim)
    cache = RepresentationCache(Path(tmpdir) / "cache", namespace=dataset.manifest.sha256[:16])
    index_path = Path(tmpdir) / "index.json"
    index = precompute_representation_cache(dataset, encoder, cache, index_path=index_path)
    seq = CachedFeatureSequenceDataset(dataset, cache, index, sequence_length=sequence_length)
    model = MultimodalWorldModel(latent_dim, 2, hidden=64, components=3, horizons=(1, 2, 4))
    return dataset, encoder, seq, model


def world_smoke_main():
    from awa.v2.world_training import (
        OfflineWorldModelTrainer,
        evaluate_world_model_horizons,
        calibrate_horizon_uncertainty,
    )
    p = argparse.ArgumentParser(description="Train and qualify the v2.2 multimodal world model on a chain-consistent synthetic dataset")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--calibration-epochs", type=int, default=2)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        _, encoder, seq, model = _synthetic_world_stack(td)
        trainer = OfflineWorldModelTrainer(
            model, encoder.projection, lr=1e-3, overshoot_horizon=4, train_projector=False
        )
        before = evaluate_world_model_horizons(model, encoder.projection, seq, (1, 2, 4), batch_size=16)
        history = trainer.fit(seq, epochs=args.epochs, batch_size=16)
        after = evaluate_world_model_horizons(model, encoder.projection, seq, (1, 2, 4), batch_size=16)
        calibration = calibrate_horizon_uncertainty(
            model, encoder.projection, seq, epochs=args.calibration_epochs, batch_size=16, lr=2e-3
        )
        print(json.dumps({
            "status": "ok",
            "version": V2_VERSION,
            "sequences": len(seq),
            "train_steps": len(history),
            "before": before.to_dict(),
            "after": after.to_dict(),
            "calibration": calibration,
        }, indent=2))


def risk_smoke_main():
    from awa.v2.risk_training import CachedRiskDataset, RiskConstraintTrainer, evaluate_risk_model
    p = argparse.ArgumentParser(description="Train and qualify the explicit v2.2 constraint-risk model")
    p.add_argument("--epochs", type=int, default=2)
    args = p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        _, encoder, seq, model = _synthetic_world_stack(td)
        risk_ds = CachedRiskDataset(seq)
        trainer = RiskConstraintTrainer(model.risk, encoder.projection, lr=2e-3)
        before = evaluate_risk_model(model.risk, encoder.projection, risk_ds, batch_size=16)
        history = trainer.fit(risk_ds, epochs=args.epochs, batch_size=16)
        after = evaluate_risk_model(model.risk, encoder.projection, risk_ds, batch_size=16)
        print(json.dumps({
            "status": "ok",
            "version": V2_VERSION,
            "train_steps": len(history),
            "before": before.to_dict(),
            "after": after.to_dict(),
        }, indent=2))


def promotion_main():
    from awa.v2.promotion import assess_world_model_promotion
    p = argparse.ArgumentParser(description="Compare two v2.2 world-model qualification JSON reports")
    p.add_argument("baseline")
    p.add_argument("candidate")
    p.add_argument("--min-average-improvement", type=float, default=0.0)
    p.add_argument("--max-horizon-regression", type=float, default=3.0)
    p.add_argument("--max-nll-regression", type=float, default=3.0)
    args = p.parse_args()
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    candidate = json.loads(Path(args.candidate).read_text(encoding="utf-8"))
    # Accept either a raw report or the `after` section emitted by world-smoke.
    baseline = baseline.get("after", baseline)
    candidate = candidate.get("after", candidate)
    decision = assess_world_model_promotion(
        candidate, baseline,
        min_average_horizon_improvement_pct=args.min_average_improvement,
        max_horizon_regression_pct=args.max_horizon_regression,
        max_nll_regression_pct=args.max_nll_regression,
    )
    print(json.dumps(decision.to_dict(), indent=2))


def world_qualify_main():
    """Train/qualify v2.2 world dynamics from a v2.1 cached representation index."""
    from awa.v2.world import MultimodalWorldModel, RiskConstraintModel
    from awa.v2.world_training import (
        CachedFeatureSequenceDataset,
        OfflineWorldModelTrainer,
        evaluate_world_model_horizons,
        calibrate_horizon_uncertainty,
    )
    from awa.v2.representation_train import GeometryPreservingProjectionTrainer
    from awa.v2.risk_training import CachedRiskDataset, RiskConstraintTrainer, evaluate_risk_model

    p = argparse.ArgumentParser(description="Train and qualify a v2.2 multimodal world model from cached frozen features")
    p.add_argument("dataset", help="chain-consistent NPZ transition dataset")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--index", required=True, help="representation index.json produced by awa-v2-cache")
    p.add_argument("--out-dir", default="runs/v2_2_world")
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--components", type=int, default=5)
    p.add_argument("--sequence-length", type=int, default=21)
    p.add_argument("--horizons", default="1,5,10,20")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--projection-warmup", type=int, default=5)
    p.add_argument("--calibration-epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--allow-noncontiguous", action="store_true", help="disable next_obs[t]==obs[t+1] verification")
    args = p.parse_args()

    horizons = tuple(sorted({int(v) for v in args.horizons.split(",") if v.strip()}))
    if not horizons or max(horizons) > args.sequence_length:
        raise ValueError("horizons must be positive and <= sequence-length")
    dataset = OfflineTransitionDataset(args.dataset)
    namespace = dataset.manifest.sha256[:16]
    cache = RepresentationCache(args.cache_dir, namespace=namespace)
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    seq = CachedFeatureSequenceDataset(
        dataset, cache, index, args.sequence_length, verify_chain=not args.allow_noncontiguous
    )
    first = cache.get_by_key(index["observation_keys"][0])
    if first is None:
        raise FileNotFoundError("first representation cache entry is missing")
    feature_dim = int(first.reshape(1, -1).shape[-1])
    projector = nn.Sequential(nn.Linear(feature_dim, args.latent_dim), nn.LayerNorm(args.latent_dim))

    if args.projection_warmup > 0:
        warm = GeometryPreservingProjectionTrainer(projector, lr=args.lr)
        keys = index["observation_keys"]
        for epoch in range(args.projection_warmup):
            for start in range(0, len(keys), args.batch_size):
                feats = [cache.get_by_key(k) for k in keys[start : start + args.batch_size]]
                feats = [v for v in feats if v is not None]
                if feats:
                    warm.step(torch.cat(feats, dim=0))

    action_dim = int(np.prod(dataset.manifest.action_shape)) if dataset.manifest.action_shape else 1
    model = MultimodalWorldModel(
        args.latent_dim, action_dim, hidden=args.hidden, components=args.components, horizons=horizons
    )
    if dataset.manifest.constraints_shape is not None:
        constraint_count = int(np.prod(dataset.manifest.constraints_shape))
        model.risk = RiskConstraintModel(args.latent_dim, action_dim, hidden=args.hidden, constraints=constraint_count)

    before = evaluate_world_model_horizons(model, projector, seq, horizons, batch_size=args.batch_size)
    trainer = OfflineWorldModelTrainer(
        model, projector, lr=args.lr, overshoot_horizon=max(horizons), train_projector=False
    )
    history = trainer.fit(seq, epochs=args.epochs, batch_size=args.batch_size)
    after = evaluate_world_model_horizons(model, projector, seq, horizons, batch_size=args.batch_size)
    calibration = calibrate_horizon_uncertainty(
        model, projector, seq, epochs=args.calibration_epochs, batch_size=args.batch_size, lr=args.lr
    )
    risk_report = None
    if dataset.manifest.constraints_shape is not None:
        risk_ds = CachedRiskDataset(seq)
        risk_trainer = RiskConstraintTrainer(model.risk, projector, lr=args.lr)
        risk_trainer.fit(risk_ds, epochs=max(1, args.epochs // 2), batch_size=args.batch_size)
        risk_report = evaluate_risk_model(model.risk, projector, risk_ds, batch_size=args.batch_size).to_dict()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "format": "awa-v2.2-world-qualification-v1",
        "version": V2_VERSION,
        "dataset_sha256": dataset.manifest.sha256,
        "encoder_fingerprint": index["encoder_fingerprint"],
        "feature_dim": feature_dim,
        "latent_dim": args.latent_dim,
        "horizons": list(horizons),
        "sequences": len(seq),
        "train_steps": len(history),
        "before": before.to_dict(),
        "after": after.to_dict(),
        "uncertainty_calibration": calibration,
        "risk": risk_report,
    }
    report_path = out_dir / "qualification.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    checkpoint_path = out_dir / "world_model.pt"
    torch.save({
        "format": "awa-v2.2-world-checkpoint-v1",
        "dataset_sha256": dataset.manifest.sha256,
        "encoder_fingerprint": index["encoder_fingerprint"],
        "model_state": model.state_dict(),
        "projector_state": projector.state_dict(),
        "feature_dim": feature_dim,
        "latent_dim": args.latent_dim,
        "action_dim": action_dim,
        "hidden": args.hidden,
        "components": args.components,
        "horizons": horizons,
        "risk_hidden": int(model.risk.net[0].out_features),
        "risk_constraints": int(model.risk.constraints),
    }, checkpoint_path)
    print(json.dumps({**report, "report_path": str(report_path), "checkpoint_path": str(checkpoint_path)}, indent=2))


def actor_baseline_smoke_main():
    """Train the v2.3 no-search continuous actor control on a small offline dataset."""
    from awa.v2.actor_baseline import LatentTransitionDataset, OfflineActorCriticBaseline
    p=argparse.ArgumentParser(description='v2.3 actor-only offline TD3+BC baseline smoke')
    p.add_argument('--epochs',type=int,default=8)
    args=p.parse_args()
    rng=np.random.default_rng(23); n=128
    states=rng.uniform(-1,1,size=(n,4)).astype(np.float32)
    # first 2 dims = point, final 2 = target; behavior is a noisy target-directed controller
    actions=np.clip((states[:,2:4]-states[:,:2])/.3 + rng.normal(scale=.12,size=(n,2)),-1,1).astype(np.float32)
    next_states=states.copy(); next_states[:,:2]=np.clip(states[:,:2]+.15*actions,-1.25,1.25)
    before=np.linalg.norm(states[:,2:4]-states[:,:2],axis=1); after=np.linalg.norm(next_states[:,2:4]-next_states[:,:2],axis=1)
    rewards=(2*(before-after)-.01*np.square(actions).sum(1)).astype(np.float32); dones=(after<.1).astype(np.float32)
    ds=LatentTransitionDataset(states,actions,rewards,next_states,dones)
    trainer=OfflineActorCriticBaseline(4,2,[-1,-1],[1,1],hidden=64,lr=1e-3)
    report=trainer.fit(ds,epochs=args.epochs,batch_size=64)
    probe=trainer.action(states[:8])
    print(json.dumps({'status':'ok','version':V2_VERSION,'report':report.to_dict(),'probe_shape':list(probe.shape)},indent=2))


def planner_qualify_main():
    """Exact-state real-branch planner qualification + VOC fitting on ContinuousPointEnv."""
    from awa.environments.continuous_point import ContinuousPointEnv
    from awa.v2.planner_qualification import PointOracleWorld,PointHeuristicActor,qualify_planners
    from awa.v2.planners import CEMPlanner,MPPIPlanner,PolicySeededMPPI,PolicySeededICEM,GradientPlanner
    from awa.v2.arena import backend_budget_for_calls
    p=argparse.ArgumentParser(description='v2.3 exact-branch planner qualification smoke')
    p.add_argument('--world-calls',type=int,default=64)
    p.add_argument('--states',type=int,default=16)
    p.add_argument('--branch-horizon',type=int,default=3)
    p.add_argument('--voc-epochs',type=int,default=120)
    p.add_argument('--out',default=None)
    args=p.parse_args()
    world=PointOracleWorld(); actor=PointHeuristicActor(.55); low=[-1,-1]; high=[1,1]
    planners=[
        CEMPlanner(world,low,high,horizon=4,candidates=16,elites=4,iterations=2),
        MPPIPlanner(world,low,high,horizon=4,candidates=16),
        PolicySeededMPPI(world,actor,low,high,horizon=4,candidates=16),
        PolicySeededICEM(world,actor,low,high,horizon=4,candidates=16,elites=4,iterations=2),
        GradientPlanner(world,low,high,horizon=4,steps=8,lr=.12),
    ]
    choices=[(pl,backend_budget_for_calls(pl,args.world_calls)) for pl in planners]
    report,table=qualify_planners(lambda seed:ContinuousPointEnv(seed=seed),world,actor,choices,
                                  episodes=4,max_states=args.states,branch_horizon=args.branch_horizon,
                                  voc_epochs=args.voc_epochs)
    payload={'status':'ok','version':V2_VERSION,'requested_world_calls':args.world_calls,**report.to_dict()}
    if args.out:
        Path(args.out).parent.mkdir(parents=True,exist_ok=True); Path(args.out).write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps(payload,indent=2))


def safety_smoke_main():
    from awa.v2.safety import SafeActionGuard
    from awa.v2.world import RiskConstraintModel
    risk=RiskConstraintModel(4,2,hidden=8,constraints=1)
    with torch.no_grad():
        for p in risk.parameters(): p.zero_()
        risk.net[-1].bias.fill_(10.0)
    guard=SafeActionGuard(risk,[-1,-1],[1,1],risk_limit=.5,recovery_action=[0,0])
    result=guard.validate(torch.zeros(1,4),torch.ones(1,2))
    print(json.dumps({'status':'ok','version':V2_VERSION,'accepted':result.accepted,'risk':result.risk,'action':result.action.tolist(),'reason':result.reason},indent=2))


def actor_train_main():
    """Train/save the v2.4 actor-only baseline in the exact latent space of a world checkpoint."""
    import hashlib
    from awa.v2.checkpoints import load_world_checkpoint, save_actor_checkpoint
    from awa.v2.actor_baseline import OfflineActorCriticBaseline, latent_dataset_from_cached_features

    p = argparse.ArgumentParser(description="Train a v2.4 actor checkpoint from cached features and a world checkpoint")
    p.add_argument("dataset")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--index", required=True)
    p.add_argument("--world-checkpoint", required=True)
    p.add_argument("--out", default="runs/v2_4_actor/actor.pt")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--low", default=None, help="comma-separated action lows; default -1")
    p.add_argument("--high", default=None, help="comma-separated action highs; default +1")
    args = p.parse_args()

    runtime = load_world_checkpoint(args.world_checkpoint)
    dataset = OfflineTransitionDataset(args.dataset)
    cache = RepresentationCache(args.cache_dir, namespace=dataset.manifest.sha256[:16])
    index = json.loads(Path(args.index).read_text(encoding="utf-8"))
    if index.get("encoder_fingerprint") != runtime.metadata.encoder_fingerprint:
        raise ValueError("world checkpoint and representation index encoder fingerprints differ")
    latent = latent_dataset_from_cached_features(dataset, cache, index, runtime.projector)
    action_dim = runtime.metadata.action_dim
    low = [-1.0] * action_dim if args.low is None else [float(v) for v in args.low.split(",")]
    high = [1.0] * action_dim if args.high is None else [float(v) for v in args.high.split(",")]
    if len(low) != action_dim or len(high) != action_dim:
        raise ValueError("action bounds must match checkpoint action_dim")
    trainer = OfflineActorCriticBaseline(runtime.metadata.latent_dim, action_dim, low, high,
                                         hidden=args.hidden, lr=args.lr)
    report = trainer.fit(latent, epochs=args.epochs, batch_size=args.batch_size)
    h = hashlib.sha256(Path(args.world_checkpoint).read_bytes()).hexdigest()
    save_actor_checkpoint(trainer, args.out, world_checkpoint_sha256=h,
                          dataset_sha256=dataset.manifest.sha256)
    print(json.dumps({"status":"ok","version":V2_VERSION,"actor_checkpoint":str(args.out),
                      "world_checkpoint_sha256":h,"report":report.to_dict()}, indent=2))


def benchmark_qualify_main():
    """Closed-loop multi-seed benchmark of actor and planners using saved v2 checkpoints."""
    import hashlib
    import yaml
    from awa.environments.factory import make_environment
    from awa.v2.checkpoints import load_world_checkpoint, load_actor_checkpoint, assert_identity_encoder_compatible
    from awa.v2.benchmark_qualification import qualify_closed_loop_controllers, bootstrap_paired_return_ci, write_closed_loop_report
    from awa.v2.planners import CEMPlanner, MPPIPlanner, PolicySeededMPPI, PolicySeededICEM, GradientPlanner

    p = argparse.ArgumentParser(description="v2.4 closed-loop checkpoint benchmark qualification")
    p.add_argument("--world-checkpoint", required=True)
    p.add_argument("--actor-checkpoint", required=True)
    p.add_argument("--env-config", required=True, help="YAML/JSON environment config accepted by make_environment")
    p.add_argument("--seeds", default="0,1,2,3,4")
    p.add_argument("--world-calls", type=int, default=128)
    p.add_argument("--planner-horizon", type=int, default=8)
    p.add_argument("--planners", default="cem,mppi,policy_mppi,policy_icem,gradient")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--out", default="runs/v2_4_benchmark/qualification.json")
    args = p.parse_args()

    runtime = load_world_checkpoint(args.world_checkpoint)
    assert_identity_encoder_compatible(runtime)
    actor_trainer = load_actor_checkpoint(args.actor_checkpoint)
    actor = actor_trainer.actor
    if actor_trainer.state_dim != runtime.metadata.latent_dim or actor_trainer.action_dim != runtime.metadata.action_dim:
        raise ValueError("actor/world checkpoint dimensions do not match")

    text = Path(args.env_config).read_text(encoding="utf-8")
    cfg = yaml.safe_load(text)
    seeds = [int(v) for v in args.seeds.split(",") if v.strip()]
    probe = make_environment(cfg, seed=seeds[0])
    low = np.asarray(probe.action_low, dtype=np.float32).reshape(-1)
    high = np.asarray(probe.action_high, dtype=np.float32).reshape(-1)
    obs_dim = int(probe.obs_dim); action_dim = int(probe.action_dim)
    close = getattr(probe, "close", None)
    if close is not None: close()
    if obs_dim != runtime.metadata.feature_dim:
        raise ValueError(f"identity benchmark env obs_dim={obs_dim} but checkpoint feature_dim={runtime.metadata.feature_dim}")
    if action_dim != runtime.metadata.action_dim:
        raise ValueError("environment action_dim does not match world checkpoint")

    H = int(args.planner_horizon)
    available = {
        "cem": CEMPlanner(runtime.world, low, high, horizon=H, candidates=64, elites=8, iterations=3),
        "mppi": MPPIPlanner(runtime.world, low, high, horizon=H, candidates=64),
        "policy_mppi": PolicySeededMPPI(runtime.world, actor, low, high, horizon=H, candidates=64),
        "policy_icem": PolicySeededICEM(runtime.world, actor, low, high, horizon=H, candidates=64, elites=8, iterations=3),
        "gradient": GradientPlanner(runtime.world, low, high, horizon=H, steps=16, lr=.08),
    }
    names = [v.strip() for v in args.planners.split(",") if v.strip()]
    unknown = [n for n in names if n not in available]
    if unknown: raise ValueError(f"unknown planners: {unknown}")
    specs = [(name, available[name], int(args.world_calls)) for name in names]
    encode = lambda obs: runtime.encode_observation(obs)
    env_factory = lambda seed: make_environment(cfg, seed=seed)
    env_meta = {
        "config": cfg,
        "world_checkpoint_sha256": hashlib.sha256(Path(args.world_checkpoint).read_bytes()).hexdigest(),
        "actor_checkpoint_sha256": hashlib.sha256(Path(args.actor_checkpoint).read_bytes()).hexdigest(),
        "requested_world_model_calls": int(args.world_calls),
        "planner_horizon": H,
    }
    report, rows = qualify_closed_loop_controllers(env_factory, encode, actor, specs, seeds=seeds,
                                                    max_steps=args.max_steps, environment_metadata=env_meta)
    by = {}
    for r in rows: by.setdefault(r.controller, {})[r.seed] = r.episode_return
    cis = {}
    actor_returns = [by["actor"][s] for s in seeds]
    for name in names:
        candidate = [by[name][s] for s in seeds]
        mean, lo, hi = bootstrap_paired_return_ci(actor_returns, candidate, samples=2000, seed=17)
        cis[name] = {"mean_gain":mean,"bootstrap_95pct_low":lo,"bootstrap_95pct_high":hi}
    out = Path(args.out)
    write_closed_loop_report(report, rows, out)
    payload = json.loads(out.read_text(encoding="utf-8")); payload["paired_bootstrap_ci"] = cis
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"status":"ok","version":V2_VERSION,"report":str(out),
                      "controllers":{k:v.to_dict() for k,v in report.controllers.items()},
                      "paired_bootstrap_ci":cis}, indent=2))


def benchmark_smoke_main():
    """Dependency-free v2.4 multi-seed closed-loop benchmark smoke using the exact point oracle."""
    from awa.environments.continuous_point import ContinuousPointEnv
    from awa.v2.planner_qualification import PointOracleWorld, PointHeuristicActor, identity_encoder
    from awa.v2.benchmark_qualification import qualify_closed_loop_controllers
    from awa.v2.planners import CEMPlanner, MPPIPlanner, PolicySeededMPPI
    world = PointOracleWorld(); actor = PointHeuristicActor(.55); low=[-1,-1]; high=[1,1]
    specs = [
        ("cem", CEMPlanner(world,low,high,horizon=4,candidates=16,elites=4,iterations=2),64),
        ("mppi", MPPIPlanner(world,low,high,horizon=4,candidates=16),64),
        ("policy_mppi", PolicySeededMPPI(world,actor,low,high,horizon=4,candidates=16),64),
    ]
    report,_=qualify_closed_loop_controllers(lambda seed:ContinuousPointEnv(seed=seed),identity_encoder,actor,specs,seeds=[0,1,2],max_steps=20,
                                              environment_metadata={"kind":"continuous_point","oracle_world":True})
    print(json.dumps({"status":"ok","version":V2_VERSION,**report.to_dict()},indent=2))



def game_representation_smoke_main():
    """Dependency-free qualification of causal game clips, telemetry fusion and streaming parity."""
    from awa.v2.video_representation import (
        VideoClipSpec, ToyVideoBackbone, EpisodeClipBuilder,
        StreamingGameFeatureEncoder, precompute_game_video_cache,
    )
    with tempfile.TemporaryDirectory() as td:
        dataset_path = write_synthetic_game_dataset(Path(td) / "game.npz", episodes=2, episode_length=6)
        dataset = OfflineTransitionDataset(dataset_path)
        spec = VideoClipSpec(frames_per_clip=4, frame_stride=1)
        backbone = ToyVideoBackbone(feature_dim=12)
        cache = RepresentationCache(Path(td) / "cache", namespace=dataset.manifest.sha256[:16])
        index_path = Path(td) / "cache" / dataset.manifest.sha256[:16] / "game-index.json"
        index = precompute_game_video_cache(dataset, backbone, cache, clip_spec=spec, index_path=index_path)
        builder = EpisodeClipBuilder(dataset.arrays["observations"], dataset.arrays["next_observations"], dataset.arrays["dones"], spec)
        stream = StreamingGameFeatureEncoder(backbone, spec, telemetry_dim=int(np.prod(dataset.manifest.telemetry_shape or (0,))))
        # Compare streaming to the offline current clip at the end of the first episode.
        end = 5
        for i in range(end + 1):
            stream_feature = stream.step(dataset.arrays["observations"][i], dataset.arrays["telemetry"][i])
        key = index["observation_keys"][end]
        offline = cache.get_by_key(key)
        parity = float((stream_feature.cpu() - offline).abs().max())
        print(json.dumps({
            "status": "ok",
            "version": V2_VERSION,
            "dataset_sha256": dataset.manifest.sha256,
            "clip_spec": spec.to_dict(),
            "feature_dim": index["feature_dim"],
            "telemetry_dim": index["telemetry_dim"],
            "cache": cache.stats(),
            "streaming_offline_max_abs_error": parity,
            "index": str(index_path),
        }, indent=2))


def game_cache_main():
    """Cache pretrained V-JEPA/game-video features into the standard Aether world-model index."""
    from awa.v2.video_representation import (
        VideoClipSpec, VJEPA2HFBackbone, VJEPA21TorchHubBackbone,
        ToyVideoBackbone, precompute_game_video_cache,
    )
    p = argparse.ArgumentParser(description="Precompute causal game-video representations for Aether v2.5")
    p.add_argument("dataset", nargs="?", default=None, help="NPZ game transition dataset; omit for dependency-free smoke data")
    p.add_argument("--cache-dir", default=".awa-game-representation-cache")
    p.add_argument("--provider", choices=("toy-video", "vjepa2-hf", "vjepa21-torchhub"), default="toy-video")
    p.add_argument("--model-name-or-path", default="facebook/vjepa2-vitl-fpc64-256")
    p.add_argument("--repo-or-dir", default=None, help="local facebookresearch/vjepa2 checkout for V-JEPA 2.1")
    p.add_argument("--hub-model", default="vjepa2_1_vit_base_384")
    p.add_argument("--feature-dim", type=int, default=None, help="required for hub models that do not expose embed_dim")
    p.add_argument("--frames-per-clip", type=int, default=None, help="defaults to model config (64 for official V-JEPA 2 checkpoints)")
    p.add_argument("--frame-stride", type=int, default=1)
    p.add_argument("--allow-download", action="store_true")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    temporary = None
    if args.dataset is None:
        temporary = tempfile.TemporaryDirectory()
        dataset_path = write_synthetic_game_dataset(Path(temporary.name) / "synthetic_game.npz", episodes=2, episode_length=8)
    else:
        dataset_path = Path(args.dataset)
    dataset = OfflineTransitionDataset(dataset_path)
    if len(dataset.manifest.observation_shape) != 3:
        raise ValueError("game video cache requires frame observations shaped HWC or CHW")

    if args.provider == "toy-video":
        backbone = ToyVideoBackbone(feature_dim=int(args.feature_dim or 16))
        default_frames = 4
    elif args.provider == "vjepa2-hf":
        backbone = VJEPA2HFBackbone(
            args.model_name_or_path,
            local_files_only=not args.allow_download,
            device=args.device,
        )
        default_frames = int(getattr(backbone, "frames_per_clip", 64))
    else:
        if not args.repo_or_dir and not args.allow_download:
            raise ValueError("vjepa21-torchhub requires --repo-or-dir for local-first loading")
        backbone = VJEPA21TorchHubBackbone(
            args.repo_or_dir or "facebookresearch/vjepa2",
            model_name=args.hub_model,
            feature_dim=args.feature_dim,
            allow_download=args.allow_download,
            device=args.device,
        )
        default_frames = 64
    spec = VideoClipSpec(frames_per_clip=int(args.frames_per_clip or default_frames), frame_stride=args.frame_stride)
    cache = RepresentationCache(args.cache_dir, namespace=dataset.manifest.sha256[:16])
    index_path = Path(args.cache_dir) / dataset.manifest.sha256[:16] / "game-index.json"
    index = precompute_game_video_cache(dataset, backbone, cache, clip_spec=spec, index_path=index_path)
    print(json.dumps({
        "status": "ok", "version": V2_VERSION, "provider": args.provider,
        "dataset_sha256": dataset.manifest.sha256,
        "transitions": len(dataset), "encoder_fingerprint": index["encoder_fingerprint"],
        "feature_dim": index["feature_dim"], "visual_feature_dim": index["visual_feature_dim"],
        "telemetry_dim": index["telemetry_dim"], "clip_spec": index["clip_spec"],
        "cache": cache.stats(), "writes": index["writes"], "hits": index["hits"],
        "index": str(index_path),
    }, indent=2))
    if temporary is not None:
        temporary.cleanup()


def curriculum_smoke_main():
    from awa.v2.curriculum import ProceduralTaskFactory, LearningFrontierScheduler, TaskOutcome
    f=ProceduralTaskFactory(26); sched=LearningFrontierScheduler(seed=26)
    current=f.batch(6,.55,4,'current'); mastered=f.batch(2,.2,3,'mastered'); harder=f.batch(7,.75,3,'harder'); novel=f.batch(10,.6,3,'novel')
    # Synthesize history demonstrating a learning frontier on current tasks and mastery on old tasks.
    for t in current:
        for i in range(12): sched.record(TaskOutcome(t.task_id,i>=5,1.0,.25,.25,.4))
    for t in mastered:
        for _ in range(12): sched.record(TaskOutcome(t.task_id,True,1.0,.02,.01,.05))
    chosen=sched.stage_mix(current,mastered,harder,novel,20)
    print(json.dumps({'status':'ok','version':V2_VERSION,'tasks':len(chosen),'stages':[t.stage for t in chosen],
                      'current_priority':sched.priority(current[0]),'mastered_priority':sched.priority(mastered[0])},indent=2))


def reusable_learning_smoke_main():
    from awa.environments.continuous_point import ContinuousPointEnv
    from awa.v2.curriculum import ProceduralTaskFactory, LearningFrontierScheduler, TaskOutcome
    from awa.v2.experience import ExperienceAnalyzer, RunningNovelty, branch_actions
    from awa.v2.replay import PrioritizedExperienceBuffer
    from awa.v2.reasoning import ReasoningEffortController
    from awa.v2.distill import TeacherPool, TeacherCandidate
    from awa.v2.skills import SkillDiscovery, SkillRegistry
    from awa.v2.engram import EngramLite
    from awa.v2.training import ExperienceQueue, RolloutEnvelope, StalenessPolicy

    factory=ProceduralTaskFactory(260); task=factory.make(10,.6,0,'novel')
    scheduler=LearningFrontierScheduler(seed=260)
    analyzer=ExperienceAnalyzer(.2,3); novelty=RunningNovelty(); replay=PrioritizedExperienceBuffer(capacity=32,seed=260)
    for i in range(6):
        n=novelty.observe([float(i%2),1,0])
        rec=analyzer.make(state=[0,0],goal=[1,1],action=[1,0],predicted_next=[0,0],actual_next=[.4,.1],reward=.1,
                          success=i>2,novelty=n,uncertainty=.2,risk=.1,td_error=.3,task_importance=.5,information_value=.2)
        replay.add(rec); scheduler.record(TaskOutcome(task.task_id,rec.success,rec.reward,rec.prediction_error,rec.novelty,.5))
    env=ContinuousPointEnv(seed=9); env.reset(); cf=branch_actions(env,[[-1,0],[1,0],[0,1]],'smoke')
    effort=ReasoningEffortController.default(.1).choose([0],predicted_gains=[0,.08,.2,.24,.25])
    teacher=TeacherPool().select([TeacherCandidate('mppi',np.array([.3,0],np.float32),1.0,.1,.1),TeacherCandidate('icem',np.array([.4,0],np.float32),.8,.2,.1)])
    skills=SkillDiscovery(min_examples=3)
    st=np.array([[0,0],[.5,0],[1,0]],np.float32); ac=np.array([[1,0],[1,0]],np.float32)
    for _ in range(4): skills.observe(st,ac,True)
    registry=SkillRegistry(); promoted=0
    for s in skills.candidates(): promoted += int(registry.register(s,min_success=.6,min_confidence=.5))
    engram=EngramLite(capacity=16,buckets=1); engram.put('physics_regime',[1,0,0],{'name':'low_friction'})
    recall=engram.query([1,.01,0],1,'physics_regime')
    q=ExperienceQueue(staleness=StalenessPolicy(max_lag=2)); q.push(RolloutEnvelope({'x':1},policy_version=1)); q.push(RolloutEnvelope({'x':2},policy_version=4))
    accepted,weights=q.pop_batch(4,current_policy_version=4)
    items,_,_=replay.sample(4)
    print(json.dumps({'status':'ok','version':V2_VERSION,'task':task.to_dict(),'curriculum_priority':scheduler.priority(task),
                      'replay_samples':len(items),'counterfactual_branches':len(cf.branches),'consequence_spread':cf.consequence_spread(),
                      'reasoning_effort':effort.to_dict(),'selected_teacher':None if teacher is None else teacher.teacher,
                      'skills_promoted':promoted,'engram_recall':recall[0][1].payload if recall else None,
                      'fresh_rollouts':len(accepted),'staleness_weights':weights},indent=2))


def closed_loop_learning_smoke_main():
    """Exercise the v2.7 reusable-learning components as one resumable loop."""
    from awa.v2.training import ReusableLearningEngine, ReusableLearningConfig, TrainingModeController
    from awa.v2.distill import TeacherCandidate
    from awa.v2.replay import LatentReplayExporter

    engine = ReusableLearningEngine(ReusableLearningConfig(replay_capacity=64, skill_min_examples=1, seed=27))
    engine.mode_controller = TrainingModeController(exploit_steps=1, explore_steps=1, qualify_steps=1)
    task = engine.factory.make(4, .45, 0, "smoke")
    engine.register_tasks([task])
    records = []
    states = []
    actions = []
    achieved = []
    for i in range(6):
        state = np.asarray([i / 6.0, 0.0], dtype=np.float32)
        action = np.asarray([.2, .0], dtype=np.float32)
        actual = state + np.asarray([.12, 0.0], dtype=np.float32)
        predicted = state + np.asarray([.10, 0.0], dtype=np.float32)
        mode = engine.mode_controller.step()
        step = engine.record_step(
            task,
            state=state,
            goal=np.asarray([1.0, 0.0], dtype=np.float32),
            action=action,
            predicted_next=predicted,
            actual_next=actual,
            task_reward=.1,
            uncertainty=.1,
            risk=.05,
            td_error=.2,
            task_importance=.5,
            information_value=.25,
            success=(i == 5),
            mode=mode,
            metadata={"done": i == 5, "episode_id": "smoke-0"},
        )
        records.append(step.record); states.append(state); actions.append(action); achieved.append(actual)
    finish = engine.finish_episode(task, records, success=True, reward=1.0, planner_actions=2, total_actions=6, states=np.asarray(states + [achieved[-1]]), actions=np.asarray(actions))
    skill_policy_report = None
    if finish["promoted_skills"]:
        _, skill_policy_report = engine.train_skill_policy(finish["promoted_skills"][0], [-1, -1], [1, 1], hidden=16, epochs=2, batch_size=4, lr=1e-2, seed=27)
    teacher = engine.add_teacher_example(
        states[-1], np.asarray([1.0, 0.0], dtype=np.float32),
        [TeacherCandidate("policy_mppi", np.asarray([.15, 0], dtype=np.float32), value=1.0, risk=.05, compute_cost=.1)],
        student_value=.5,
    )
    her = engine.relabel_episode(np.asarray(achieved), np.asarray(actions), np.asarray([1.0, 0.0], dtype=np.float32))
    with tempfile.TemporaryDirectory() as td:
        checkpoint = engine.save(Path(td) / "reusable.json")
        restored = ReusableLearningEngine.load(checkpoint)
        dataset_path = LatentReplayExporter.export(engine.replay.items, Path(td) / "replay.npz")
        dataset = OfflineTransitionDataset(dataset_path)
        payload = {
            "status": "ok",
            "version": V2_VERSION,
            "summary": engine.summary(),
            "restored_summary": restored.summary(),
            "finish": finish,
            "her_examples": len(her),
            "teacher": None if teacher is None else teacher.teacher,
            "skill_policy_train": skill_policy_report,
            "exported_transitions": len(dataset),
            "checkpoint_bytes": checkpoint.stat().st_size,
        }
    print(json.dumps(payload, indent=2))


def transfer_smoke_main():
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.transfer import OODCategory, TransferEpisode, TransferBenchmark, build_transfer_suite

    factory = ProceduralTaskFactory(27)
    suite = build_transfer_suite(factory, count_per_category=2, difficulty=.6)
    rows = []
    for c_idx, category in enumerate(OODCategory):
        threshold = 1 + min(c_idx, 4)
        for task in suite[category]:
            for i in range(8):
                rows.append(TransferEpisode(
                    category, task.task_id, i,
                    success=i >= threshold,
                    reward=float(i >= threshold),
                    planner_dependency=max(0.05, .8 - .08 * i),
                    prediction_error=max(.02, .6 - .06 * i),
                ))
    report = TransferBenchmark(target_success=.8, window=3).evaluate(rows)
    print(json.dumps({"status": "ok", "version": V2_VERSION, **report.to_dict()}, indent=2))


def skill_contract_smoke_main():
    from awa.v2.skills import SkillContractModel, SkillContractTrainer

    rng = np.random.default_rng(27)
    states = rng.normal(size=(256, 3)).astype(np.float32)
    labels = np.stack([
        states[:, 0] > 0.0,
        states[:, 1] > .5,
        states[:, 2] < -.5,
    ], axis=1).astype(np.float32)
    model = SkillContractModel(3, hidden=32)
    trainer = SkillContractTrainer(model, lr=5e-3)
    report = trainer.fit(states, labels, epochs=12, batch_size=64, seed=27)
    decision = model.decision(np.asarray([1.0, 1.0, 0.0], dtype=np.float32))
    print(json.dumps({"status": "ok", "version": V2_VERSION, "train": report, "decision": decision.to_dict()}, indent=2))


def rollout_pool_smoke_main():
    from awa.environments.continuous_point import ContinuousPointEnv
    from awa.v2.training import LocalRolloutWorkerPool

    pool = LocalRolloutWorkerPool(workers=2, base_seed=27)
    episodes = pool.collect(
        lambda worker_id, seed: ContinuousPointEnv(seed=seed),
        lambda obs: np.zeros(2, dtype=np.float32),
        episodes_per_worker=2,
        max_steps=5,
        policy_version=7,
    )
    print(json.dumps({
        "status": "ok", "version": V2_VERSION,
        "episodes": len(episodes),
        "transitions": sum(len(x.payload) for x in episodes),
        "policy_versions": sorted({x.policy_version for x in episodes}),
        "worker_ids": sorted({x.worker_id for x in episodes}),
    }, indent=2))


def hardening_smoke_main():
    """Exercise v2.8 fixes that previously escaped the normal regression surface."""
    from awa.v2.training import ReusableLearningEngine, ReusableLearningConfig, StalenessPolicy, RolloutEnvelope
    from awa.v2.skills import SkillContractModel
    from awa.v2.experience import RunningNovelty
    from awa.v2.branching import verify_snapshot_roundtrip
    from awa.v2.safety import SafeActionGuard

    engine = ReusableLearningEngine(ReusableLearningConfig(skill_min_examples=1, seed=28))
    first = [t.task_id for t in engine.generate_adaptive_tasks(2)]
    second = [t.task_id for t in engine.generate_adaptive_tasks(2)]
    if set(first) & set(second):
        raise RuntimeError("adaptive curriculum emitted duplicate task ids")

    novelty = RunningNovelty(8)
    novelty.observe([1.0, 0.0])
    mixed_dim_novelty = novelty.observe([1.0, 0.0, 0.0])

    class Env:
        action_dim = 1
        def __init__(self): self.x = 0
        def reset(self, seed=None): self.x = 0; return np.asarray([0.0], np.float32), {}
        def step(self, action): self.x += 1; return np.asarray([self.x], np.float32), 1.0, False, False, {}
        def state_dict(self): return {"x": self.x, "observation": np.asarray([self.x], np.float32)}
        def load_state_dict(self, state): self.x = int(state["x"])

    env = Env(); env.reset(seed=28)
    verify_snapshot_roundtrip(env, np.zeros(1, np.float32))

    class Risk:
        def aggregate_risk(self, belief, action):
            return torch.abs(action).max(dim=-1, keepdim=True).values

    guard = SafeActionGuard(Risk(), [-1], [1], risk_limit=.5, recovery_action=[.25])
    guarded = guard.validate(torch.zeros(1, 1), torch.tensor([[.9]]))

    task = engine.factory.make(1, .2, 0)
    states = np.asarray([[0., 0.], [.2, 0.], [.4, 0.]], np.float32)
    actions = np.asarray([[.2, 0.], [.2, 0.]], np.float32)
    records = []
    for i in range(2):
        records.append(engine.record_step(
            task, state=states[i], goal=[.4, 0], action=actions[i],
            predicted_next=states[i + 1], actual_next=states[i + 1],
            task_reward=.5, uncertainty=.1, risk=0., success=i == 1,
            metadata={"array": np.asarray([i], dtype=np.int64)},
        ).record)
    finish = engine.finish_episode(task, records, success=True, reward=1., states=states, actions=actions)
    skill = finish["promoted_skills"][0]
    before, _ = engine.train_skill_policy(skill, [-1, -1], [1, 1], hidden=8, epochs=1, batch_size=2, seed=28)
    contract = SkillContractModel(2, goal_dim=2, hidden=8)
    engine.executable_skills.contracts.attach(skill, contract)

    with tempfile.TemporaryDirectory() as td:
        path = engine.save(Path(td) / "engine.json")
        restored = ReusableLearningEngine.load(path)
        after = restored.executable_skills.policies[skill]
        same = all(torch.equal(a.cpu(), b.cpu()) for a, b in zip(before.state_dict().values(), after.state_dict().values()))
        if not same:
            raise RuntimeError("executable skill state changed across checkpoint restore")

    stale = StalenessPolicy(max_lag=4).weight(10, RolloutEnvelope(None, 11))
    if stale != 0.0:
        raise RuntimeError("future-policy rollout was not rejected")

    print(json.dumps({
        "status": "ok",
        "version": V2_VERSION,
        "adaptive_unique": True,
        "mixed_dimension_novelty": mixed_dim_novelty,
        "gymnasium_snapshot_5tuple": True,
        "recovery_risk": guarded.risk,
        "skill_checkpoint_exact": True,
        "future_rollout_weight": stale,
    }, indent=2))


def game_lab_generate_main():
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import collect_game_dataset, LogicalArenaTeacher
    p=argparse.ArgumentParser(description="Generate a procedural Aether game-training dataset")
    p.add_argument("--out",default="runs/v2_10_game/game.npz")
    p.add_argument("--stages",default="1,2,3,4,5,6,7,8,9,10,11,12")
    p.add_argument("--difficulty",type=float,default=.5)
    p.add_argument("--tasks-per-stage",type=int,default=2)
    p.add_argument("--episodes-per-task",type=int,default=2)
    p.add_argument("--horizon",type=int,default=120)
    p.add_argument("--seed",type=int,default=29)
    args=p.parse_args()
    factory=ProceduralTaskFactory(args.seed)
    tasks=[]
    for stage in [int(x) for x in args.stages.split(",") if x.strip()]:
        tasks.extend(factory.batch(stage,args.difficulty,args.tasks_per_stage,"game_train"))
    report=collect_game_dataset(tasks,args.out,episodes_per_task=args.episodes_per_task,horizon=args.horizon,policy=LogicalArenaTeacher())
    print(json.dumps({"status":"ok","version":V2_VERSION,"dataset":str(Path(args.out)),**report.to_dict()},indent=2))


def game_train_main():
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import train_game_stack
    p=argparse.ArgumentParser(description="Train the v2.10 goal-conditioned temporal game world model and actor")
    p.add_argument("dataset")
    p.add_argument("--out-dir",default="runs/v2_10_game/train")
    p.add_argument("--stages",default="1,2,3,5,6,9,10,12")
    p.add_argument("--difficulty",type=float,default=.5)
    p.add_argument("--sequence-length",type=int,default=5)
    p.add_argument("--world-epochs",type=int,default=4)
    p.add_argument("--actor-epochs",type=int,default=8)
    p.add_argument("--calibration-epochs",type=int,default=3)
    p.add_argument("--hidden",type=int,default=96)
    p.add_argument("--seed",type=int,default=29)
    p.add_argument("--device",default="cpu",help="cpu/cuda/mps")
    p.add_argument("--precision",choices=["fp32","bf16","fp16"],default="fp32")
    args=p.parse_args()
    factory=ProceduralTaskFactory(args.seed)
    tasks=[factory.make(int(s),args.difficulty,i,"game_eval") for i,s in enumerate([x for x in args.stages.split(",") if x.strip()])]
    report=train_game_stack(args.dataset,tasks,args.out_dir,sequence_length=args.sequence_length,
                            world_epochs=args.world_epochs,actor_epochs=args.actor_epochs,
                            calibration_epochs=args.calibration_epochs,hidden=args.hidden,seed=args.seed,
                            device=args.device,precision=args.precision)
    print(json.dumps({"status":"ok","version":V2_VERSION,**report.to_dict()},indent=2))


def game_loop_smoke_main():
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import ReusableGameLoop, LogicalArenaTeacher
    f=ProceduralTaskFactory(29)
    tasks=[f.make(s,.35,s,"game_smoke") for s in (1,2,3,4,5,6,9,12)]
    loop=ReusableGameLoop(horizon=80)
    report=loop.run_tasks(tasks,episodes_per_task=1,policy=LogicalArenaTeacher())
    with tempfile.TemporaryDirectory() as td:
        path=loop.save(Path(td)/"game_loop.json")
        restored=ReusableGameLoop.load(path,horizon=80)
        restored_summary=restored.engine.summary()
    print(json.dumps({"status":"ok","version":V2_VERSION,"report":report.to_dict(),"restored":restored_summary},indent=2))


def game_lab_smoke_main():
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import collect_game_dataset, train_game_stack, LogicalArenaTeacher
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); f=ProceduralTaskFactory(29)
        tasks=[f.make(s,.3,s,"game_lab_smoke") for s in (1,2,3,5,6,9)]
        collected=collect_game_dataset(tasks,root/"game.npz",episodes_per_task=1,horizon=70,policy=LogicalArenaTeacher())
        trained=train_game_stack(root/"game.npz",tasks,root/"train",sequence_length=4,horizons=(1,2,4),world_epochs=1,actor_epochs=1,calibration_epochs=1,batch_size=32,hidden=48,seed=29)
        print(json.dumps({"status":"ok","version":V2_VERSION,"collection":collected.to_dict(),"training":trained.to_dict()},indent=2))


def game_adaptation_smoke_main():
    """Exercise the v2.10 objective/belief/replay integration in one command."""
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import ReusableGameLoop, LogicalArenaTeacher, train_game_stack, GOAL_DIM, OBS_DIM
    from awa.v2.game.continual import export_prioritized_game_replay
    from awa.v2.transfer import build_transfer_suite

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        factory = ProceduralTaskFactory(210)
        tasks = [factory.make(s, .22, s, "v210_adapt") for s in (1, 3)]
        loop = ReusableGameLoop(horizon=50)
        loop_report = loop.run_tasks(
            tasks, episodes_per_task=1, policy=LogicalArenaTeacher(), counterfactual_branches=1
        )
        replay_report = export_prioritized_game_replay(loop.engine, root / "prioritized_game.npz")
        trained = train_game_stack(
            root / "prioritized_game.npz", tasks, root / "train",
            sequence_length=1, horizons=(1,), world_epochs=1, actor_epochs=1,
            calibration_epochs=1, batch_size=32, hidden=16, seed=210,
            one_step_aux_epochs=0,
        )
        suite = build_transfer_suite(factory, count_per_category=1, difficulty=.4)
        comp = suite["compositional_ood"][0]
        task_ood = suite["task_ood"][0]
        if len(comp.goal.get("objectives", ())) < 2 or len(task_ood.goal.get("objectives", ())) < 2:
            raise RuntimeError("OOD objective programs are not executable compositions")
        print(json.dumps({
            "status": "ok",
            "version": V2_VERSION,
            "observation_dim": OBS_DIM,
            "goal_dim": GOAL_DIM,
            "loop": loop_report.to_dict(),
            "replay": replay_report.to_dict(),
            "training": trained.to_dict(),
            "compositional_objectives": list(comp.goal["objectives"]),
            "task_ood_objectives": list(task_ood.goal["objectives"]),
        }, indent=2))



def empirical_scaling_smoke_main():
    """v2.11 smoke: stochastic planning, epistemic uncertainty, sharded replay and iterative aggregation."""
    from awa.v2.world import MultimodalWorldModel
    from awa.v2.ensemble import WorldModelEnsemble
    from awa.v2.planners import RiskAwarePolicySeededMPPI
    from awa.v2.game import HybridGameActionCodec, IterativeDataAggregator, ReusableGameLoop, LogicalArenaTeacher, SeedMetric, summarize_seed_metrics
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.replay import ShardedReplayStore

    class ZeroActor:
        def deterministic_action(self, belief):
            return torch.zeros(belief.shape[0], 4, device=belief.device)

    torch.manual_seed(211)
    w1=MultimodalWorldModel(8,4,hidden=16,components=3,horizons=(1,2))
    torch.manual_seed(212)
    w2=MultimodalWorldModel(8,4,hidden=16,components=3,horizons=(1,2))
    ensemble=WorldModelEnsemble([w1,w2])
    belief=torch.zeros(1,8); action=torch.zeros(1,4)
    epistemic=float(ensemble.disagreement(belief,action).next_state.item())
    planner=RiskAwarePolicySeededMPPI(w1,ZeroActor(),[-1]*4,[1]*4,horizon=2,candidates=4,samples=3)
    plan=planner.plan(belief,budget=4)
    codec=HybridGameActionCodec(); hybrid=codec.decode(codec.quantize(plan.action.squeeze(0).cpu().numpy()))

    with tempfile.TemporaryDirectory() as td:
        store=ShardedReplayStore(Path(td)/"replay",shard_size=3)
        store.append({"observation":np.arange(7,dtype=np.float32)[:,None],"reward":np.arange(7,dtype=np.float32)})
        integrity=store.verify()

    factory=ProceduralTaskFactory(211); task=factory.make(1,.12,0,"v211")
    loop=ReusableGameLoop(horizon=30); teacher=LogicalArenaTeacher()
    rounds,_,_=IterativeDataAggregator(loop,seed=211).run([task],teacher,teacher,rounds=2,planner_schedule=lambda r:.5)
    ci=summarize_seed_metrics([SeedMetric(i,.5+.1*i,1+i,.4-.1*i,.2) for i in range(3)],resamples=250,seed=211)
    print(json.dumps({
        "status":"ok","version":V2_VERSION,"epistemic":epistemic,
        "robust_planner":{"calls":plan.world_model_calls,"cvar":plan.metadata["cvar_score"],"samples":plan.metadata["samples"]},
        "hybrid_action":{"attack":hybrid.attack,"interact":hybrid.interact},
        "replay_integrity":integrity,"aggregation":[x.to_dict() for x in rounds],"multiseed":ci.to_dict(),
    },indent=2))


def controlled_scaling_smoke_main():
    """v2.12 smoke: matched-compute architecture, data and context scaling with resume."""
    import yaml
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import collect_game_dataset, LogicalArenaTeacher
    from awa.v2.scaling import run_controlled_scaling_experiment
    with tempfile.TemporaryDirectory() as td:
        root=Path(td); factory=ProceduralTaskFactory(212)
        tasks=[factory.make(1,.12,0,"v212"),factory.make(3,.18,1,"v212")]
        dataset=root/"game.npz"
        collect_game_dataset(tasks,dataset,episodes_per_task=1,horizon=28,policy=LogicalArenaTeacher())
        cfg={
            "seed":212,
            "runtime":{"device":"cpu","precision":"fp32"},
            "model":{"hidden":16,"components":2,"horizons":[1,2],"experts":4,"top_k":1},
            "training":{"epochs":1,"batch_size":16,"sequence_length":2,"lr":7e-4},
            "data":{"fractions":[.5,1.0],"model_kind":"monolithic"},
            "scaling":{"hidden_sizes":[16]},
            "context":{"specs":[[2,2,1],[3,4,2]],"model_kind":"monolithic"},
            "qualification":{"seeds":[212,213],"bootstrap_resamples":200,"confidence":.95},
        }
        report,state=run_controlled_scaling_experiment(dataset,root/"scaling",cfg)
        # Run a second time to prove exact resume rather than recomputing phases.
        report2,state2=run_controlled_scaling_experiment(dataset,root/"scaling",cfg)
        print(json.dumps({
            "status":"ok","version":V2_VERSION,"points":len(report.points),"summary_groups":len(report.summary),
            "device":report.device,"precision":report.precision,"resumed":state["config_hash"]==state2["config_hash"] and len(report2.points)==len(report.points),
            "phases":{k:v["status"] for k,v in state2["phases"].items()},
        },indent=2))


def scaling_sweep_main():
    """Run a resumable v2.12 controlled scaling experiment from YAML configuration."""
    import yaml
    from awa.v2.scaling import run_controlled_scaling_experiment
    p=argparse.ArgumentParser(description="Aether v2.12 matched-compute model/data/context scaling sweep")
    p.add_argument("dataset",help="goal-conditioned procedural-game NPZ dataset")
    p.add_argument("--config",default="configs/v2_12_controlled_scaling.yaml")
    p.add_argument("--out-dir",default="runs/v2_12_scaling")
    p.add_argument("--device",default=None,help="override config runtime.device (auto/cpu/cuda/mps)")
    p.add_argument("--precision",choices=["fp32","bf16","fp16"],default=None)
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.device is not None: cfg.setdefault("runtime",{})["device"]=args.device
    if args.precision is not None: cfg.setdefault("runtime",{})["precision"]=args.precision
    report,state=run_controlled_scaling_experiment(args.dataset,args.out_dir,cfg)
    print(json.dumps({"status":"ok","version":V2_VERSION,"report":str(Path(args.out_dir)/"scaling_report.json"),"points":len(report.points),"summary_groups":len(report.summary),"phases":{k:v["status"] for k,v in state["phases"].items()}},indent=2))


def training_campaign_main():
    """Plan or execute a durable v2.13 cumulative game-training campaign."""
    import yaml
    from awa.v2.campaign import GameTrainingCampaign, campaign_plan
    p=argparse.ArgumentParser(description="Aether v2.13 durable 100k->1M+ training campaign")
    p.add_argument("--config",default="configs/v2_13_training_campaign.yaml")
    p.add_argument("--out-dir",default="runs/v2_13_campaign")
    p.add_argument("--execute",action="store_true",help="actually collect/train; without this flag only print the campaign plan")
    p.add_argument("--stop-after-stage",default=None,help="stop after the named stage (useful for staged qualification)")
    p.add_argument("--device",choices=["auto","cpu","cuda","mps"],default=None)
    p.add_argument("--precision",choices=["fp32","bf16","fp16"],default=None)
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if args.device is not None: cfg.setdefault("runtime",{})["device"]=args.device
    if args.precision is not None: cfg.setdefault("runtime",{})["precision"]=args.precision
    if not args.execute:
        print(json.dumps({"status":"plan","version":V2_VERSION,**campaign_plan(cfg)},indent=2))
        return
    runner=GameTrainingCampaign(cfg,args.out_dir)
    report=runner.run(stop_after_stage=args.stop_after_stage)
    print(json.dumps({"status":"ok","version":V2_VERSION,"report":str(Path(args.out_dir)/"campaign_report.json"),**report},indent=2))


def training_campaign_smoke_main():
    """Tiny end-to-end campaign proving collection, training, promotion and exact resume."""
    from awa.v2.campaign import GameTrainingCampaign, campaign_plan
    cfg={
        "seed":213,
        "runtime":{"device":"cpu","precision":"fp32","strict_resources":False},
        "replay":{"shard_size":20},
        "promotion":{"max_actor_success_regression":1.0,"max_planner_success_regression":1.0,"max_risk_brier":1.0},
        "stop_on_failed_promotion":True,
        "stages":[
            {"name":"smoke-30","target_transitions":30,"curriculum_stages":[1],"difficulty":.1,
             "sequence_length":1,"hidden":16,"world_epochs":1,"actor_epochs":1,"calibration_epochs":0,
             "batch_size":16,"episodes_per_task":1,"tasks_per_stage":1,"horizon":20,"teacher_fraction":1.0},
            {"name":"smoke-60","target_transitions":60,"curriculum_stages":[1,3],"difficulty":.12,
             "sequence_length":1,"hidden":16,"world_epochs":1,"actor_epochs":1,"calibration_epochs":0,
             "batch_size":16,"episodes_per_task":1,"tasks_per_stage":1,"horizon":20,"teacher_fraction":1.0},
        ],
    }
    with tempfile.TemporaryDirectory() as td:
        runner=GameTrainingCampaign(cfg,Path(td)/"campaign")
        first=runner.run()
        second=GameTrainingCampaign(cfg,Path(td)/"campaign").run()
        phases=second["stages"]
        resumed=(first["replay_transitions"]==second["replay_transitions"] and len(phases)==2)
        print(json.dumps({"status":"ok","version":V2_VERSION,"plan":campaign_plan(cfg),
                          "replay_transitions":second["replay_transitions"],"stable_stage":second["stable_checkpoint"]["stage"],
                          "resumed":resumed,"replay_integrity_failures":second["replay_integrity_failures"],
                          "checkpoint_integrity_failures":second["checkpoint_integrity_failures"]},indent=2))


def game_bridge_check_main():
    """Connect to a real-game bridge and verify reset/step and optional snapshot replay."""
    from awa.v2.game.bridge_protocol import JsonLineGameBridgeClient
    p = argparse.ArgumentParser(description="Aether v2.14 external game-bridge qualification")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--seed", type=int, default=214)
    p.add_argument("--steps", type=int, default=2)
    args = p.parse_args()
    with JsonLineGameBridgeClient(args.host, args.port) as client:
        caps = client.capabilities
        obs, goal, info = client.reset(seed=args.seed)
        snapshot_ok = None
        snap = client.snapshot() if caps.supports_snapshot else None
        rows = []
        for _ in range(max(1, args.steps)):
            tr = client.step(np.zeros(caps.action_dim, dtype=np.float32))
            rows.append({"reward": tr.reward, "terminated": tr.terminated, "truncated": tr.truncated})
            if tr.terminated or tr.truncated:
                break
        if snap is not None:
            restored, restored_goal, _ = client.restore(snap)
            snapshot_ok = bool(np.allclose(restored, obs) and np.allclose(restored_goal, goal))
        print(json.dumps({"status": "ok", "version": V2_VERSION, "capabilities": caps.to_dict(), "observation_dim": len(obs), "goal_dim": len(goal), "steps": rows, "snapshot_roundtrip": snapshot_ok}, indent=2))


def qualification_report_main():
    """Build the standardized multi-milestone qualification report from JSONL rows."""
    from awa.v2.qualification_report import read_milestone_jsonl, build_milestone_report, write_milestone_report
    p = argparse.ArgumentParser(description="Aether v2.14 100k/250k/500k/1M qualification report")
    p.add_argument("input", help="JSONL with one MilestoneRow per seed/checkpoint")
    p.add_argument("--out-dir", default="runs/v2_14_qualification")
    p.add_argument("--resamples", type=int, default=2000)
    args = p.parse_args()
    rows = read_milestone_jsonl(args.input)
    report = build_milestone_report(rows, resamples=args.resamples)
    jp, mp = write_milestone_report(report, args.out_dir)
    print(json.dumps({"status": "ok", "version": V2_VERSION, "json": str(jp), "markdown": str(mp), "milestones": len(report["milestones"])}, indent=2))


def empirical_qualification_smoke_main():
    """v2.14 smoke: telemetry, triage, tracks, ablations, report and process rollouts."""
    from awa.v2.telemetry import TelemetryRecorder
    from awa.v2.failure_triage import FailureEvidence, triage_failure
    from awa.v2.benchmark_tracks import BenchmarkTrack, RepresentationContract
    from awa.v2.ablation import run_ablation_suite, DEFAULT_ABLATIONS
    from awa.v2.qualification_report import MilestoneRow, build_milestone_report
    from awa.v2.training.process_rollout import ProcessArenaRolloutPool
    from awa.v2.curriculum import ProceduralTaskFactory

    with tempfile.TemporaryDirectory() as td:
        tele = TelemetryRecorder(Path(td) / "telemetry.jsonl", run_id="v214-smoke")
        tele.event("train", step=10, metrics={"loss": 1.0, "samples_per_sec": 42.0})
        tele.event("train", step=20, metrics={"loss": 0.5, "samples_per_sec": 44.0})
        telemetry = tele.summarize().to_dict()
        triage = triage_failure(FailureEvidence(False, world_prediction_error=.7, actor_return=0., planner_return=.1, planner_available=True)).to_dict()
        structured = RepresentationContract(BenchmarkTrack.STRUCTURED, observation_dim=32, telemetry_dim=0)
        def evaluator(spec, seed):
            penalty = 0.0 if spec.name == "full" else 0.05
            return {"success_rate": .8 + (seed % 2) * .01 - penalty}
        ablation = run_ablation_suite(evaluator, seeds=[214, 215], specs=DEFAULT_ABLATIONS[:3], resamples=100)
        rows = [
            MilestoneRow(100_000, 214, .4, .5, .3, .5, .7, .1, 8, 100),
            MilestoneRow(100_000, 215, .42, .51, .31, .48, .68, .11, 7, 95),
            MilestoneRow(1_000_000, 214, .8, .85, .7, .2, .2, .5, 3, 35),
            MilestoneRow(1_000_000, 215, .82, .86, .71, .19, .19, .52, 2, 33),
        ]
        milestone = build_milestone_report(rows, resamples=100)
        factory = ProceduralTaskFactory(214)
        tasks = [factory.make(1, .1, 0, "v214-smoke"), factory.make(3, .15, 1, "v214-smoke")]
        import multiprocessing as _mp
        _method = "fork" if "fork" in _mp.get_all_start_methods() else "spawn"
        process_report = ProcessArenaRolloutPool(workers=1, base_seed=214, start_method=_method).collect(tasks, horizon=12, policy="teacher").to_dict()
        print(json.dumps({
            "status": "ok", "version": V2_VERSION, "telemetry": telemetry, "triage": triage,
            "representation_fingerprint": structured.fingerprint, "ablations": list(ablation["summary"]),
            "milestones": len(milestone["milestones"]), "process_rollouts": {k: process_report[k] for k in ("episodes", "successes", "total_steps", "workers")},
        }, indent=2))


def vizdoom_check_main():
    """Open a real ViZDoom scenario and verify Aether's action/state contract."""
    from awa.v2.game.vizdoom_env import ViZDoomScenario, ViZDoomConfig, ViZDoomAetherEnv
    p=argparse.ArgumentParser(description="Aether v2.15 real ViZDoom adapter check")
    p.add_argument("--scenario",choices=[x.value for x in ViZDoomScenario],default=ViZDoomScenario.MY_WAY_HOME.value)
    p.add_argument("--scenario-path",default=None)
    p.add_argument("--track",choices=["structured","pixel"],default="structured")
    p.add_argument("--seed",type=int,default=215)
    p.add_argument("--frame-skip",type=int,default=4)
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--visible",action="store_true")
    args=p.parse_args()
    env=ViZDoomAetherEnv(ViZDoomConfig(
        scenario=ViZDoomScenario(args.scenario),scenario_path=args.scenario_path,track=args.track,
        seed=args.seed,frame_skip=args.frame_skip,visible=args.visible,
    ))
    try:
        obs,info=env.reset(seed=args.seed)
        rows=[]
        for i in range(max(1,args.steps)):
            # Safe integration action: alternate gentle turns while moving forward.
            action=np.asarray([0.15 if i%2==0 else -0.15,0.65,-1.0,-1.0],dtype=np.float32)
            nxt,reward,terminated,truncated,step_info=env.step(action)
            rows.append({"reward":float(reward),"terminated":bool(terminated),"truncated":bool(truncated),"shape":list(np.asarray(nxt).shape)})
            if terminated or truncated: break
        print(json.dumps({
            "status":"ok","version":V2_VERSION,"scenario":args.scenario,"track":args.track,
            "observation_shape":list(np.asarray(obs).shape),"goal_dim":len(env.goal_vector()),
            "telemetry_dim":env.telemetry_dim,"buttons":list(env._button_names),"steps":rows,
            "snapshot_fidelity":"world_state_only",
        },indent=2))
    finally:
        env.close()


def vizdoom_collect_main():
    """Collect a structured or RGB ViZDoom transition dataset."""
    from awa.v2.game.vizdoom_env import ViZDoomScenario, ViZDoomConfig, ViZDoomAetherEnv
    from awa.v2.game.vizdoom_dataset import collect_vizdoom_dataset, ViZDoomExplorerPolicy
    p=argparse.ArgumentParser(description="Aether v2.15 ViZDoom dataset collector")
    p.add_argument("output")
    p.add_argument("--scenario",choices=[x.value for x in ViZDoomScenario],default=ViZDoomScenario.MY_WAY_HOME.value)
    p.add_argument("--scenario-path",default=None)
    p.add_argument("--track",choices=["structured","pixel"],default="pixel")
    p.add_argument("--episodes",type=int,default=8)
    p.add_argument("--horizon",type=int,default=600)
    p.add_argument("--seed",type=int,default=215)
    p.add_argument("--frame-skip",type=int,default=4)
    p.add_argument("--visible",action="store_true")
    args=p.parse_args()
    cfg=ViZDoomConfig(
        scenario=ViZDoomScenario(args.scenario),scenario_path=args.scenario_path,track=args.track,
        seed=args.seed,frame_skip=args.frame_skip,visible=args.visible,
    )
    def factory(): return ViZDoomAetherEnv(cfg)
    report=collect_vizdoom_dataset(factory,args.output,episodes=args.episodes,horizon=args.horizon,base_seed=args.seed,policy=ViZDoomExplorerPolicy(args.seed))
    print(json.dumps({"status":"ok","version":V2_VERSION,"output":str(Path(args.output)),**report.to_dict()},indent=2))


def vizdoom_vjepa_cache_main():
    """Precompute frozen V-JEPA 2 clip embeddings for a pixel ViZDoom dataset."""
    from awa.v2.datasets import OfflineTransitionDataset
    from awa.v2.representation import RepresentationCache
    from awa.v2.video_representation import VJEPA2HFBackbone, VideoClipSpec, precompute_game_video_cache
    from awa.v2.game.vizdoom_vjepa import validate_vjepa_clip_contract, write_vizdoom_vjepa_manifest, DEFAULT_VJEPA2_MODEL
    p=argparse.ArgumentParser(description="Aether v2.15 ViZDoom -> V-JEPA 2 representation cache")
    p.add_argument("dataset")
    p.add_argument("--cache-dir",default=".awa-vizdoom-vjepa-cache")
    p.add_argument("--index",default=None)
    p.add_argument("--manifest",default=None)
    p.add_argument("--model",default=DEFAULT_VJEPA2_MODEL)
    p.add_argument("--device",default=None)
    p.add_argument("--frames",type=int,default=64)
    p.add_argument("--stride",type=int,default=1)
    p.add_argument("--allow-download",action="store_true")
    args=p.parse_args()
    dataset=OfflineTransitionDataset(args.dataset)
    if len(dataset.manifest.observation_shape)!=3 or dataset.manifest.observation_shape[-1] not in (1,3,4):
        raise ValueError("ViZDoom V-JEPA cache expects an HWC pixel dataset; collect with --track pixel")
    backbone=VJEPA2HFBackbone(args.model,local_files_only=not args.allow_download,device=args.device)
    spec=VideoClipSpec(args.frames,args.stride); validate_vjepa_clip_contract(backbone,spec)
    cache=RepresentationCache(args.cache_dir,namespace="vizdoom-vjepa2")
    index_path=Path(args.index) if args.index else Path(args.cache_dir)/"vizdoom_index.json"
    index=precompute_game_video_cache(dataset,backbone,cache,clip_spec=spec,index_path=index_path)
    meta_path=Path(args.dataset).with_suffix(".json")
    meta=json.loads(meta_path.read_text()) if meta_path.exists() else {}
    manifest_path=Path(args.manifest) if args.manifest else Path(args.cache_dir)/"vizdoom_vjepa_manifest.json"
    write_vizdoom_vjepa_manifest(
        manifest_path,backbone=backbone,frame_shape=tuple(dataset.manifest.observation_shape),clip_spec=spec,
        scenario=str(meta.get("scenario","unknown")),dataset_sha256=dataset.manifest.sha256,
        telemetry_dim=(int(np.prod(dataset.manifest.telemetry_shape)) if dataset.manifest.telemetry_shape else 0),
    )
    print(json.dumps({
        "status":"ok","version":V2_VERSION,"dataset":str(args.dataset),"model":args.model,
        "transitions":len(dataset),"feature_dim":index["feature_dim"],"visual_feature_dim":index["visual_feature_dim"],
        "telemetry_dim":index["telemetry_dim"],"hits":index["hits"],"writes":index["writes"],
        "index":str(index_path),"manifest":str(manifest_path),
    },indent=2))


def vizdoom_smoke_main():
    """Dependency-free v2.15 ViZDoom/V-JEPA integration smoke using a DoomGame-compatible fake."""
    from types import SimpleNamespace
    from awa.v2.game.vizdoom_env import ViZDoomConfig, ViZDoomScenario, ViZDoomAetherEnv
    from awa.v2.game.vizdoom_dataset import collect_vizdoom_dataset, ViZDoomExplorerPolicy
    from awa.v2.datasets import OfflineTransitionDataset
    from awa.v2.video_representation import ToyVideoBackbone, VideoClipSpec, precompute_game_video_cache
    from awa.v2.representation import RepresentationCache
    from awa.v2.game.vizdoom_vjepa import vizdoom_pixel_representation_contract

    class FakeGame:
        variable_enums={name:name for name in ("HEALTH","ARMOR","SELECTED_WEAPON","SELECTED_WEAPON_AMMO","KILLCOUNT","ITEMCOUNT","POSITION_X","POSITION_Y","POSITION_Z","ANGLE","PITCH","ATTACK_READY","ON_GROUND","DEAD")}
        def __init__(self):
            self.seed=0; self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self.last_reward=0.; self._buttons=["TURN_LEFT_RIGHT_DELTA","MOVE_FORWARD_BACKWARD_DELTA","ATTACK","USE"]
            self.game_variables={}
        def get_available_buttons(self): return list(self._buttons)
        def set_seed(self,seed): self.seed=int(seed)
        def new_episode(self): self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self.last_reward=0.
        def get_state(self):
            if self.finished: return None
            frame=np.zeros((3,32,48),dtype=np.uint8); frame[0,:,:]=np.uint8(min(255,self.t*25)); frame[1,:,int(self.x)%48:]=80; frame[2,:,:]=40
            return SimpleNamespace(screen_buffer=frame)
        def make_action(self,a,skip):
            self.angle+=float(a[0]); self.x+=max(0.,float(a[1]))*.1; self.t+=1; self.last_reward=float(self.x*.01); self.finished=self.t>=6; return self.last_reward
        def is_episode_finished(self): return self.finished
        def is_player_dead(self): return self.dead
        def get_episode_time(self): return self.t
        def get_game_variable(self,name):
            vals={"HEALTH":100.,"ARMOR":0.,"SELECTED_WEAPON":2.,"SELECTED_WEAPON_AMMO":20.,"KILLCOUNT":0.,"ITEMCOUNT":0.,"POSITION_X":self.x,"POSITION_Y":0.,"POSITION_Z":0.,"ANGLE":self.angle,"PITCH":0.,"ATTACK_READY":1.,"ON_GROUND":1.,"DEAD":float(self.dead)}
            return vals.get(str(name),0.)
        def save(self,path): Path(path).write_text(json.dumps({"t":self.t,"x":self.x,"angle":self.angle,"finished":self.finished}))
        def load(self,path):
            row=json.loads(Path(path).read_text()); self.t=int(row["t"]); self.x=float(row["x"]); self.angle=float(row["angle"]); self.finished=bool(row["finished"])
        def close(self): pass

    cfg=ViZDoomConfig(scenario=ViZDoomScenario.MY_WAY_HOME,track="pixel",frame_skip=1,max_episode_steps=8)
    def factory(): return ViZDoomAetherEnv(cfg,game=FakeGame())
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); data=td/"doom.npz"
        report=collect_vizdoom_dataset(factory,data,episodes=2,horizon=8,base_seed=215,policy=ViZDoomExplorerPolicy(215))
        ds=OfflineTransitionDataset(data); backbone=ToyVideoBackbone(12); backbone.frames_per_clip=4
        spec=VideoClipSpec(4,1); cache=RepresentationCache(td/"cache",namespace="doom-smoke")
        index=precompute_game_video_cache(ds,backbone,cache,clip_spec=spec,index_path=td/"index.json")
        contract=vizdoom_pixel_representation_contract(backbone,frame_shape=tuple(ds.manifest.observation_shape),clip_spec=spec,telemetry_dim=int(np.prod(ds.manifest.telemetry_shape)))
        env=factory(); obs,_=env.reset(seed=215); snap=env.snapshot(); env.step([.1,.8,-1,-1]); restored,ri=env.restore(snap); env.close()
        print(json.dumps({
            "status":"ok","version":V2_VERSION,"dataset":report.to_dict(),"cached_features":index["writes"],
            "feature_dim":index["feature_dim"],"representation_fingerprint":contract.fingerprint,
            "snapshot_shape":list(np.asarray(restored).shape),"snapshot_fidelity":ri["snapshot_fidelity"],
        },indent=2))


def vizdoom_materialize_main():
    """Materialize cached V-JEPA Doom features into the temporal-belief training ABI."""
    from awa.v2.representation import RepresentationCache
    from awa.v2.game.vizdoom_training import materialize_vizdoom_cached_features
    p=argparse.ArgumentParser(description="Aether v2.16 materialize ViZDoom V-JEPA cache")
    p.add_argument("dataset"); p.add_argument("index"); p.add_argument("output")
    p.add_argument("--cache-dir",default=".awa-vizdoom-vjepa-cache")
    args=p.parse_args()
    cache=RepresentationCache(args.cache_dir,namespace="vizdoom-vjepa2")
    out=materialize_vizdoom_cached_features(args.dataset,cache,args.index,args.output)
    ds=OfflineTransitionDataset(out)
    print(json.dumps({"status":"ok","version":V2_VERSION,"output":str(out),"transitions":len(ds),"observation_dim":int(ds.manifest.observation_shape[0])},indent=2))


def vizdoom_train_main():
    """Train the unified v2.16 temporal-belief Doom world + hybrid actor stack."""
    from awa.v2.game.vizdoom_training import train_vizdoom_stack
    p=argparse.ArgumentParser(description="Aether v2.16 unified ViZDoom trainer")
    p.add_argument("dataset"); p.add_argument("--out-dir",default="runs/v2_16_vizdoom")
    p.add_argument("--track",choices=["structured","pixel"],default="structured")
    p.add_argument("--device",default="cpu"); p.add_argument("--precision",choices=["fp32","bf16","fp16"],default="fp32")
    p.add_argument("--sequence-length",type=int,default=8); p.add_argument("--world-epochs",type=int,default=2)
    p.add_argument("--actor-epochs",type=int,default=4); p.add_argument("--batch-size",type=int,default=64); p.add_argument("--hidden",type=int,default=128)
    args=p.parse_args()
    rep=train_vizdoom_stack(args.dataset,args.out_dir,track=args.track,sequence_length=args.sequence_length,world_epochs=args.world_epochs,actor_epochs=args.actor_epochs,batch_size=args.batch_size,hidden=args.hidden,device=args.device,precision=args.precision)
    print(json.dumps({"status":"ok","version":V2_VERSION,**rep.to_dict()},indent=2))


def vizdoom_campaign_main():
    """Plan or execute the durable v2.16 ViZDoom campaign."""
    import yaml
    from awa.v2.game.vizdoom_env import ViZDoomScenario, ViZDoomConfig, ViZDoomAetherEnv
    from awa.v2.game.vizdoom_campaign import ViZDoomTrainingCampaign
    from awa.v2.video_representation import VJEPA2HFBackbone, VideoClipSpec
    p=argparse.ArgumentParser(description="Aether v2.16 durable ViZDoom training campaign")
    p.add_argument("--config",default="configs/v2_16_vizdoom_training.yaml"); p.add_argument("--out-dir",default="runs/v2_16_vizdoom_campaign")
    p.add_argument("--execute",action="store_true"); p.add_argument("--stop-after-stage",default=None); p.add_argument("--allow-download",action="store_true")
    args=p.parse_args(); cfg=yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    if not args.execute:
        print(json.dumps({"status":"plan-only","version":V2_VERSION,"track":cfg.get("track","structured"),"stages":cfg.get("stages",[]),"note":"pass --execute to collect/train"},indent=2)); return
    env_cfg=cfg.get("environment") or {}; track=str(cfg.get("track","structured"))
    def factory(scenario,requested_track):
        return ViZDoomAetherEnv(ViZDoomConfig(scenario=ViZDoomScenario(str(scenario)),track=requested_track,frame_skip=int(env_cfg.get("frame_skip",4)),max_episode_steps=env_cfg.get("max_episode_steps"),visible=bool(env_cfg.get("visible",False)),sound=bool(env_cfg.get("sound",False))))
    backbone=None; spec=VideoClipSpec(int((cfg.get("video") or {}).get("frames",64)),int((cfg.get("video") or {}).get("stride",1)))
    if track=="pixel":
        vcfg=cfg.get("video") or {}
        backbone=VJEPA2HFBackbone(vcfg.get("model","facebook/vjepa2-vitl-fpc64-256"),local_files_only=not args.allow_download,device=(cfg.get("runtime") or {}).get("device"))
    campaign=ViZDoomTrainingCampaign(cfg,args.out_dir,env_factory=factory,video_backbone=backbone,clip_spec=spec)
    result=campaign.run(stop_after_stage=args.stop_after_stage)
    print(json.dumps({"version":V2_VERSION,**result},indent=2,default=str))


def vizdoom_training_closure_smoke_main():
    """Dependency-free proof of the complete pixel Doom -> belief/world/hybrid-action path."""
    from types import SimpleNamespace
    from awa.v2.game.vizdoom_env import ViZDoomConfig, ViZDoomScenario, ViZDoomAetherEnv
    from awa.v2.game.vizdoom_dataset import collect_vizdoom_dataset, ViZDoomExplorerPolicy
    from awa.v2.game.vizdoom_training import materialize_vizdoom_cached_features, train_vizdoom_stack
    from awa.v2.game.vizdoom_runtime import load_vizdoom_runtime
    from awa.v2.video_representation import ToyVideoBackbone, VideoClipSpec, precompute_game_video_cache
    from awa.v2.representation import RepresentationCache
    class FakeGame:
        variable_enums={name:name for name in ("HEALTH","ARMOR","SELECTED_WEAPON","SELECTED_WEAPON_AMMO","KILLCOUNT","ITEMCOUNT","POSITION_X","POSITION_Y","POSITION_Z","ANGLE","PITCH","ATTACK_READY","ON_GROUND","DEAD")}
        def __init__(self): self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False; self._buttons=["TURN_LEFT_RIGHT_DELTA","MOVE_FORWARD_BACKWARD_DELTA","ATTACK","USE"]
        def get_available_buttons(self): return list(self._buttons)
        def set_seed(self,seed): self.seed=int(seed)
        def new_episode(self): self.t=0; self.x=0.; self.angle=0.; self.finished=False; self.dead=False
        def get_state(self):
            if self.finished:return None
            f=np.zeros((3,20,28),dtype=np.uint8); f[0]=np.uint8(min(255,self.t*30)); f[1,:,min(27,int(self.x)):] = 90; f[2]=40; return SimpleNamespace(screen_buffer=f)
        def make_action(self,a,skip): self.angle+=float(a[0]); self.x+=max(0.,float(a[1]))*.12; self.t+=1; self.finished=self.t>=8; return float(self.x*.02 + (1.0 if self.finished else 0.0))
        def is_episode_finished(self): return self.finished
        def is_player_dead(self): return self.dead
        def get_episode_time(self): return self.t
        def get_game_variable(self,name):
            vals={"HEALTH":100.,"ARMOR":0.,"SELECTED_WEAPON":2.,"SELECTED_WEAPON_AMMO":20.,"KILLCOUNT":0.,"ITEMCOUNT":0.,"POSITION_X":self.x,"POSITION_Y":0.,"POSITION_Z":0.,"ANGLE":self.angle,"PITCH":0.,"ATTACK_READY":1.,"ON_GROUND":1.,"DEAD":0.}
            return vals.get(str(name),0.)
        def close(self): pass
    cfg=ViZDoomConfig(scenario=ViZDoomScenario.MY_WAY_HOME,track="pixel",frame_skip=1,max_episode_steps=10)
    def factory(): return ViZDoomAetherEnv(cfg,game=FakeGame())
    with tempfile.TemporaryDirectory() as td:
        td=Path(td); raw=td/"doom.npz"; collect_vizdoom_dataset(factory,raw,episodes=3,horizon=10,base_seed=216,policy=ViZDoomExplorerPolicy(216))
        ds=OfflineTransitionDataset(raw); bb=ToyVideoBackbone(10); bb.frames_per_clip=4; spec=VideoClipSpec(4,1); cache=RepresentationCache(td/"cache",namespace="vizdoom-vjepa2")
        idx=precompute_game_video_cache(ds,bb,cache,clip_spec=spec,index_path=td/"index.json",batch_size=4)
        features=materialize_vizdoom_cached_features(raw,cache,idx,td/"features.npz")
        rep=train_vizdoom_stack(features,td/"trained",track="pixel",sequence_length=3,world_epochs=1,actor_epochs=1,calibration_epochs=1,batch_size=8,hidden=24,seed=216)
        runtime=load_vizdoom_runtime(td/"trained/vizdoom_world.pt",td/"trained/vizdoom_actor.pt",video_backbone=bb,clip_spec=spec,use_planner=False,risk_limit=1.0)
        env=factory(); obs,_=env.reset(seed=216); runtime.reset(); action,info=runtime.act(env,obs); env.close()
        print(json.dumps({"status":"ok","version":V2_VERSION,"track":"pixel","transitions":rep.transitions,"belief_dim":rep.belief_dim,"cached_writes":idx["writes"],"action":action.tolist(),"binary_exact":bool(all(abs(float(x))==1.0 for x in action[2:4])),"decision":info.to_dict()},indent=2))


def empirical_closure_main():
    """Build a fail-closed v2.17 empirical evidence bundle from JSONL run records."""
    from awa.v2.empirical_closure import load_jsonl, build_evidence_bundle
    p=argparse.ArgumentParser(description="Aether v2.17 empirical closure evidence builder")
    p.add_argument("records",help="JSONL RunRecord stream")
    p.add_argument("--output",default="aether-v2.17-evidence.json")
    p.add_argument("--primary-metric",default="success_rate")
    p.add_argument("--minimum-seeds",type=int,default=5)
    args=p.parse_args()
    rows=load_jsonl(args.records)
    report=build_evidence_bundle(rows,primary_metric=args.primary_metric,minimum_seeds=args.minimum_seeds)
    Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["qualification"]["status"],"version":V2_VERSION,"output":args.output,"runs":len(rows),"failures":report["qualification"]["failures"]},indent=2))


def evolving_exploration_smoke_main():
    """Dependency-free v2.20 DREAM-RSI-style replay + adaptive-compute smoke."""
    from awa.v2.meta_exploration import (
        DiscoveryTree, DiscoveryNode, EvidenceClass, DeclarativeExplorationPolicy,
        ReplaySimulatorPool, ReplayWorld, DeclarativePolicyMutator, MetaPolicyOptimizer,
    )
    from awa.v2.reasoning import AdaptiveReasoningEffortController, EffortSignals

    tree=DiscoveryTree("smoke-world")
    def add(node_id,parent,index,quality,novelty,transfer=0.0,info=0.0):
        tree.add(DiscoveryNode(
            node_id=node_id,parent_id=parent,iteration=index+1,generation_index=index,
            evidence=EvidenceClass.OBSERVED,
            metrics={"quality":quality,"novelty":novelty,"transfer":transfer,
                     "information_gain":info,"uncertainty_reduction":info/2,
                     "constraint_violations":0.0},
            compute_cost=.05, provenance_sha256=f"{index+1:064x}"[-64:],
        ))
    add("branch-a","root",0,.1,.8,info=.2); add("branch-b","root",1,.4,.1,info=.1)
    add("branch-a-continue","branch-a",0,.9,.3,.7,.3); add("branch-b-continue","branch-b",0,.45,.1,.1,.1)
    pool=ReplaySimulatorPool([ReplayWorld(tree)])
    active=DeclarativeExplorationPolicy("active",quality_weight=0.0,novelty_weight=-1.0,new_world_bias=.2,stop_threshold=-1.0,max_parallel=1)
    proposal=MetaPolicyOptimizer(pool,DeclarativePolicyMutator(step=1.5)).propose(active,max_rounds=3)
    effort=AdaptiveReasoningEffortController().choose(
        EffortSignals(.4,.7,.3,.5,.2,.6,model_reliability=.9,resource_pressure=.1),
        [0.0,.10,.25,.31,.32],
        predicted_risks=[.3,.28,.25,.24,.23],
        predicted_latencies_ms=[0,2,5,9,16],
    )
    print(json.dumps({
        "status":"ok","version":V2_VERSION,
        "grounded_nodes":len(tree.grounded_nodes()),
        "active_policy":active.policy_id,
        "candidate_policy":proposal.candidate.policy_id,
        "replay_gain":proposal.replay_gain,
        "promotion_required":True,
        "effort":effort.to_dict(),
    },indent=2))


def evolving_campaign_main():
    """Plan or execute the v2.26 autonomous cumulative milestone-validation loop."""
    import yaml
    from awa.v2.evolving_campaign import (
        load_history_evidence,
        orchestrator_from_config,
        runner_from_config,
    )

    p = argparse.ArgumentParser(description="Aether v2.26 evolving campaign orchestrator")
    p.add_argument("--config", default="configs/v2_25_reduction_empirical_closure.yaml")
    p.add_argument("--out-dir", default="runs/v2_25_evolving_campaign")
    p.add_argument("--history-records", default=None, help="Optional JSONL RunRecord history to import before planning")
    p.add_argument("--history-provenance", default=None, help="JSON mapping run_id -> RunProvenance for --history-records")
    p.add_argument("--max-rounds", type=int, default=None)
    p.add_argument("--execute", action="store_true", help="Run the configured campaign adapter; otherwise plan only")
    p.add_argument("--bootstrap-active", action="store_true", help="When history is empty, execute the active-policy bootstrap matrix before dreaming")
    p.add_argument("--rerun-existing", action="store_true", help="Re-run validation cells even if identical evidence is already committed")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    orchestrator = orchestrator_from_config(cfg, args.out_dir)
    if (args.history_records is None) != (args.history_provenance is None):
        raise ValueError("--history-records and --history-provenance must be supplied together")
    if args.history_records is not None:
        imported = orchestrator.import_history(load_history_evidence(args.history_records, args.history_provenance))
    else:
        imported = 0

    max_rounds = int(args.max_rounds or (((cfg.get("meta_exploration") or {}).get("replay") or {}).get("max_rounds", 64)))
    report = orchestrator.plan_iteration(max_rounds=max_rounds)
    runner = None
    bootstrap_receipt = None
    if report.get("status") == "NEEDS_BOOTSTRAP" and args.execute and args.bootstrap_active:
        runner = runner_from_config(cfg)
        bootstrap_receipt = orchestrator.bootstrap_active(runner, rerun_existing=args.rerun_existing)
        report = orchestrator.plan_iteration(max_rounds=max_rounds)
    if not args.execute or report.get("status") in {"NO_CHANGE", "NEEDS_BOOTSTRAP"}:
        print(json.dumps({
            "status": report.get("status"),
            "version": V2_VERSION,
            "mode": "plan",
            "imported_history": imported,
            "bootstrap_receipt": bootstrap_receipt,
            "out_dir": str(Path(args.out_dir)),
            "iteration": report.get("iteration"),
            "proposal": report.get("proposal"),
            "bootstrap_cells": len((report.get("bootstrap_plan") or {}).get("cells", [])),
            "validation_cells": len((report.get("validation_plan") or {}).get("cells", [])),
        }, indent=2))
        return

    runner = runner or runner_from_config(cfg)
    receipt = orchestrator.execute_planned_iteration(runner, rerun_existing=args.rerun_existing)
    print(json.dumps({
        "status": receipt.status,
        "version": V2_VERSION,
        "mode": "execute",
        "out_dir": str(Path(args.out_dir)),
        "bootstrap_receipt": bootstrap_receipt,
        "receipt": receipt.to_dict(),
    }, indent=2))


def evolving_campaign_smoke_main():
    """Dependency-free v2.21 proof of exact paired execution, protocol closure and promotion."""
    from awa.v2.empirical_closure import RunRecord
    from awa.v2.evidence_integrity import RunProvenance
    from awa.v2.evolving_campaign import (
        CampaignEvidence,
        EvolvingCampaignOrchestrator,
        ValidationMatrix,
    )
    from awa.v2.meta_exploration import DeclarativeExplorationPolicy, EvidenceClass

    h1, h2, h3, h4 = "1" * 64, "2" * 64, "3" * 64, "4" * 64

    def evidence(system, seed, task, split, transitions, score, *, novelty=0.0, info=0.0):
        metrics = {
            "success_rate": float(score), "episode_return": float(score),
            "constraint_violations": 0.0, "planner_calls_per_episode": 1.0,
            "world_model_calls_per_episode": 4.0, "inference_latency_ms": 2.0,
            "wall_clock_seconds": 10.0, "novelty": float(novelty),
            "information_gain": float(info), "uncertainty_reduction": float(info) / 2.0,
            "normalized_compute_cost": .01,
        }
        record = RunRecord(system, seed, task, split, transitions, metrics, h1, h2)
        run_id = "|".join(map(str, (system, seed, task, split, transitions)))
        provenance = RunProvenance(run_id, h1, h2, h3, h4, (f"scenario-{split}",))
        return CampaignEvidence(record, provenance, EvidenceClass.VALIDATED)

    history = [
        evidence("branch-a", 7, "task-a", "heldout", 10, .10, novelty=.90, info=.20),
        evidence("branch-a", 7, "task-a", "heldout", 20, 1.00, novelty=.30, info=.30),
        evidence("branch-b", 7, "task-b", "heldout", 10, .50, novelty=.10, info=.10),
        evidence("branch-b", 7, "task-b", "heldout", 20, .60, novelty=.10, info=.10),
    ]

    class SmokeRunner:
        def __init__(self): self.calls = 0
        def run(self, cell, policy, work_dir):
            self.calls += 1
            score = .75 if cell.policy_id != "active" else .60
            return evidence(cell.policy_id, cell.seed, cell.task, cell.split, cell.transitions, score)

    with tempfile.TemporaryDirectory() as td:
        active = DeclarativeExplorationPolicy(
            "active", quality_weight=0.0, novelty_weight=-1.0,
            new_world_bias=.2, stop_threshold=-1.0, max_parallel=1,
        )
        validation = ValidationMatrix(
            seeds=(1, 2, 3, 4, 5), tasks=("doom",), splits=("heldout",),
            milestones=(25_000,), minimum_seeds=5,
        )
        orchestrator = EvolvingCampaignOrchestrator(
            td, active_policy=active, validation=validation,
            mutation_step=1.5, maximum_seed_regression=.2,
        )
        orchestrator.import_history(history)
        plan = orchestrator.plan_iteration(max_rounds=3)
        runner = SmokeRunner()
        receipt = orchestrator.execute_planned_iteration(runner)
        print(json.dumps({
            "status": "ok" if receipt.status == "PASS" else "fail",
            "version": V2_VERSION,
            "proposal": plan["proposal"],
            "validation_cells": runner.calls,
            "protocol_status": receipt.protocol_status,
            "promotion_status": receipt.promotion_status,
            "active_policy_after": receipt.active_policy_after,
            "receipt": receipt.to_dict(),
        }, indent=2))


def native_campaign_smoke_main():
    """Run two tiny split-disjoint cells through the v2.26 cached native path."""
    from awa.v2.evolving_campaign import CampaignCell
    from awa.v2.meta_exploration import DeclarativeExplorationPolicy
    from awa.v2.native_campaign import NativeProceduralCampaignRunner, NativeProceduralRunnerConfig

    cfg = NativeProceduralRunnerConfig(
        device="cpu", precision="fp32", curriculum_stages=(1, 2, 3, 10, 12),
        tasks_per_batch=4, episodes_per_task=1, horizon=35,
        train_difficulty=.25, eval_difficulty=.35, eval_tasks=2,
        sequence_length=2, hidden=16, world_epochs=1, actor_epochs=1,
        calibration_epochs=0, batch_size=16, one_step_aux_epochs=0,
        minimum_transitions=32,
    )
    with tempfile.TemporaryDirectory() as td:
        runner = NativeProceduralCampaignRunner(cfg)
        policy = DeclarativeExplorationPolicy("active")
        root = Path(td) / "runs"
        heldout = runner.run(CampaignCell("active", 101, "procedural", "heldout", 64), policy, root / "heldout")
        transfer = runner.run(CampaignCell("active", 101, "procedural", "transfer", 64), policy, root / "transfer")
        print(json.dumps({
            "status": "ok",
            "version": V2_VERSION,
            "evidence_format": heldout.to_dict()["format"],
            "heldout_cell": heldout.cell.to_dict(),
            "transfer_cell": transfer.cell.to_dict(),
            "heldout_metrics": heldout.record.metrics,
            "transfer_metrics": transfer.record.metrics,
            "same_checkpoint": heldout.record.checkpoint_sha256 == transfer.record.checkpoint_sha256,
            "same_training_dataset": heldout.provenance.dataset_sha256 == transfer.provenance.dataset_sha256,
            "disjoint_scenarios": set(heldout.provenance.scenario_ids).isdisjoint(set(transfer.provenance.scenario_ids)),
            "checkpoint_sha256": heldout.record.checkpoint_sha256,
            "dataset_sha256": heldout.provenance.dataset_sha256,
            "source_sha256": heldout.provenance.source_sha256,
        }, indent=2))


def milestone_scorecard_main():
    """Build the paired milestone scorecard from empirical RunRecord JSONL."""
    from awa.v2.empirical_closure import load_jsonl
    from awa.v2.milestone_analysis import (
        MilestonePromotionConfig,
        build_milestone_scorecard,
        evaluate_milestone_promotion,
    )

    p = argparse.ArgumentParser(description="Aether paired milestone empirical scorecard")
    p.add_argument("--records", required=True, help="JSONL file of RunRecord rows")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)
    p.add_argument("--primary-metric", default="success_rate")
    p.add_argument("--minimum-seeds", type=int, default=5)
    p.add_argument("--resamples", type=int, default=4000)
    p.add_argument("--minimum-final-mean-gain", type=float, default=0.0)
    p.add_argument("--minimum-curve-mean-gain", type=float, default=0.0)
    p.add_argument("--maximum-seed-final-regression", type=float, default=0.05)
    p.add_argument("--maximum-split-final-regression", type=float, default=0.02)
    p.add_argument("--require-positive-final-ci", action="store_true")
    p.add_argument("--output", default=None)
    args = p.parse_args()

    scorecard = build_milestone_scorecard(
        load_jsonl(args.records),
        baseline=args.baseline,
        candidate=args.candidate,
        primary_metric=args.primary_metric,
        minimum_seeds=args.minimum_seeds,
        resamples=args.resamples,
    )
    promotion = evaluate_milestone_promotion(
        scorecard,
        MilestonePromotionConfig(
            minimum_final_mean_gain=args.minimum_final_mean_gain,
            minimum_curve_mean_gain=args.minimum_curve_mean_gain,
            maximum_seed_final_regression=args.maximum_seed_final_regression,
            maximum_split_final_regression=args.maximum_split_final_regression,
            require_positive_final_ci=args.require_positive_final_ci,
            bootstrap_resamples=args.resamples,
        ),
    )
    payload = {"version": V2_VERSION, "scorecard": scorecard, "promotion": promotion.to_dict()}
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def ablation_protocol_main():
    """Emit the preregistered v2.26 component-ablation matrix and config hashes."""
    from awa.v2.ablation import build_empirical_ablation_protocol
    p = argparse.ArgumentParser(description="Aether v2.26 preregistered component ablation protocol")
    p.add_argument("--seeds", default="1701,1702,1703,1704,1705")
    p.add_argument("--milestones", default="25000,100000")
    p.add_argument("--output", default=None)
    args = p.parse_args()
    seeds=tuple(int(x) for x in args.seeds.split(",") if x.strip())
    milestones=tuple(int(x) for x in args.milestones.split(",") if x.strip())
    protocol=build_empirical_ablation_protocol(seeds=seeds,milestones=milestones,minimum_seeds=min(5,len(seeds)))
    payload={"version":V2_VERSION,"protocol":protocol.to_dict(),"protocol_sha256":protocol.sha256}
    if args.output:
        Path(args.output).write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print(json.dumps(payload,indent=2,sort_keys=True))


def autonomous_collection_smoke_main():
    """Prove coverage bootstrap -> frozen Aether actor collection on real procedural transitions."""
    from awa.v2.evolving_campaign import CampaignCell
    from awa.v2.meta_exploration import DeclarativeExplorationPolicy
    from awa.v2.native_campaign import NativeProceduralCampaignRunner, NativeProceduralRunnerConfig
    cfg=NativeProceduralRunnerConfig(
        device="cpu",precision="fp32",tasks_per_batch=4,episodes_per_task=1,horizon=12,eval_tasks=1,
        sequence_length=1,hidden=16,world_epochs=1,actor_epochs=1,calibration_epochs=0,one_step_aux_epochs=0,
        batch_size=8,minimum_transitions=16,collection_mode="aether_actor",bootstrap_collection_mode="coverage",
        collector_planner_budget=4,collector_planner_horizon=2,collector_planner_stride=3,
    )
    with tempfile.TemporaryDirectory() as td:
        root=Path(td)/"run"; runner=NativeProceduralCampaignRunner(cfg); policy=DeclarativeExplorationPolicy("active")
        runner.run(CampaignCell("active",7,"procedural","heldout",16),policy,root/"m16")
        runner.run(CampaignCell("active",7,"procedural","heldout",32),policy,root/"m32")
        receipts=[]
        for rp in (root/"_training_cache").glob("*/collection_receipt.json"):
            receipts.append(json.loads(rp.read_text()))
        receipts.sort(key=lambda x:int(x["retained_transitions"]))
        print(json.dumps({
            "status":"ok","version":V2_VERSION,
            "first_mode":receipts[0]["batches"][0]["actual_collection_mode"],
            "second_mode":receipts[-1]["batches"][0]["actual_collection_mode"],
            "first_transitions":receipts[0]["retained_transitions"],
            "second_transitions":receipts[-1]["retained_transitions"],
        },indent=2,sort_keys=True))


def executable_ablation_main():
    """Run one concrete v2.27 system variant on a grounded procedural dataset."""
    import argparse, json
    from pathlib import Path
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import collect_game_dataset
    from awa.v2.game.collectors import CoverageArenaPolicy
    from awa.v2.game.variant_runtime import VARIANT_BY_ID, train_and_evaluate_variant
    from awa.v2.meta_exploration import DeclarativeExplorationPolicy
    from awa.v2.native_campaign import ExplorationPolicyCurriculumAdapter

    ap=argparse.ArgumentParser(description="Execute one Aether v2.27 ablation variant")
    ap.add_argument("--variant",choices=sorted(VARIANT_BY_ID),default="world_actor")
    ap.add_argument("--dataset",default=None)
    ap.add_argument("--out-dir",required=True)
    ap.add_argument("--seed",type=int,default=1701)
    ap.add_argument("--device",default="cpu")
    ap.add_argument("--episodes-per-task",type=int,default=1)
    ap.add_argument("--horizon",type=int,default=60)
    ap.add_argument("--world-epochs",type=int,default=2)
    ap.add_argument("--actor-epochs",type=int,default=4)
    ap.add_argument("--representation-epochs",type=int,default=2)
    ap.add_argument("--voc-epochs",type=int,default=40)
    args=ap.parse_args()
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    variant=VARIANT_BY_ID[args.variant]
    factory=ProceduralTaskFactory(args.seed)
    curriculum_mode="adaptive" if variant.adaptive_curriculum else "fixed"
    if variant.adaptive_curriculum:
        policy=DeclarativeExplorationPolicy("ablation-full")
        allocation=ExplorationPolicyCurriculumAdapter().allocate(policy,tuple(range(1,13)),12)
        train_tasks=[]; idx=0
        for stage,count in sorted(allocation.items()):
            for _ in range(count):
                train_tasks.append(factory.make(stage,0.45,idx,"v227-ablation-adaptive")); idx+=1
    else:
        train_tasks=[factory.make(stage,0.45,stage,"v227-ablation-fixed") for stage in range(1,13)]
    eval_tasks=[factory.make(stage,0.65,50_000+stage,"v227-ablation-heldout") for stage in (2,4,6,9,11,12)]
    dataset=Path(args.dataset) if args.dataset else out/"grounded_training.npz"
    if args.dataset is None:
        collect_game_dataset(train_tasks,dataset,episodes_per_task=args.episodes_per_task,horizon=args.horizon,policy=CoverageArenaPolicy(seed=args.seed))
    tr,ev=train_and_evaluate_variant(args.variant,dataset,train_tasks,eval_tasks,out/args.variant,device=args.device,world_epochs=args.world_epochs,actor_epochs=args.actor_epochs,representation_epochs=args.representation_epochs,voc_epochs=args.voc_epochs,seed=args.seed,horizon=args.horizon,curriculum_mode=curriculum_mode)
    print(json.dumps({"status":"ok","training":tr.to_dict(),"evaluation":ev.to_dict()},indent=2,sort_keys=True))

def executable_ablation_smoke_main():
    """Grounded miniature proving all seven v2.27 variants have executable runtimes."""
    import json, tempfile
    from pathlib import Path
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import collect_game_dataset
    from awa.v2.game.collectors import CoverageArenaPolicy
    from awa.v2.game.variant_runtime import SYSTEM_VARIANTS, train_and_evaluate_variant
    from awa.v2.meta_exploration import DeclarativeExplorationPolicy
    from awa.v2.native_campaign import ExplorationPolicyCurriculumAdapter

    with tempfile.TemporaryDirectory(prefix="awa-v227-ablation-") as td:
        root=Path(td); factory=ProceduralTaskFactory(227)
        fixed_tasks=[factory.make(s,0.30,s,"v227-smoke-fixed") for s in (1,4,10)]
        policy=DeclarativeExplorationPolicy("smoke-full")
        allocation=ExplorationPolicyCurriculumAdapter().allocate(policy,tuple(range(1,13)),4)
        adaptive_tasks=[]; idx=0
        for stage,count in sorted(allocation.items()):
            for _ in range(count):
                adaptive_tasks.append(factory.make(stage,0.30,200+idx,"v227-smoke-adaptive")); idx+=1
        eval_tasks=[factory.make(s,0.40,100+s,"v227-smoke-eval") for s in (2,11)]
        fixed_dataset=root/"fixed.npz"; adaptive_dataset=root/"adaptive.npz"
        collect_game_dataset(fixed_tasks,fixed_dataset,episodes_per_task=1,horizon=10,policy=CoverageArenaPolicy(seed=227))
        collect_game_dataset(adaptive_tasks,adaptive_dataset,episodes_per_task=1,horizon=10,policy=CoverageArenaPolicy(seed=228))
        rows=[]
        for variant in SYSTEM_VARIANTS:
            mode="adaptive" if variant.adaptive_curriculum else "fixed"
            ds=adaptive_dataset if variant.adaptive_curriculum else fixed_dataset
            tasks=adaptive_tasks if variant.adaptive_curriculum else fixed_tasks
            tr,ev=train_and_evaluate_variant(variant.variant_id,ds,tasks,eval_tasks,root/variant.variant_id,device="cpu",sequence_length=1,hidden=16,world_epochs=1,actor_epochs=1,representation_epochs=1,calibration_epochs=0,batch_size=16,seed=227,horizon=10,voc_epochs=2,curriculum_mode=mode)
            rows.append({"variant":variant.variant_id,"success_rate":ev.success_rate,"planner_calls":ev.planner_calls_per_episode,"auxiliary_rows":tr.auxiliary_rows,"curriculum_mode":mode})
        print(json.dumps({"status":"ok","variants":rows},indent=2,sort_keys=True))


def ablation_campaign_main():
    """Plan or execute the resumable v2.30 25K/100K ablation matrix."""
    from awa.v2.ablation_campaign import AblationCampaignConfig, AblationCampaignRunner

    ap = argparse.ArgumentParser(description="Aether v2.31 resumable ablation campaign")
    ap.add_argument("--config", default="configs/v2_31_system_split.yaml")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--execute", action="store_true", help="Execute pending jobs; otherwise print the frozen plan/status")
    ap.add_argument("--max-jobs", type=int, default=None, help="Bound jobs completed by this invocation")
    ap.add_argument("--worker-index", type=int, default=0, help="Deterministic shard index for parallel workers")
    ap.add_argument("--worker-count", type=int, default=1, help="Number of deterministic campaign shards")
    args = ap.parse_args()
    cfg = AblationCampaignConfig.from_yaml(args.config)
    runner = AblationCampaignRunner(cfg, args.out_dir)
    if not args.execute:
        payload = {
            "status": "plan_only", "version": V2_VERSION,
            "plan": runner.plan.to_dict() | {"plan_sha256": runner.plan.sha256},
            "campaign": runner.status(),
        }
    else:
        payload = {"version": V2_VERSION, **runner.execute(
            max_jobs=args.max_jobs, worker_index=args.worker_index, worker_count=args.worker_count
        )}
    print(json.dumps(payload, indent=2, sort_keys=True))


def ablation_report_main():
    """Rebuild consolidated protocol and keep/remove report from completed v2.30 jobs."""
    from awa.v2.ablation_campaign import AblationCampaignConfig, AblationCampaignRunner

    ap = argparse.ArgumentParser(description="Aether v2.31 ablation report")
    ap.add_argument("--config", default="configs/v2_31_system_split.yaml")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    runner = AblationCampaignRunner(AblationCampaignConfig.from_yaml(args.config), args.out_dir)
    payload = {"version": V2_VERSION, "campaign": runner.status(), "consolidated": runner.consolidate()}
    print(json.dumps(payload, indent=2, sort_keys=True))


def ablation_campaign_smoke_main():
    """Dependency-light smoke of plan freezing, sharding, resume and decision gating."""
    from awa.v2.ablation_campaign import AblationCampaignConfig, AblationCampaignRunner

    cfg = AblationCampaignConfig(
        seeds=(11, 12), milestones=(32,), systems=("actor_only", "belief_actor"),
        device="cpu", horizon=8, hidden=16, sequence_length=1,
        world_epochs=1, actor_epochs=1, representation_epochs=1,
        calibration_epochs=0, batch_size=16, voc_epochs=1,
        eval_tasks_per_split=2, minimum_seeds=2, bootstrap_resamples=100,
    )
    with tempfile.TemporaryDirectory(prefix="awa-v230-campaign-") as td:
        runner = AblationCampaignRunner(cfg, td)
        first = runner.execute(max_jobs=1)
        second = runner.execute(max_jobs=1)
        status = runner.status()
        print(json.dumps({
            "status": "ok", "version": V2_VERSION,
            "training_jobs": len(runner.plan.jobs),
            "first_completed": first["completed_now"], "second_completed": second["completed_now"],
            "completed_jobs": status["completed_jobs"], "pending_jobs": status["pending_jobs"],
        }, indent=2, sort_keys=True))



def empirical_status_main():
    """Render the fail-closed empirical maturity ladder from actual evidence artifacts."""
    from awa.v2.empirical_status import build_empirical_status, empirical_status_markdown

    ap = argparse.ArgumentParser(description="Aether v2.31 empirical status")
    ap.add_argument("--evidence-root", default=None)
    ap.add_argument("--json-output", default=None)
    ap.add_argument("--markdown-output", default=None)
    args = ap.parse_args()
    payload = build_empirical_status(args.evidence_root)
    if args.json_output:
        Path(args.json_output).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = empirical_status_markdown(payload)
    if args.markdown_output:
        Path(args.markdown_output).write_text(md, encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


def compositional_benchmark_main():
    """Generate the factor-isolated/heldout compositional benchmark manifest."""
    from awa.v2.curriculum.environment_factory import FactorizedEnvironmentFactory

    ap = argparse.ArgumentParser(description="Aether v2.31 compositional task benchmark")
    ap.add_argument("--output", required=True)
    ap.add_argument("--base-seed", type=int, default=2300)
    ap.add_argument("--train-replicates", type=int, default=2)
    ap.add_argument("--eval-replicates", type=int, default=1)
    args = ap.parse_args()
    benchmark = FactorizedEnvironmentFactory(args.base_seed).build_canonical_benchmark(
        train_replicates=args.train_replicates,
        eval_replicates=args.eval_replicates,
    )
    benchmark.write(args.output)
    print(json.dumps({
        "status": "ok",
        "version": V2_VERSION,
        "sha256": benchmark.sha256,
        "train_tasks": len(benchmark.train),
        "heldout_tasks": len(benchmark.heldout),
        "transfer_tasks": len(benchmark.transfer),
        "output": str(Path(args.output)),
    }, indent=2, sort_keys=True))


def vizdoom_real_qualification_main():
    """Run a qualification that refuses fake/injected ViZDoom and V-JEPA backends."""
    from awa.v2.game.vizdoom_qualification import (
        RealViZDoomQualificationConfig,
        run_real_vizdoom_qualification,
    )

    ap = argparse.ArgumentParser(description="Aether v2.31 real ViZDoom qualification")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scenario", default="my_way_home")
    ap.add_argument("--track", choices=("structured", "pixel"), default="structured")
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--horizon", type=int, default=600)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--seed", type=int, default=2300)
    ap.add_argument("--visible", action="store_true")
    ap.add_argument("--require-vjepa", action="store_true")
    ap.add_argument("--vjepa-model", default="facebook/vjepa2-vitl-fpc64-256")
    ap.add_argument("--allow-vjepa-download", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    cfg = RealViZDoomQualificationConfig(
        scenario=args.scenario,
        track=args.track,
        episodes=args.episodes,
        horizon=args.horizon,
        frame_skip=args.frame_skip,
        seed=args.seed,
        visible=args.visible,
        require_vjepa=args.require_vjepa,
        vjepa_model=args.vjepa_model,
        vjepa_local_files_only=not args.allow_vjepa_download,
        device=args.device,
    )
    payload = run_real_vizdoom_qualification(cfg, args.out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))


def planner_hardware_benchmark_main():
    """Measure matched logical planner work under different real execution schedules."""
    from awa.v2.game.procedural_runtime import load_procedural_game_stack
    from awa.v2.planner_hardware_benchmark import (
        PlannerHardwareBenchmarkConfig,
        benchmark_planner_schedules,
        write_hardware_benchmark,
    )

    ap = argparse.ArgumentParser(description="Aether v2.31 planner hardware schedule benchmark")
    ap.add_argument("--world-checkpoint", required=True)
    ap.add_argument("--actor-checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--candidates", type=int, default=128)
    ap.add_argument("--horizon", type=int, default=8)
    ap.add_argument("--batch-limits", default="0,32,8,4,1")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=30)
    ap.add_argument("--seed", type=int, default=2300)
    ap.add_argument("--allow-cpu", action="store_true")
    args = ap.parse_args()
    stack = load_procedural_game_stack(
        args.world_checkpoint,
        args.actor_checkpoint,
        device=args.device,
    )
    belief = torch.zeros(1, int(stack.world.belief_dim), device=stack.device)
    cfg = PlannerHardwareBenchmarkConfig(
        candidates=args.candidates,
        horizon=args.horizon,
        batch_limits=tuple(int(x) for x in args.batch_limits.split(",") if x.strip()),
        warmup=args.warmup,
        repeats=args.repeats,
        seed=args.seed,
    )
    report = benchmark_planner_schedules(
        stack.world,
        stack.actor.actor,
        belief,
        action_low=[-1.0] * stack.world.action_dim,
        action_high=[1.0] * stack.world.action_dim,
        config=cfg,
        require_cuda=not args.allow_cpu,
    )
    write_hardware_benchmark(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True))
