from __future__ import annotations
import argparse
import json
from pathlib import Path
import tempfile
import torch

from awa.config import load_config
from awa.utils import seed_everything, resolve_device
from awa.training.trainer import train, evaluate, build_components
from awa.connectome.graph import compile_edge_csv
from awa.connectome.motifs import graph_summary
from awa.connectome.official import compile_official_connections, detect_connection_columns


def smoke_main():
    repo_cfg = Path(__file__).parents[2] / "configs" / "v1_minimal.yaml"
    cfg = load_config(repo_cfg)
    cfg["device"] = "cpu"; cfg["training"]["steps"] = 48; cfg["training"]["batch_size"] = 4
    cfg["training"]["sequence_length"] = 3; cfg["training"]["checkpoint_every"] = 1000
    cfg["planner"]["enabled"] = False; cfg["model"]["latent_dim"] = 16
    cfg["model"]["deterministic_dim"] = 32; cfg["model"]["stochastic_dim"] = 8
    cfg["model"]["hidden_dim"] = 32; cfg["training"]["imagination_horizon"] = 3
    seed_everything(cfg["seed"]); device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as td:
        comps = train(cfg, device, td); result = evaluate(cfg, comps, device, episodes=3)
    print(json.dumps({"status": "ok", "evaluation": result}, indent=2))



def continuous_smoke_main():
    repo_cfg = Path(__file__).parents[2] / "configs" / "v1_6_continuous.yaml"
    cfg = load_config(repo_cfg)
    cfg["device"] = "cpu"
    cfg["training"]["steps"] = 40
    cfg["training"]["batch_size"] = 2
    cfg["training"]["sequence_length"] = 3
    cfg["training"]["burn_in"] = 0
    cfg["training"]["checkpoint_every"] = 1000
    cfg["training"]["imagination_horizon"] = 3
    cfg["planner"]["enabled"] = False
    cfg["model"]["latent_dim"] = 8
    cfg["model"]["deterministic_dim"] = 16
    cfg["model"]["stochastic_dim"] = 4
    cfg["model"]["hidden_dim"] = 24
    cfg["model"]["slow_enabled"] = False
    seed_everything(cfg["seed"]); device = torch.device("cpu")
    with tempfile.TemporaryDirectory() as td:
        comps = train(cfg, device, td); result = evaluate(cfg, comps, device, episodes=2)
    print(json.dumps({"status":"ok","continuous":True,"evaluation":result}, indent=2))

def train_main():
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--out", default="runs/default")
    p.add_argument("--resume")
    args = p.parse_args(); cfg = load_config(args.config); seed_everything(cfg["seed"])
    device = resolve_device(cfg.get("device", "auto")); comps = train(cfg, device, args.out, args.resume)
    print(json.dumps(evaluate(cfg, comps, device), indent=2))


def eval_main():
    from awa.evaluation.checkpoint import load_components_checkpoint
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--checkpoint"); p.add_argument("--episodes",type=int,default=10); p.add_argument("--planner",action="store_true")
    args=p.parse_args(); cfg=load_config(args.config); seed_everything(cfg["seed"]); device=resolve_device(cfg.get("device","auto")); comps=build_components(cfg,device)
    if args.checkpoint: load_components_checkpoint(args.checkpoint,comps,device)
    print(json.dumps(evaluate(cfg,comps,device,episodes=args.episodes,use_planner=args.planner),indent=2))


def connectome_main():
    p = argparse.ArgumentParser(); p.add_argument("--edges", help="Connection CSV")
    p.add_argument("--official", action="store_true", help="Auto-detect common MaleCNS/FlyWire connection column aliases")
    args = p.parse_args()
    if not args.edges: p.print_help(); return
    graph = compile_official_connections(args.edges) if args.official else compile_edge_csv(args.edges)
    payload = graph_summary(graph)
    if args.official: payload["columns"] = detect_connection_columns(args.edges)
    print(json.dumps(payload, indent=2))


def benchmark_main():
    from awa.evaluation.benchmark import run_multiseed
    p = argparse.ArgumentParser(); p.add_argument("--config", required=True); p.add_argument("--seeds", default="1,2,3,4,5")
    p.add_argument("--out", default="runs/benchmark"); p.add_argument("--episodes", type=int, default=25)
    args = p.parse_args(); cfg = load_config(args.config); seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    print(json.dumps(run_multiseed(cfg, seeds, args.out, args.episodes).summary(), indent=2))


def robustness_main():
    from awa.evaluation.robustness import evaluate_robustness
    from awa.evaluation.checkpoint import load_components_checkpoint
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--checkpoint")
    args=p.parse_args(); cfg=load_config(args.config); seed_everything(cfg["seed"]); device=resolve_device(cfg.get("device","auto")); comps=build_components(cfg,device)
    if args.checkpoint: load_components_checkpoint(args.checkpoint,comps,device)
    print(json.dumps(evaluate_robustness(cfg,comps,device,args.episodes),indent=2))



def suite_main():
    from awa.evaluation.suite import run_generalization_suite
    from awa.evaluation.checkpoint import load_components_checkpoint
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--episodes",type=int,default=20); p.add_argument("--checkpoint")
    args=p.parse_args(); cfg=load_config(args.config); seed_everything(cfg["seed"]); device=resolve_device(cfg.get("device","auto")); comps=build_components(cfg,device)
    if args.checkpoint: load_components_checkpoint(args.checkpoint,comps,device)
    print(json.dumps(run_generalization_suite(cfg,comps,device,args.episodes),indent=2))


def ablation_main():
    from awa.evaluation.ablation import load_result_files, write_markdown_report
    p=argparse.ArgumentParser(); p.add_argument("results",nargs="+"); p.add_argument("--out",default="ablation_report.md"); p.add_argument("--title",default="AWA Ablation Report")
    args=p.parse_args(); report=write_markdown_report(args.out,args.title,load_result_files(args.results)); print(json.dumps(report,indent=2))



def matrix_main():
    import yaml
    from awa.evaluation.matrix import run_matrix
    p=argparse.ArgumentParser(); p.add_argument("--config",required=True); p.add_argument("--variants",required=True,help="YAML mapping of variant name -> config patch")
    p.add_argument("--seeds",default="1,2,3,4,5"); p.add_argument("--episodes",type=int,default=20); p.add_argument("--out",default="runs/matrix")
    args=p.parse_args(); cfg=load_config(args.config); variants=yaml.safe_load(Path(args.variants).read_text(encoding="utf-8")); seeds=[int(x) for x in args.seeds.split(",") if x.strip()]
    print(json.dumps(run_matrix(cfg,variants,seeds,args.out,args.episodes),indent=2))


def provenance_main():
    from awa.connectome.provenance import build_provenance_manifest, write_manifest
    p=argparse.ArgumentParser(); p.add_argument("files",nargs="+"); p.add_argument("--dataset",default="MaleCNS"); p.add_argument("--version",default="unknown"); p.add_argument("--out",default="provenance.json")
    args=p.parse_args(); manifest=build_provenance_manifest(args.files,args.dataset,args.version); write_manifest(args.out,manifest); print(json.dumps(manifest,indent=2))



def calibrate_main():
    from awa.evaluation.checkpoint import load_components_checkpoint
    from awa.planning.auto_calibrate import collect_transition_uncertainty, fit_calibrator_from_errors
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint'); p.add_argument('--episodes',type=int,default=5); p.add_argument('--out',default='runs/calibration.pt'); p.add_argument('--quantile',type=float,default=.75)
    args=p.parse_args(); cfg=load_config(args.config); device=resolve_device(cfg.get('device','auto')); comps=build_components(cfg,device)
    if comps.calibrator is None: raise SystemExit('planner.calibrated_uncertainty must be true')
    if args.checkpoint: load_components_checkpoint(args.checkpoint,comps,device)
    u,e=collect_transition_uncertainty(cfg,comps,device,args.episodes); report=fit_calibrator_from_errors(comps.calibrator,u,e,args.quantile)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'calibrator':comps.calibrator.state_dict(),'report':report.as_dict()},args.out)
    print(json.dumps(report.as_dict(),indent=2))


def gate_fit_main():
    from awa.evaluation.checkpoint import load_components_checkpoint
    from awa.evaluation.planning_benefit import collect_planning_benefit, fit_arbitrator_dataset
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--checkpoint'); p.add_argument('--episodes',type=int,default=5); p.add_argument('--epochs',type=int,default=100); p.add_argument('--branch-horizon',type=int,default=3); p.add_argument('--max-samples',type=int,default=512); p.add_argument('--planning-cost-per-1k',type=float,default=0.0); p.add_argument('--out',default='runs/arbitrator.pt')
    args=p.parse_args(); cfg=load_config(args.config); device=resolve_device(cfg.get('device','auto')); comps=build_components(cfg,device)
    if comps.arbitrator is None: raise SystemExit('planner.learned_gate must be true')
    if args.checkpoint: load_components_checkpoint(args.checkpoint,comps,device)
    data=collect_planning_benefit(cfg,comps,device,args.episodes,max_samples=args.max_samples,branch_horizon=args.branch_horizon,planning_cost_per_1k=args.planning_cost_per_1k); report=fit_arbitrator_dataset(comps.arbitrator,data,args.epochs)
    Path(args.out).parent.mkdir(parents=True,exist_ok=True); torch.save({'arbitrator':comps.arbitrator.state_dict(),'report':report,'samples':len(data)},args.out)
    print(json.dumps({'samples':len(data),'branch_horizon':data.branch_horizon,**report},indent=2))


def env_check_main():
    from awa.evaluation.env_qualification import qualify_environment
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--steps',type=int,default=8)
    args=p.parse_args(); cfg=load_config(args.config); print(json.dumps(qualify_environment(cfg,args.steps),indent=2))


def env_multiseed_main():
    from awa.evaluation.env_qualification import qualify_environment_multiseed
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--steps',type=int,default=16); p.add_argument('--seeds',default='1,2,3,4,5')
    args=p.parse_args(); cfg=load_config(args.config); seeds=[int(x) for x in args.seeds.split(',') if x.strip()]
    print(json.dumps(qualify_environment_multiseed(cfg,seeds,args.steps),indent=2))
