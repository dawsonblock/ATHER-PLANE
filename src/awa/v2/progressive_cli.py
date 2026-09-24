from __future__ import annotations

import argparse
import json
from pathlib import Path

from awa.v2 import __version__ as V2_VERSION
from awa.v2.research_os.progressive_capability import (
    HorizonGateConfig,
    build_progressive_capability_report,
    load_progressive_protocol,
    qualify_world_model_horizon,
    write_horizon_gate,
    write_progressive_capability_report,
)


def progressive_capability_main() -> None:
    parser = argparse.ArgumentParser(description="Aether v2.38 progressive empirical capability gate")
    parser.add_argument("--config", default="configs/v2_38_progressive_capability.yaml")
    parser.add_argument("--evidence-root", default="runs/v2_38_6")
    parser.add_argument("--output", default="runs/v2_38_6/progressive_capability_report.json")
    args = parser.parse_args()
    protocol = load_progressive_protocol(args.config)
    report = build_progressive_capability_report(args.evidence_root, protocol=protocol)
    write_progressive_capability_report(report, args.output)
    print(json.dumps({"release": V2_VERSION, **report}, indent=2, sort_keys=True))


def world_model_horizon_gate_main() -> None:
    parser = argparse.ArgumentParser(description="Aether v2.38 held-out world-model horizon reliability gate")
    parser.add_argument("--curve", required=True, help="JSON file containing a list or {'points': [...]} horizon measurements")
    parser.add_argument("--output", default="runs/world_model_horizon_gate.json")
    parser.add_argument("--max-error", type=float, default=0.20)
    parser.add_argument("--max-calibration-error", type=float, default=0.10)
    parser.add_argument("--minimum-samples", type=int, default=256)
    parser.add_argument("--minimum-horizon", type=int, default=4)
    args = parser.parse_args()
    payload = json.loads(Path(args.curve).read_text(encoding="utf-8"))
    points = payload.get("points", []) if isinstance(payload, dict) else payload
    config = HorizonGateConfig(
        max_normalized_prediction_error=args.max_error,
        max_calibration_error=args.max_calibration_error,
        minimum_samples=args.minimum_samples,
        minimum_qualified_horizon=args.minimum_horizon,
    )
    report = qualify_world_model_horizon(points, config)
    write_horizon_gate(report, args.output)
    print(json.dumps({"release": V2_VERSION, **report}, indent=2, sort_keys=True))
    if not report["qualified"]:
        raise SystemExit(2)


def reward_diagnostic_gate_main() -> None:
    from awa.v2.research_os.planner_diagnostic import write_reward_diagnostic_report

    parser = argparse.ArgumentParser(description="Aether v2.38.6 paired reward-scale evidence gate")
    parser.add_argument("--input", required=True, help="raw P1D paired episode/component evidence JSON")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = write_reward_diagnostic_report(args.input, args.output)
    print(json.dumps({"release": V2_VERSION, **report}, indent=2, sort_keys=True))
    if report.get("status") != "PASS":
        raise SystemExit(2)


def planner_diagnostic_gate_main() -> None:
    from awa.v2.research_os.planner_diagnostic import write_planner_diagnostic_report

    parser = argparse.ArgumentParser(description="Aether v2.38.6 planner real-branch evidence gate")
    parser.add_argument("--input", required=True, help="raw fixed-seed P1P real-branch comparison JSON")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/v2_38_6_planner_diagnostic.yaml")
    args = parser.parse_args()
    import yaml
    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    thresholds = dict(payload.get("diagnostic") or {})
    report = write_planner_diagnostic_report(args.input, args.output, **thresholds)
    print(json.dumps({"release": V2_VERSION, **report}, indent=2, sort_keys=True))
    if report.get("status") != "PASS":
        raise SystemExit(2)



def progressive_executor_main() -> None:
    parser = argparse.ArgumentParser(description="Aether v2.38 resumable one-phase progressive empirical executor")
    parser.add_argument("--config", default="configs/v2_38_progressive_executor.yaml")
    parser.add_argument("--evidence-root", default="runs/v2_38_6")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-transition-target", type=int, default=None)
    parser.add_argument("--execute-next", action="store_true")
    parser.add_argument("--allow-expensive", action="store_true")
    args = parser.parse_args()
    from awa.v2.research_os.progressive_executor import (
        build_next_phase_execution_plan,
        execute_next_progressive_phase,
    )
    if not args.execute_next:
        payload = build_next_phase_execution_plan(
            executor_config=args.config,
            evidence_root=args.evidence_root,
            repo_root=args.repo_root,
            max_transition_target=args.max_transition_target,
        )
    else:
        payload = execute_next_progressive_phase(
            executor_config=args.config,
            evidence_root=args.evidence_root,
            repo_root=args.repo_root,
            allow_expensive=args.allow_expensive,
            max_transition_target=args.max_transition_target,
        )
    print(json.dumps({"release": V2_VERSION, **payload}, indent=2, sort_keys=True))
    if args.execute_next and payload.get("execution", {}).get("status") != "PASS":
        raise SystemExit(2)


def runpod_bootstrap_main() -> None:
    parser = argparse.ArgumentParser(description="Print the shipped Aether v2.38 RunPod bootstrap path and next commands")
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    script = root / "deploy" / "runpod" / "bootstrap.sh"
    payload = {
        "release": V2_VERSION,
        "bootstrap_script": str(script),
        "exists": script.exists(),
        "next_plan_command": "awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_6",
        "next_execute_command": "awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_6 --execute-next",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not script.exists():
        raise SystemExit(2)
