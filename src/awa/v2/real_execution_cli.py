from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from awa.v2 import __version__ as V2_VERSION
from awa.v2.research_os.execution_preflight import (
    PreflightRequirement,
    run_execution_preflight,
    write_execution_preflight,
)
from awa.v2.research_os.real_vizdoom_campaign import (
    build_real_vizdoom_campaign_plan,
    run_real_vizdoom_training_campaign,
)


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def execution_preflight_main() -> None:
    parser = argparse.ArgumentParser(description="Aether v2.36 real-execution hardware/dependency preflight")
    parser.add_argument("--config", default="configs/v2_35_empirical_execution.yaml")
    parser.add_argument("--out-dir", default="runs/v2_35_execution")
    args = parser.parse_args()

    payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    preflight = dict(payload.get("preflight") or {})
    requirement = PreflightRequirement(**preflight)
    report = run_execution_preflight(args.out_dir, requirement)
    report["release"] = V2_VERSION
    report["config_path"] = str(Path(args.config).resolve())
    report["config_sha256"] = _sha256(args.config)
    target = Path(args.out_dir) / "execution_preflight.json"
    write_execution_preflight(report, target)
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["ready"]:
        raise SystemExit(2)


def real_vizdoom_campaign_main() -> None:
    parser = argparse.ArgumentParser(description="Aether v2.36 fail-closed real ViZDoom training campaign")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stop-after-stage", default=None)
    parser.add_argument("--allow-vjepa-download", action="store_true")
    args = parser.parse_args()

    if not args.execute:
        plan = build_real_vizdoom_campaign_plan(args.config, args.out_dir)
        print(json.dumps({"release": V2_VERSION, "status": "plan_only", **plan}, indent=2, sort_keys=True))
        return

    receipt = run_real_vizdoom_training_campaign(
        args.config,
        args.out_dir,
        stop_after_stage=args.stop_after_stage,
        allow_vjepa_download=args.allow_vjepa_download,
    )
    print(json.dumps({"release": V2_VERSION, "status": "ok", **receipt}, indent=2, sort_keys=True, default=str))


def vizdoom_branch_replay_main() -> None:
    from awa.v2.game.vizdoom_env import ViZDoomAetherEnv, ViZDoomConfig, ViZDoomScenario
    from awa.v2.research_os.execution_preflight import _vizdoom_ipc_probe
    from awa.v2.research_os.branch_replay import qualify_vizdoom_reset_replay

    parser = argparse.ArgumentParser(description="Check real my_way_home reset/branch reproducibility for P1P")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="9101,9102,9103,9104,9105,9106")
    parser.add_argument("--frame-skip", type=int, default=4)
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]
    ipc_available, ipc_error = _vizdoom_ipc_probe()
    if not ipc_available:
        report = {"format": "awa-v2.38.6-vizdoom-reset-replay-v1", "status": "BLOCKED",
                  "qualified": False, "real_backend_checked": False,
                  "seeds": seeds, "probes": [], "results": [], "release": V2_VERSION,
                  "reason": f"ViZDoom local IPC unavailable: {ipc_error}"}
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2, sort_keys=True))
        raise SystemExit(2)

    def make_env() -> ViZDoomAetherEnv:
        return ViZDoomAetherEnv(ViZDoomConfig(
            scenario=ViZDoomScenario.MY_WAY_HOME, track="structured", frame_skip=args.frame_skip))

    report = qualify_vizdoom_reset_replay(make_env, seeds=seeds, require_real_backend=True)
    report["release"] = V2_VERSION
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "PASS":
        raise SystemExit(2)


def vizdoom_planner_collect_main() -> None:
    """Measure P1P on real ViZDoom, refusing unsupported local IPC hosts."""
    from awa.v2.game.vizdoom_env import ViZDoomAetherEnv, ViZDoomConfig, ViZDoomScenario
    from awa.v2.game.vizdoom_runtime import load_vizdoom_runtime
    from awa.v2.research_os.execution_preflight import _vizdoom_ipc_probe
    from awa.v2.research_os.planner_collection import PlannerBranchCollector, maximum_branch_steps

    parser = argparse.ArgumentParser(description="Collect real P1P planner branches and closed-loop episodes")
    parser.add_argument("--world-checkpoint", required=True)
    parser.add_argument("--actor-checkpoint", required=True)
    parser.add_argument("--replay-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", default="9101,9102,9103,9104,9105,9106")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-episode-steps", type=int, default=525)
    parser.add_argument("--candidates", type=int, default=16)
    parser.add_argument("--max-branch-steps", type=int, default=30_000_000)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    seeds = [int(x) for x in args.seeds.split(",")]
    if len(set(seeds)) < 6 or args.candidates < 16 or not 1 <= args.max_episode_steps <= 525:
        parser.error("P1P needs six distinct seeds, at least 16 candidates, and 1–525 episode steps")
    projected = maximum_branch_steps(len(seeds), args.candidates, args.max_episode_steps)
    if projected > args.max_branch_steps:
        raise SystemExit(f"BLOCKED: worst-case branch work {projected} exceeds cap {args.max_branch_steps}")
    if not args.execute:
        print(json.dumps({"status": "PLAN_ONLY", "worst_case_branch_steps": projected,
                          "max_branch_steps": args.max_branch_steps,
                          "closed_loop_episodes": len(seeds) * (1 + 3 * 6 * 2 + 1),
                          "note": "Use --execute on a qualified GPU host; full P2 qualification requires 525 episode steps."},
                         sort_keys=True))
        return
    ipc_available, ipc_error = _vizdoom_ipc_probe()
    if not ipc_available:
        raise SystemExit(f"BLOCKED: ViZDoom local IPC unavailable: {ipc_error}")
    if args.device.startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise SystemExit("BLOCKED: CUDA device requested but unavailable")
    controller = load_vizdoom_runtime(args.world_checkpoint, args.actor_checkpoint,
                                      device=args.device, use_planner=False)
    if controller.track != "structured":
        raise SystemExit("BLOCKED: P1P requires the frozen structured P1 checkpoint")

    def make_env():
        return ViZDoomAetherEnv(ViZDoomConfig(
            scenario=ViZDoomScenario.MY_WAY_HOME, track="structured", frame_skip=4))

    collector = PlannerBranchCollector(
        make_env, controller, seeds=seeds,
        world_checkpoint=args.world_checkpoint, actor_checkpoint=args.actor_checkpoint,
        replay_report=args.replay_report, max_episode_steps=args.max_episode_steps,
        candidates=args.candidates, require_real_backend=True)
    result = collector.collect(args.output)
    print(json.dumps({"status": "MEASURED", "format": result["format"],
                      "seed_ids": result["seed_ids"], "output": str(Path(args.output).resolve())},
                     sort_keys=True))


def real_evidence_bundle_main() -> None:
    from awa.v2.research_os.real_campaign_evidence import (
        EvidenceBundleConfig,
        build_real_execution_evidence_bundle,
        write_real_execution_evidence_bundle,
    )

    parser = argparse.ArgumentParser(description="Aether v2.36 bind real-execution evidence into one fail-closed bundle")
    parser.add_argument("--preflight", default=None)
    parser.add_argument("--training-receipt", default=None)
    parser.add_argument("--planner-benchmark", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-planner", action="store_true")
    args = parser.parse_args()
    payload = build_real_execution_evidence_bundle(
        preflight_path=args.preflight,
        training_receipt_path=args.training_receipt,
        planner_benchmark_path=args.planner_benchmark,
        config=EvidenceBundleConfig(require_planner_benchmark=args.require_planner),
    )
    write_real_execution_evidence_bundle(payload, args.output)
    print(json.dumps({"release": V2_VERSION, **payload}, indent=2, sort_keys=True))
    if not payload["qualified"]:
        raise SystemExit(2)


def external_baseline_compare_main() -> None:
    from awa.v2.research_os.external_baseline import (
        MatchedBaselineProtocol,
        compare_external_baseline,
        load_baseline_records,
        write_baseline_comparison,
    )

    parser = argparse.ArgumentParser(description="Aether v2.36 matched external-baseline comparison")
    parser.add_argument("--aether-records", required=True)
    parser.add_argument("--baseline-records", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--required-pairs", type=int, default=5)
    parser.add_argument("--max-accelerator-ratio", type=float, default=1.25)
    args = parser.parse_args()
    protocol = MatchedBaselineProtocol(
        required_pairs=args.required_pairs,
        max_accelerator_ratio=args.max_accelerator_ratio,
    )
    report = compare_external_baseline(
        load_baseline_records(args.aether_records),
        load_baseline_records(args.baseline_records),
        protocol,
    )
    write_baseline_comparison(report, args.output)
    print(json.dumps({"release": V2_VERSION, **report}, indent=2, sort_keys=True))
    if report["status"] != "QUALIFIED_COMPARISON":
        raise SystemExit(2)


def vizdoom_reward_diagnostic_main() -> None:
    from awa.v2.research_os.reward_diagnostic import run_reward_diagnostic

    parser = argparse.ArgumentParser(description="Aether v2.38.6 paired my_way_home raw-vs-clipped reward diagnostic")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--execute", action="store_true", help="required to collect, train, and evaluate on real ViZDoom")
    args = parser.parse_args()
    if not args.execute:
        payload = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        print(json.dumps({"release": V2_VERSION, "status": "plan_only", "config": payload,
                          "out_dir": str(Path(args.out_dir).resolve()),
                          "claim_boundary": "No collection or training is performed without --execute."}, indent=2, sort_keys=True))
        return
    report = run_reward_diagnostic(args.config, args.out_dir)
    print(json.dumps({"release": V2_VERSION, **report}, indent=2, sort_keys=True, allow_nan=False))
    if report.get("status") != "PASS":
        raise SystemExit(2)
