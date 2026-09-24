"""Resumable one-phase-at-a-time execution planner for Aether v2.38.

The executor deliberately does not invent missing scientific measurements.  It
reads the v2.38 progressive capability report, prepares the next eligible argv
sequence, and optionally executes exactly one phase with ``shell=False``.  High
cost phases require an additional explicit acknowledgement.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable

import yaml

from .progressive_capability import build_progressive_capability_report, load_progressive_protocol


_ALLOWED_COMMANDS = {
    "awa-v2-execution-preflight",
    "awa-v2-vizdoom-real-campaign",
    "awa-v2-ablation-campaign",
    "awa-v2-ablation-report",
    "awa-v2-world-model-horizon-gate",
    "awa-v2-planner-hardware-benchmark",
    "awa-v2-dream-rsi-campaign",
    "awa-v2-reward-diagnostic-gate",
    "awa-v2-planner-diagnostic-gate",
    "awa-v2-vizdoom-reward-diagnostic",
}


@dataclass(frozen=True)
class ExecutorLimits:
    default_max_transition_target: int = 25_000
    execute_one_phase_per_invocation: bool = True
    require_expensive_ack_for: tuple[str, ...] = ("P11", "P14")


@dataclass(frozen=True)
class CommandSpec:
    argv: tuple[str, ...]
    description: str

    def to_dict(self) -> dict[str, Any]:
        return {"argv": list(self.argv), "description": self.description}


def _sha(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_executor_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("progressive executor config must decode to a mapping")
    execution = payload.get("execution") or {}
    if not isinstance(execution, dict):
        raise ValueError("executor execution section must be a mapping")
    raw_limits = dict(payload.get("limits") or {})
    limits = ExecutorLimits(
        default_max_transition_target=int(raw_limits.get("default_max_transition_target", 25_000)),
        execute_one_phase_per_invocation=bool(raw_limits.get("execute_one_phase_per_invocation", True)),
        require_expensive_ack_for=tuple(str(x) for x in raw_limits.get("require_expensive_ack_for", ["P11", "P14"])),
    )
    progressive_path = Path(str(payload.get("progressive_config", "configs/v2_37_progressive_capability.yaml")))
    if not progressive_path.is_absolute():
        progressive_path = (config_path.parent.parent / progressive_path).resolve()
    return {
        "format": "awa-v2.38-progressive-executor-config-v1",
        "release": str(payload.get("release", "2.38.6")),
        "path": str(config_path),
        "execution": execution,
        "limits": limits,
        "progressive_config": str(progressive_path),
        "claim_boundary": list(payload.get("claim_boundary") or []),
    }


def _latest_json(root: Path, name: str) -> tuple[Path, dict[str, Any]] | None:
    candidates: list[tuple[float, Path, dict[str, Any]]] = []
    for path in root.rglob(name):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(payload, dict):
            continue
        try:
            stamp = path.stat().st_mtime
        except OSError:
            stamp = 0.0
        candidates.append((stamp, path, payload))
    if not candidates:
        return None
    _, path, payload = max(candidates, key=lambda row: (row[0], str(row[1])))
    return path, payload


def _structured_stage(phase_id: str) -> str:
    return {
        "P1": "real-wiring-2k",
        "P2": "real-navigation-10k",
        "P3": "real-control-25k",
    }[phase_id]


def _focused_ablation(phase_id: str) -> tuple[str, str]:
    return {
        "P4": ("configs/v2_38_ablation_temporal_memory.yaml", "temporal_memory"),
        "P6": ("configs/v2_38_ablation_fixed_planner.yaml", "fixed_planner"),
        "P7": ("configs/v2_38_ablation_adaptive_compute.yaml", "adaptive_compute"),
    }[phase_id]


def _validate_commands(commands: list[CommandSpec]) -> None:
    for command in commands:
        if not command.argv:
            raise ValueError("empty command argv")
        if command.argv[0] not in _ALLOWED_COMMANDS:
            raise ValueError(f"command is outside the v2.38 executor allowlist: {command.argv[0]}")
        if any("\n" in token or "\x00" in token for token in command.argv):
            raise ValueError("command argv contains an invalid control character")


def _planner_checkpoints(root: Path) -> tuple[str, str] | None:
    candidates = []
    for path in root.rglob("real_vizdoom_training_receipt.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8"))
            if (receipt.get("track") != "structured" or
                receipt.get("checkpoint_registry_verified") is not True or
                (receipt.get("stable_checkpoint") or {}).get("stage") != "real-control-25k"):
                continue
            rows = (receipt.get("result") or {}).get("results") or []
            if not any(row.get("stage") == "real-control-25k" and row.get("promoted") is True for row in rows):
                continue
            artifacts = receipt.get("stable_checkpoint_artifacts") or {}
            resolved = []
            for key in ("world", "actor"):
                artifact = artifacts.get(key) or {}
                checkpoint = Path(str(artifact.get("path", "")))
                expected = str(artifact.get("sha256", ""))
                if not checkpoint.is_file() or len(expected) != 64:
                    break
                digest = hashlib.sha256()
                with checkpoint.open("rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
                if digest.hexdigest() != expected:
                    break
                resolved.append(str(checkpoint))
            if len(resolved) == 2:
                candidates.append((path.stat().st_mtime, path.as_posix(), tuple(resolved)))
        except (OSError, ValueError, TypeError, json.JSONDecodeError, AttributeError):
            continue
    if candidates:
        return max(candidates)[2]
    return None


def build_next_phase_execution_plan(
    *,
    executor_config: str | Path,
    evidence_root: str | Path,
    repo_root: str | Path | None = None,
    max_transition_target: int | None = None,
) -> dict[str, Any]:
    """Build an argv-only plan for the first blocked progressive phase."""
    cfg = load_executor_config(executor_config)
    repo = Path(repo_root or Path(cfg["path"]).parent.parent).resolve()
    evidence = Path(evidence_root).resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    protocol = load_progressive_protocol(cfg["progressive_config"])
    progress = build_progressive_capability_report(evidence, protocol=protocol)
    next_action = progress.get("next_action")
    if next_action is None:
        payload = {
            "format": "awa-v2.38-progressive-execution-plan-v1",
            "status": "COMPLETE",
            "phase_id": None,
            "commands": [],
            "progressive_report": progress,
            "claim_boundary": "All configured gates are satisfied; no command is scheduled automatically.",
        }
        return payload | {"plan_sha256": _sha(payload)}

    phase_id = str(next_action["id"])
    execution_meta = dict(cfg["execution"].get(phase_id) or {})
    transition_target = int(execution_meta.get("transition_target", 0) or 0)
    limit = int(max_transition_target if max_transition_target is not None else cfg["limits"].default_max_transition_target)
    if transition_target > limit:
        payload = {
            "format": "awa-v2.38-progressive-execution-plan-v1",
            "status": "BUDGET_BLOCKED",
            "phase_id": phase_id,
            "transition_target": transition_target,
            "max_transition_target": limit,
            "commands": [],
            "progressive_report": progress,
            "reason": "phase transition target exceeds the invocation budget guard",
        }
        return payload | {"plan_sha256": _sha(payload)}

    commands: list[CommandSpec] = []
    manual: str | None = None
    if phase_id == "P0":
        commands.append(CommandSpec((
            "awa-v2-execution-preflight", "--config", str(repo / "configs/v2_38_6_empirical_execution.yaml"),
            "--out-dir", str(evidence / "execution"),
        ), "Qualify CUDA, ViZDoom, Transformers and the writable execution host."))
    elif phase_id in {"P1", "P2", "P3"}:
        prior = _latest_json(evidence / "structured", "real_vizdoom_training_receipt.json")
        rows = ((prior[1].get("result") or {}).get("results") or []) if prior else []
        legacy = next((row for row in rows if row.get("stage") == _structured_stage(phase_id)
                       and not row.get("comparison_contract") and phase_id != "P1"), None)
        if legacy is not None:
            manual = (f"The cached {_structured_stage(phase_id)} training verdict used the v2.38.0 cross-scenario "
                      "comparison. Archive this evidence and use a fresh v2.38.6 evidence root for requalification; "
                      "cached training phases cannot be silently rescored.")
        else:
            commands.append(CommandSpec((
                "awa-v2-vizdoom-real-campaign", "--config", str(repo / "configs/v2_35_vizdoom_structured.yaml"),
                "--out-dir", str(evidence / "structured"), "--execute", "--stop-after-stage", _structured_stage(phase_id),
            ), f"Execute/resume real structured ViZDoom through {_structured_stage(phase_id)}."))
    elif phase_id == "P1D":
        commands.append(CommandSpec((
            "awa-v2-vizdoom-reward-diagnostic", "--config",
            str(repo / "configs/v2_38_6_reward_diagnostic.yaml"), "--out-dir",
            str(evidence / "reward_diagnostic"), "--execute",
        ), "Collect one shared my_way_home dataset, train paired raw/clipped rewards, evaluate fixed seeds, and apply the behavioral gate."))
    elif phase_id == "P1P":
        raw = evidence / str(execution_meta.get("input", "planner_diagnostic_raw.json"))
        if not raw.is_file():
            manual = (
                f"Run the fixed-seed P1P real-branch comparison and save candidate-level and episode-level measurements to {raw}. "
                "Include P1P-A through P1P-D, horizons 1/2/4/8/16/32, actor_seeded and mixed proposals, actual "
                "ViZDoom realized returns and candidate action sequences/origins. "
                "First run awa-v2-vizdoom-branch-replay on the real host and verify every fixed branch is "
                "reproducible from reset. B compares joint learned-dynamics/learned-reward predictions against real returns."
            )
        else:
            commands.append(CommandSpec((
                "awa-v2-planner-diagnostic-gate", "--config", str(repo / "configs/v2_38_6_planner_diagnostic.yaml"),
                "--input", str(raw), "--output",
                str(evidence / str(execution_meta.get("required_output", "planner_diagnostic.json"))),
            ), "Compute fixed-seed candidate ranking, objective ablations, horizon prefix, and proposal-diversity report."))
    elif phase_id in {"P4", "P6", "P7"}:
        rel_cfg, mechanism = _focused_ablation(phase_id)
        out = evidence / "focused_ablation" / mechanism
        abs_cfg = str(repo / rel_cfg)
        commands.extend([
            CommandSpec(("awa-v2-ablation-campaign", "--config", abs_cfg, "--out-dir", str(out), "--execute"),
                        f"Execute/resume the focused five-seed {mechanism} campaign."),
            CommandSpec(("awa-v2-ablation-report", "--config", abs_cfg, "--out-dir", str(out)),
                        f"Consolidate the focused {mechanism} keep/remove evidence."),
        ])
    elif phase_id == "P5":
        curve = evidence / "world_model_horizon_curve.json"
        if not curve.exists():
            manual = f"Create held-out horizon measurements at {curve}; no prediction curve is fabricated by the executor."
        else:
            commands.append(CommandSpec((
                "awa-v2-world-model-horizon-gate", "--curve", str(curve),
                "--output", str(evidence / "world_model_horizon_gate.json"),
            ), "Qualify the longest contiguous trustworthy world-model planning horizon."))
    elif phase_id == "P8":
        commands.append(CommandSpec((
            "awa-v2-vizdoom-real-campaign", "--config", str(repo / "configs/v2_35_vizdoom_pixel.yaml"),
            "--out-dir", str(evidence / "pixel"), "--execute", "--stop-after-stage", "real-pixel-navigation-5k",
        ), "Execute/resume the genuine RGB + frozen V-JEPA real ViZDoom campaign."))
    elif phase_id == "P9":
        pair = _planner_checkpoints(evidence)
        if pair is None:
            manual = "Stable structured world/actor checkpoint paths could not be resolved from real training receipts."
        else:
            world, actor = pair
            horizon_item = _latest_json(evidence, "world_model_horizon_gate.json")
            horizon = 8
            if horizon_item is not None:
                horizon = max(1, int(horizon_item[1].get("selected_horizon", 8)))
            commands.append(CommandSpec((
                "awa-v2-planner-hardware-benchmark", "--world-checkpoint", world, "--actor-checkpoint", actor,
                "--device", "cuda", "--batch-limits", "0,32,8,4,1", "--horizon", str(horizon),
                "--output", str(evidence / "planner_schedule_gpu.json"),
            ), "Measure action-equivalent planner schedules on the physical CUDA device."))
    elif phase_id == "P11":
        commands.append(CommandSpec((
            "awa-v2-dream-rsi-campaign", "--config", str(repo / "configs/v2_34_dream_rsi_empirical.yaml"),
            "--out-dir", str(evidence / "dream_rsi"), "--execute",
        ), "Execute/resume the canonical paired five-seed fixed-vs-DREAM-RSI campaign."))
    elif phase_id == "P14":
        out = evidence / "full_ablation"
        abs_cfg = str(repo / "configs/v2_31_system_split.yaml")
        commands.extend([
            CommandSpec(("awa-v2-ablation-campaign", "--config", abs_cfg, "--out-dir", str(out), "--execute"),
                        "Execute/resume the full seven-system five-seed 25K/100K campaign."),
            CommandSpec(("awa-v2-ablation-report", "--config", abs_cfg, "--out-dir", str(out)),
                        "Consolidate the final seven-system ablation and compute reports."),
        ])
    elif phase_id in {"P10", "P12", "P13", "P15"}:
        required = execution_meta.get("required_output") or {
            "P10": "compositional_report.json", "P12": "transfer_report.json",
            "P13": "external_baseline_comparison.json", "P15": "scale_authorization.json",
        }[phase_id]
        manual = f"This phase requires an independently produced evidence artifact: {evidence / str(required)}"
    else:
        manual = f"No v2.38 automatic executor mapping exists for {phase_id}."

    _validate_commands(commands)
    expensive = phase_id in set(cfg["limits"].require_expensive_ack_for)
    status = "MANUAL_REQUIRED" if manual else "READY"
    payload = {
        "format": "awa-v2.38-progressive-execution-plan-v1",
        "status": status,
        "phase_id": phase_id,
        "phase_name": str(next_action.get("name")),
        "phase_class": execution_meta.get("class"),
        "transition_target": transition_target,
        "max_transition_target": limit,
        "requires_expensive_ack": bool(expensive),
        "commands": [c.to_dict() for c in commands],
        "manual_requirement": manual,
        "repo_root": str(repo),
        "evidence_root": str(evidence),
        "progressive_report": progress,
        "claim_boundary": (
            "This plan executes only the first blocked phase. Missing scientific measurements remain missing; "
            "the executor never synthesizes benchmark evidence."
        ),
    }
    return payload | {"plan_sha256": _sha(payload)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def execute_phase_plan(
    plan: dict[str, Any],
    *,
    allow_expensive: bool = False,
    cwd: str | Path | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    if plan.get("status") != "READY":
        raise RuntimeError(f"execution plan is not READY: {plan.get('status')}")
    if plan.get("requires_expensive_ack") and not allow_expensive:
        raise PermissionError("this phase requires --allow-expensive")
    commands = [CommandSpec(tuple(row["argv"]), str(row.get("description", ""))) for row in plan.get("commands", [])]
    _validate_commands(commands)
    if not commands:
        raise RuntimeError("READY plan contains no commands")
    receipts = []
    started = time.time()
    for index, command in enumerate(commands):
        executable = shutil.which(command.argv[0])
        if executable is None:
            raise FileNotFoundError(f"required CLI entry point is not installed: {command.argv[0]}")
        argv = (executable, *command.argv[1:])
        result = runner(list(argv), cwd=str(cwd) if cwd is not None else None, shell=False, text=True, capture_output=True)
        row = {
            "index": index,
            "argv": list(command.argv),
            "returncode": int(result.returncode),
            "stdout": result.stdout[-20000:],
            "stderr": result.stderr[-20000:],
        }
        receipts.append(row)
        if result.returncode != 0:
            return {
                "format": "awa-v2.38-progressive-phase-execution-v1",
                "status": "FAILED",
                "phase_id": plan.get("phase_id"),
                "plan_sha256": plan.get("plan_sha256"),
                "elapsed_seconds": time.time() - started,
                "commands": receipts,
            }
    return {
        "format": "awa-v2.38-progressive-phase-execution-v1",
        "status": "PASS",
        "phase_id": plan.get("phase_id"),
        "plan_sha256": plan.get("plan_sha256"),
        "elapsed_seconds": time.time() - started,
        "commands": receipts,
    }


def execute_next_progressive_phase(
    *,
    executor_config: str | Path,
    evidence_root: str | Path,
    repo_root: str | Path | None = None,
    allow_expensive: bool = False,
    max_transition_target: int | None = None,
    runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> dict[str, Any]:
    plan = build_next_phase_execution_plan(
        executor_config=executor_config,
        evidence_root=evidence_root,
        repo_root=repo_root,
        max_transition_target=max_transition_target,
    )
    execution = execute_phase_plan(plan, allow_expensive=allow_expensive, cwd=repo_root, runner=runner)
    evidence = Path(evidence_root).resolve()
    cfg = load_executor_config(executor_config)
    protocol = load_progressive_protocol(cfg["progressive_config"])
    after = build_progressive_capability_report(evidence, protocol=protocol)
    if execution["status"] == "PASS" and after["first_blocked_phase"] == plan["phase_id"]:
        execution["status"] = "FAILED"
        execution["reason"] = "command completed, but the phase did not meet its evidence gate"
    receipt_path = evidence / "progressive_executor" / f"{str(plan['phase_id']).lower()}_execution_receipt.json"
    _atomic_json(receipt_path, execution)
    combined = {
        "format": "awa-v2.38-progressive-executor-run-v1",
        "plan": plan,
        "execution": execution,
        "after": after,
        "receipt_path": str(receipt_path),
    }
    return combined | {"run_sha256": _sha(combined)}
