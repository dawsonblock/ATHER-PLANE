"""Progressive empirical capability gates for Aether v2.38.

The module does not change action-time cognition.  It turns the staged training
roadmap into a fail-closed protocol that advances only when prerequisite evidence
exists.  It also provides a reliability gate for selecting the longest world-
model planning horizon supported by held-out error/calibration measurements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml


@dataclass(frozen=True)
class HorizonPoint:
    horizon: int
    normalized_prediction_error: float
    calibration_error: float
    samples: int

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "HorizonPoint":
        return cls(
            horizon=int(payload["horizon"]),
            normalized_prediction_error=float(payload["normalized_prediction_error"]),
            calibration_error=float(payload["calibration_error"]),
            samples=int(payload.get("samples", 0)),
        )


@dataclass(frozen=True)
class HorizonGateConfig:
    max_normalized_prediction_error: float = 0.20
    max_calibration_error: float = 0.10
    minimum_samples: int = 256
    minimum_qualified_horizon: int = 4


def qualify_world_model_horizon(
    points: Iterable[HorizonPoint | dict[str, Any]],
    config: HorizonGateConfig = HorizonGateConfig(),
) -> dict[str, Any]:
    """Select the largest *contiguous* trustworthy planning horizon.

    A later horizon cannot be accepted after an earlier horizon fails.  This
    prevents a noisy long-horizon point from bypassing a broken short-horizon
    dynamics regime.
    """
    rows = [p if isinstance(p, HorizonPoint) else HorizonPoint.from_dict(p) for p in points]
    rows.sort(key=lambda x: x.horizon)
    if not rows:
        return {
            "format": "awa-v2.38-world-model-horizon-gate-v1",
            "qualified": False,
            "selected_horizon": 0,
            "reason": "no horizon measurements",
            "points": [],
            "thresholds": asdict(config),
        }
    if len({r.horizon for r in rows}) != len(rows) or any(r.horizon < 1 for r in rows):
        raise ValueError("horizons must be unique positive integers")

    accepted: list[int] = []
    annotated = []
    stopped = False
    for row in rows:
        meets = (
            row.samples >= config.minimum_samples
            and row.normalized_prediction_error <= config.max_normalized_prediction_error
            and row.calibration_error <= config.max_calibration_error
        )
        accepted_here = bool(meets and not stopped)
        if accepted_here:
            accepted.append(row.horizon)
        else:
            stopped = True
        annotated.append(asdict(row) | {"meets_thresholds": bool(meets), "accepted": accepted_here})

    selected = max(accepted, default=0)
    qualified = selected >= config.minimum_qualified_horizon
    return {
        "format": "awa-v2.38-world-model-horizon-gate-v1",
        "qualified": bool(qualified),
        "selected_horizon": int(selected),
        "reason": (
            "held-out reliability supports the selected contiguous horizon"
            if qualified
            else "held-out reliability does not support the minimum planning horizon"
        ),
        "points": annotated,
        "thresholds": asdict(config),
        "claim_boundary": (
            "This gate selects a planning horizon from held-out prediction/calibration measurements. "
            "It does not establish that planning improves realized return; planner benefit requires grounded branch evidence."
        ),
    }


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _latest(root: Path, name: str) -> tuple[Path, dict[str, Any]] | None:
    rows: list[tuple[float, Path, dict[str, Any]]] = []
    for path in root.rglob(name):
        payload = _read_json(path)
        if payload is not None:
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            rows.append((mtime, path, payload))
    if not rows:
        return None
    _, path, payload = max(rows, key=lambda x: (x[0], str(x[1])))
    return path, payload


def _all_json(root: Path, name: str) -> list[tuple[Path, dict[str, Any]]]:
    out = []
    for path in root.rglob(name):
        payload = _read_json(path)
        if payload is not None:
            out.append((path, payload))
    return out


def _mechanism_decision(root: Path, mechanism: str) -> tuple[bool, str | None, str | None]:
    matches: list[tuple[float, Path, str]] = []
    for path, report in _all_json(root, "ablation_report.json"):
        if report.get("status") != "QUALIFIED":
            continue
        for row in report.get("marginal_components", []):
            if row.get("mechanism") == mechanism:
                try:
                    mtime = path.stat().st_mtime
                except OSError:
                    mtime = 0.0
                matches.append((mtime, path, str(row.get("decision", ""))))
    if not matches:
        return False, None, None
    _, path, decision = max(matches, key=lambda x: (x[0], str(x[1])))
    return decision == "KEEP", decision, str(path)


def _full_ablation(root: Path, min_records: int = 140) -> tuple[bool, str | None]:
    required = {
        "temporal_memory", "world_model", "fixed_planner", "adaptive_compute",
        "grounded_hindsight_replay", "adaptive_curriculum",
    }
    matches: list[tuple[float, Path]] = []
    for path, report in _all_json(root, "ablation_report.json"):
        if str(report.get("status", "")).upper() != "QUALIFIED" or not bool(report.get("complete")):
            continue
        if int(report.get("records", 0) or 0) < int(min_records):
            continue
        mechanisms = {str(row.get("mechanism")) for row in report.get("marginal_components", [])}
        if not required.issubset(mechanisms):
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        matches.append((mtime, path))
    if not matches:
        return False, None
    _, path = max(matches, key=lambda x: (x[0], str(x[1])))
    return True, str(path)


def _real_training(root: Path, *, track: str, min_transitions: int, require_vjepa: bool = False,
                   stage: str | None = None, require_paired_comparison: bool = False,
                   require_success: bool = False, scenario: str | None = None,
                   incumbent_stage: str | None = None) -> tuple[bool, str | None]:
    for path, payload in _all_json(root, "real_vizdoom_training_receipt.json"):
        if str(payload.get("track")) != track:
            continue
        if not bool(payload.get("backend", {}).get("real_vizdoom")):
            continue
        if int(payload.get("transitions", 0)) < int(min_transitions):
            continue
        if require_vjepa and not bool(payload.get("vjepa", {}).get("real_backbone")):
            continue
        if payload.get("checkpoint_registry_verified") is False:
            continue
        if stage is not None:
            rows = (payload.get("result") or {}).get("results") or []
            row = next((r for r in rows if isinstance(r, dict) and r.get("stage") == stage), None)
            promoted_checkpoint = (row or {}).get("stable") or {}
            baseline_registration = bool(
                stage == "real-wiring-2k"
                and row is not None
                and row.get("baseline_registered") is True
                and row.get("promotion_status") == "baseline_registration"
            )
            if row is None or not (row.get("promoted") is True or baseline_registration) or promoted_checkpoint.get("stage") != stage:
                continue
            if require_paired_comparison:
                contract = row.get("comparison_contract") or {}
                previous = next((r for r in rows if isinstance(r, dict) and
                                 r.get("stage") == incumbent_stage and r.get("promoted") is True), None)
                previous_checkpoint = (previous or {}).get("stable") or {}
                training = row.get("training") or {}
                previous_training = (previous or {}).get("training") or {}
                if (contract.get("scenario") != scenario
                    or contract.get("track") != track
                    or contract.get("incumbent_stage") != incumbent_stage
                    or previous is None
                    or training.get("resumed") is not True
                    or training.get("resume_world_sha256") != contract.get("incumbent_world_sha256")
                    or training.get("resume_actor_sha256") != contract.get("incumbent_actor_sha256")
                    or int(training.get("world_global_updates") or 0) <= int(previous_training.get("world_global_updates") or 0)
                    or any(not isinstance(previous_checkpoint.get(f"{key}_sha256"), str) or
                           len(previous_checkpoint[f"{key}_sha256"]) != 64 or
                           contract.get(f"incumbent_{key}_sha256") != previous_checkpoint[f"{key}_sha256"]
                           for key in ("world", "actor"))
                    or not isinstance(row.get("incumbent_evaluation"), dict)
                    or not isinstance(contract.get("seed_start"), int)
                    or int(contract.get("eval_episodes", 0)) < 1):
                    continue
            if require_success:
                metrics = row.get("evaluation") or {}
                if max(float(metrics.get("actor_success_rate", 0) or 0),
                       float(metrics.get("planner_success_rate", 0) or 0)) <= 0:
                    continue
        return True, str(path)
    return False, None


def _preflight(root: Path) -> tuple[bool, str | None]:
    item = _latest(root, "execution_preflight.json")
    if item is None:
        return False, None
    path, payload = item
    return bool(payload.get("ready")) and not payload.get("required_failures"), str(path)


def _planner_hardware(root: Path) -> tuple[bool, str | None]:
    # Support both the canonical filename and renamed copies of the same report.
    candidates = _all_json(root, "planner_schedule_gpu.json") + _all_json(root, "planner_hardware_benchmark.json")
    for path, payload in candidates:
        action_ok = payload.get("action_schedule_equivalent")
        if action_ok is None:
            action_ok = payload.get("selected_action_equivalence")
        status = str(payload.get("status", "")).upper()
        if bool(action_ok) and status not in {"FAILED", "INVALID"}:
            return True, str(path)
    return False, None


def _qualified_named_report(root: Path, filename: str) -> tuple[bool, str | None]:
    item = _latest(root, filename)
    if item is None:
        return False, None
    path, payload = item
    status = str(payload.get("status", "")).upper()
    qualified = bool(payload.get("qualified")) or status in {
        "QUALIFIED", "PASS", "PASSED", "COMPLETE", "QUALIFIED_COMPARISON"
    }
    return qualified, str(path)


def _reward_diagnostic(root: Path) -> tuple[bool, str | None, str]:
    item = _latest(root, "reward_diagnostic.json")
    if item is None:
        return False, None, "missing_paired_reward_diagnostic"
    path, payload = item
    from .planner_diagnostic import validate_reward_diagnostic_report
    ok, detail = validate_reward_diagnostic_report(payload)
    return ok, str(path), detail


def _planner_diagnostic(root: Path) -> tuple[bool, str | None, str]:
    item = _latest(root, "planner_diagnostic.json")
    if item is None:
        return False, None, "missing_planner_branch_diagnostic_report"
    path, payload = item
    raw_path = path.with_name("planner_diagnostic_raw.json")
    raw = _read_json(raw_path)
    if raw is None:
        return False, str(path), "missing_or_invalid_planner_raw_evidence"
    replay_path = path.with_name("vizdoom_branch_replay.json")
    replay = _read_json(replay_path)
    raw_seeds = raw.get("seed_ids")
    from .branch_replay import validate_vizdoom_reset_replay_report
    if (replay is None or not isinstance(raw_seeds, list)
        or not validate_vizdoom_reset_replay_report(replay, raw_seeds)
        or raw.get("branch_replay_report_sha256") != _file_sha256(replay_path)):
        return False, str(path), "missing_or_unqualified_bound_branch_replay"
    from .planner_diagnostic import build_planner_diagnostic_report
    try:
        # The report is a derived artifact. A PASS flag and a plausible-looking
        # digest are not evidence unless they reproduce from the retained raw rows.
        if payload != build_planner_diagnostic_report(raw):
            return False, str(path), "planner_report_does_not_match_raw_evidence"
    except (KeyError, TypeError, ValueError, OverflowError):
        return False, str(path), "invalid_planner_raw_evidence"
    checkpoint_hashes = raw.get("checkpoint_sha256") or {}
    checkpoint_verified = False
    for _, receipt in _all_json(root, "real_vizdoom_training_receipt.json"):
        if (receipt.get("track") != "structured" or
            receipt.get("checkpoint_registry_verified") is not True or
            not bool((receipt.get("backend") or {}).get("real_vizdoom"))):
            continue
        rows = (receipt.get("result") or {}).get("results") or []
        p1 = next((r for r in rows if isinstance(r, dict) and
                   r.get("stage") == "real-wiring-2k" and
                   (r.get("promoted") is True or
                    (r.get("baseline_registered") is True and
                     r.get("promotion_status") == "baseline_registration"))), None)
        stable = (p1 or {}).get("stable") or {}
        if stable.get("stage") != "real-wiring-2k":
            continue
        verified = True
        for key in ("world", "actor"):
            checkpoint = Path(str(stable.get(key, "")))
            expected = stable.get(f"{key}_sha256")
            if (not checkpoint.is_file() or not isinstance(expected, str) or
                len(expected) != 64 or checkpoint_hashes.get(key) != expected or
                _file_sha256(checkpoint) != expected):
                verified = False
                break
        if verified:
            checkpoint_verified = True
            break
    if not checkpoint_verified:
        return False, str(path), "planner_checkpoint_not_bound_to_verified_p1"
    answers = payload.get("question_answers") or {}
    required = {"oracle_search_navigates", "joint_model_reward_ranking_horizon",
                "value_and_risk_effects", "actor_seed_search_restriction"}
    ranking = payload.get("candidate_ranking") or {}
    by_horizon = ((ranking.get("P1P-B") or {}).get("by_horizon") or {})
    effects = answers.get("actor_seed_search_restriction", {}).get("proposal_effects") or {}
    paired = answers.get("value_and_risk_effects", {}).get("paired_deltas") or {}
    complete = (payload.get("format") == "awa-v2.38.6-planner-diagnostic-v2"
                and payload.get("diagnosis_complete") is True
                and payload.get("p2_authorized_by_planner_gate") is True
                and payload.get("status") == "PASS"
                and isinstance(payload.get("raw_evidence_sha256"), str)
                and len(payload.get("raw_evidence_sha256", "")) == 64
                and len(payload.get("seed_ids", [])) >= 6
                and payload.get("horizons") == [1, 2, 4, 8, 16, 32]
                and required.issubset(answers)
                and all(bool(answers[k].get("answered")) for k in required)
                and answers.get("oracle_search_navigates", {}).get("value") is True
                and set(by_horizon) == {"1", "2", "4", "8", "16", "32"}
                and all(int(by_horizon[h].get("groups", 0)) >= 12 for h in by_horizon)
                and payload.get("identifiability", {}).get("dynamics_and_reward") == "joint_only"
                and {"learned_terminal_value_vs_none", "learned_risk_vs_none"}.issubset(paired)
                and {"P1P-B", "P1P-C", "P1P-D"}.issubset(effects))
    if complete:
        detail = "complete_planner_branch_diagnostic_and_p2_requirements_pass"
    elif payload.get("diagnosis_complete") and answers.get("oracle_search_navigates", {}).get("value") is not True:
        detail = "oracle_search_does_not_clear_navigation_check"
    elif payload.get("diagnosis_complete"):
        detail = "joint_model_reward_ranking_does_not_qualify_p2_horizon"
    else:
        detail = "incomplete_planner_branch_diagnostic"
    return complete, str(path), detail


def _dream_campaign(root: Path) -> tuple[bool, str | None]:
    # Names used by campaign code and exported reports are both accepted.
    for filename in ("dream_rsi_campaign_report.json", "fixed_vs_dream_report.json", "campaign_report.json"):
        for path, payload in _all_json(root, filename):
            status = str(payload.get("status", "")).upper()
            seeds = int(payload.get("seeds", payload.get("seed_count", 0)) or 0)
            complete = bool(payload.get("complete", status in {"QUALIFIED", "COMPLETE", "PASS"}))
            if complete and seeds >= 5:
                return True, str(path)
    return False, None


def _default_phases() -> list[dict[str, Any]]:
    return [
        {"id": "P0", "name": "cuda_vizdoom_preflight", "kind": "preflight"},
        {"id": "P1", "name": "structured_bringup_2k", "kind": "real_training", "track": "structured", "min_transitions": 2000, "stage": "real-wiring-2k"},
        {"id": "P1D", "name": "paired_reward_scale_diagnostic", "kind": "reward_diagnostic"},
        {"id": "P1P", "name": "planner_branch_diagnostic", "kind": "planner_diagnostic"},
        {"id": "P2", "name": "structured_learning_10k", "kind": "real_training", "track": "structured", "min_transitions": 10000, "stage": "real-navigation-10k", "scenario": "my_way_home", "incumbent_stage": "real-wiring-2k", "require_paired_comparison": True, "require_success": True},
        {"id": "P3", "name": "structured_control_25k", "kind": "real_training", "track": "structured", "min_transitions": 25000, "stage": "real-control-25k", "scenario": "health_gathering", "incumbent_stage": "real-navigation-10k", "require_paired_comparison": True, "require_success": True},
        {"id": "P4", "name": "temporal_memory_value", "kind": "ablation", "mechanism": "temporal_memory"},
        {"id": "P5", "name": "world_model_reliable_horizon", "kind": "report", "filename": "world_model_horizon_gate.json"},
        {"id": "P6", "name": "grounded_planner_benefit", "kind": "ablation", "mechanism": "fixed_planner"},
        {"id": "P7", "name": "adaptive_compute_value", "kind": "ablation", "mechanism": "adaptive_compute"},
        {"id": "P8", "name": "real_vjepa_pixel_5k", "kind": "real_training", "track": "pixel", "min_transitions": 5000, "require_vjepa": True, "stage": "real-pixel-navigation-5k", "scenario": "my_way_home", "incumbent_stage": "real-pixel-wiring-1k", "require_paired_comparison": True, "require_success": True},
        {"id": "P9", "name": "planner_physical_schedule", "kind": "planner_hardware"},
        {"id": "P10", "name": "compositional_generalization", "kind": "report", "filename": "compositional_report.json"},
        {"id": "P11", "name": "dream_rsi_five_seed", "kind": "dream_campaign"},
        {"id": "P12", "name": "unseen_map_transfer", "kind": "report", "filename": "transfer_report.json"},
        {"id": "P13", "name": "matched_external_baseline", "kind": "report", "filename": "external_baseline_comparison.json"},
        {"id": "P14", "name": "full_seven_system_25k_100k_ablation", "kind": "full_ablation", "min_records": 140},
        {"id": "P15", "name": "scale_250k_plus", "kind": "manual_scale_gate"},
    ]


def load_progressive_protocol(path: str | Path | None = None) -> dict[str, Any]:
    if path is None:
        payload: dict[str, Any] = {}
    else:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(payload, dict):
            raise ValueError("progressive capability config must decode to a mapping")
    phases = payload.get("phases") or _default_phases()
    if not isinstance(phases, list) or not phases:
        raise ValueError("progressive capability protocol requires phases")
    ids = [str(x["id"]) for x in phases]
    if len(ids) != len(set(ids)):
        raise ValueError("progressive capability phase ids must be unique")
    return {
        "format": "awa-v2.38-progressive-capability-protocol-v1",
        "release": str(payload.get("release", "2.38.6")),
        "phases": phases,
        "gating": payload.get("gating") or {
            "minimum_final_success_gain": 0.03,
            "minimum_relative_curve_gain": 0.10,
            "minimum_efficiency_reduction": 0.20,
            "protected_regression_margin": 0.01,
        },
        "claim_boundary": payload.get("claim_boundary") or [
            "A phase advances only from evidence artifacts; the protocol does not infer missing empirical results.",
            "Planning horizon reliability and realized planner benefit are separate gates.",
            "Scaling to 250K+ is blocked until earlier real-environment, transfer, and baseline evidence is present.",
        ],
    }


def _eval_phase(root: Path, phase: dict[str, Any]) -> tuple[bool, str, str | None]:
    kind = str(phase.get("kind", ""))
    if kind == "preflight":
        ok, evidence = _preflight(root); return ok, "ready" if ok else "missing_or_failed_preflight", evidence
    if kind == "real_training":
        ok, evidence = _real_training(
            root,
            track=str(phase.get("track", "structured")),
            min_transitions=int(phase.get("min_transitions", 0)),
            require_vjepa=bool(phase.get("require_vjepa", False)),
            stage=str(phase["stage"]) if phase.get("stage") else None,
            require_paired_comparison=bool(phase.get("require_paired_comparison", False)),
            require_success=bool(phase.get("require_success", False)),
            scenario=str(phase["scenario"]) if phase.get("scenario") else None,
            incumbent_stage=str(phase["incumbent_stage"]) if phase.get("incumbent_stage") else None,
        )
        detail = ("real_training_evidence" if ok else
                  "missing_stage_promotion_paired_comparison_or_success" if phase.get("stage") else
                  "missing_real_training_evidence")
        return ok, detail, evidence
    if kind == "ablation":
        ok, decision, evidence = _mechanism_decision(root, str(phase["mechanism"]))
        return ok, f"decision={decision or 'missing'}", evidence
    if kind == "planner_hardware":
        ok, evidence = _planner_hardware(root); return ok, "action_equivalent_schedule" if ok else "missing_action_equivalent_gpu_sweep", evidence
    if kind == "dream_campaign":
        ok, evidence = _dream_campaign(root); return ok, "five_seed_campaign_complete" if ok else "missing_complete_five_seed_campaign", evidence
    if kind == "full_ablation":
        ok, evidence = _full_ablation(root, int(phase.get("min_records", 140)))
        return ok, "full_seven_system_matrix_complete" if ok else "missing_complete_full_ablation_matrix", evidence
    if kind == "report":
        ok, evidence = _qualified_named_report(root, str(phase["filename"]))
        return ok, "qualified_report" if ok else "missing_or_unqualified_report", evidence
    if kind == "reward_diagnostic":
        ok, evidence, detail = _reward_diagnostic(root)
        return ok, detail, evidence
    if kind == "planner_diagnostic":
        ok, evidence, detail = _planner_diagnostic(root)
        return ok, detail, evidence
    if kind == "manual_scale_gate":
        item = _latest(root, "scale_authorization.json")
        if item is None:
            return False, "manual_scale_authorization_missing", None
        path, payload = item
        authorized = bool(payload.get("authorized")) and str(payload.get("status", "")).upper() in {"AUTHORIZED", "PASS", "QUALIFIED"}
        return authorized, "explicit_scale_authorization" if authorized else "scale_not_authorized", str(path)
    raise ValueError(f"unknown progressive capability phase kind: {kind}")


def build_progressive_capability_report(
    evidence_root: str | Path,
    *,
    protocol: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = Path(evidence_root).resolve()
    proto = protocol or load_progressive_protocol()
    phase_rows = []
    chain_open = True
    first_blocked: str | None = None
    for phase in proto["phases"]:
        evidence_ok, detail, evidence = _eval_phase(root, phase)
        status = "PASS" if chain_open and evidence_ok else "BLOCKED"
        if chain_open and not evidence_ok:
            chain_open = False
            first_blocked = str(phase["id"])
        phase_rows.append({
            "id": str(phase["id"]),
            "name": str(phase["name"]),
            "kind": str(phase["kind"]),
            "status": status,
            "evidence_condition_met": bool(evidence_ok),
            "detail": detail,
            "evidence": evidence,
        })
    next_phase = next((row for row in phase_rows if row["status"] == "BLOCKED"), None)
    serial = json.dumps({"protocol": proto, "phases": phase_rows}, sort_keys=True, separators=(",", ":")).encode()
    return {
        "format": "awa-v2.38-progressive-capability-report-v1",
        "evidence_root": str(root),
        "protocol": proto,
        "phases": phase_rows,
        "completed_prefix": sum(1 for row in phase_rows if row["status"] == "PASS"),
        "first_blocked_phase": first_blocked,
        "next_action": ({"id": next_phase["id"], "name": next_phase["name"], "detail": next_phase["detail"]} if next_phase else None),
        "report_sha256": hashlib.sha256(serial).hexdigest(),
        "claim_boundary": (
            "PASS means the configured evidence condition exists and all earlier phases passed. "
            "It is not a claim beyond the scope of the bound evidence artifact."
        ),
    }


def write_progressive_capability_report(report: dict[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def write_horizon_gate(report: dict[str, Any], output: str | Path) -> Path:
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
