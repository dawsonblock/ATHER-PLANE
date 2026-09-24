"""Fail-closed consolidation of real-execution evidence.

This module never runs an environment or trains a model.  It verifies and binds
already-produced hardware/preflight, real ViZDoom training, and planner benchmark
artifacts into one portable evidence bundle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{target} must contain a JSON object")
    return payload


@dataclass(frozen=True)
class EvidenceBundleConfig:
    require_preflight: bool = True
    require_training: bool = True
    require_planner_benchmark: bool = False
    require_action_schedule_equivalence: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "size_bytes": int(path.stat().st_size),
    }


def build_real_execution_evidence_bundle(
    *,
    preflight_path: str | Path | None = None,
    training_receipt_path: str | Path | None = None,
    planner_benchmark_path: str | Path | None = None,
    config: EvidenceBundleConfig | None = None,
) -> dict[str, Any]:
    """Verify real-execution artifacts and return one content-bound bundle.

    Missing optional evidence is reported, never silently converted to PASS.
    Any required evidence failure makes ``qualified`` false.
    """

    cfg = config or EvidenceBundleConfig()
    checks: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}

    def add_check(name: str, required: bool, ok: bool, details: dict[str, Any]) -> None:
        checks.append(
            {
                "name": name,
                "required": bool(required),
                "status": "PASS" if ok else ("FAIL" if required else "MISSING"),
                "details": details,
            }
        )

    if preflight_path is None:
        add_check("preflight", cfg.require_preflight, False, {"reason": "not supplied"})
    else:
        path = Path(preflight_path)
        payload = _load_json(path)
        ok = bool(payload.get("ready")) and not payload.get("required_failures")
        artifacts["preflight"] = _artifact(path)
        add_check(
            "preflight",
            cfg.require_preflight,
            ok,
            {
                "ready": bool(payload.get("ready")),
                "required_failures": list(payload.get("required_failures") or []),
                "environment_fingerprint_sha256": payload.get("environment_fingerprint_sha256"),
            },
        )

    training_payload: dict[str, Any] | None = None
    if training_receipt_path is None:
        add_check("real_training", cfg.require_training, False, {"reason": "not supplied"})
    else:
        path = Path(training_receipt_path)
        training_payload = _load_json(path)
        backend = dict(training_payload.get("backend") or {})
        track = str(training_payload.get("track", ""))
        vjepa = dict(training_payload.get("vjepa") or {})
        stable = training_payload.get("stable_checkpoint")
        real_backend = bool(backend.get("real_vizdoom"))
        real_vjepa = track != "pixel" or bool(vjepa.get("real_backbone"))
        transitions = int(training_payload.get("transitions", 0) or 0)
        stable_ok = isinstance(stable, dict) and bool(stable.get("world")) and bool(stable.get("actor"))
        stable_digests_ok = True
        checkpoint_artifacts: dict[str, Any] = {}
        if stable_ok:
            for key in ("world", "actor"):
                checkpoint = Path(stable[key])
                if not checkpoint.exists():
                    stable_digests_ok = False
                    checkpoint_artifacts[key] = {"path": str(checkpoint), "missing": True}
                    continue
                actual = sha256_file(checkpoint)
                expected = stable.get(f"{key}_sha256")
                if expected and actual != expected:
                    stable_digests_ok = False
                checkpoint_artifacts[key] = {
                    **_artifact(checkpoint),
                    "expected_sha256": expected,
                    "digest_match": expected is None or actual == expected,
                }
        else:
            stable_digests_ok = False
        artifacts["training_receipt"] = _artifact(path)
        artifacts["stable_checkpoints"] = checkpoint_artifacts
        ok = real_backend and real_vjepa and transitions > 0 and stable_ok and stable_digests_ok
        add_check(
            "real_training",
            cfg.require_training,
            ok,
            {
                "track": track,
                "real_vizdoom": real_backend,
                "real_vjepa": real_vjepa,
                "transitions": transitions,
                "stable_checkpoint": stable_ok,
                "stable_checkpoint_digests_match": stable_digests_ok,
            },
        )

    if planner_benchmark_path is None:
        add_check(
            "planner_benchmark",
            cfg.require_planner_benchmark,
            False,
            {"reason": "not supplied"},
        )
    else:
        path = Path(planner_benchmark_path)
        payload = _load_json(path)
        logical = bool(payload.get("logical_work_matched"))
        action_equiv = bool(payload.get("action_schedule_equivalent"))
        rows = list(payload.get("rows") or [])
        hardware = bool(payload.get("cuda_device_name") or (payload.get("cuda_properties") or {}).get("name"))
        ok = logical and bool(rows) and hardware
        if cfg.require_action_schedule_equivalence:
            ok = ok and action_equiv
        artifacts["planner_benchmark"] = _artifact(path)
        add_check(
            "planner_benchmark",
            cfg.require_planner_benchmark,
            ok,
            {
                "logical_work_matched": logical,
                "action_schedule_equivalent": action_equiv,
                "row_count": len(rows),
                "hardware_identity_present": hardware,
            },
        )

    required_failures = [row["name"] for row in checks if row["required"] and row["status"] != "PASS"]
    binding = {
        "config": cfg.to_dict(),
        "artifacts": artifacts,
        "checks": checks,
    }
    bundle_sha = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "format": "awa-v2.36-real-execution-evidence-bundle-v1",
        "qualified": not required_failures,
        "required_failures": required_failures,
        "config": cfg.to_dict(),
        "checks": checks,
        "artifacts": artifacts,
        "bundle_sha256": bundle_sha,
        "claim_boundary": (
            "This bundle binds and verifies supplied execution evidence. It does not create benchmark evidence, "
            "infer missing measurements, or establish multi-seed superiority by itself."
        ),
    }


def write_real_execution_evidence_bundle(payload: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
