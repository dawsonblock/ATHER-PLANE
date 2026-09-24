"""Matched external-baseline comparison contract.

The comparator consumes result records only.  It never downloads or runs an
external implementation, which keeps provenance and execution ownership clear.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class BaselineResultRecord:
    system: str
    seed: int
    milestone: int
    track: str
    transitions: int
    success_rate: float
    mean_return: float
    accelerator_seconds: float
    artifact_sha256: str

    def __post_init__(self) -> None:
        if not self.system or not self.track:
            raise ValueError("system and track are required")
        if self.milestone <= 0 or self.transitions <= 0:
            raise ValueError("milestone/transitions must be > 0")
        if not 0.0 <= float(self.success_rate) <= 1.0:
            raise ValueError("success_rate must be in [0,1]")
        if self.accelerator_seconds < 0:
            raise ValueError("accelerator_seconds must be >= 0")
        if len(self.artifact_sha256) != 64:
            raise ValueError("artifact_sha256 must be a 64-character digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaselineResultRecord":
        return cls(**payload)


@dataclass(frozen=True)
class MatchedBaselineProtocol:
    required_pairs: int = 5
    require_equal_transitions: bool = True
    max_accelerator_ratio: float | None = 1.25
    bootstrap_samples: int = 5000
    bootstrap_seed: int = 2360

    def __post_init__(self) -> None:
        if self.required_pairs <= 0 or self.bootstrap_samples <= 0:
            raise ValueError("required_pairs/bootstrap_samples must be positive")
        if self.max_accelerator_ratio is not None and self.max_accelerator_ratio < 1.0:
            raise ValueError("max_accelerator_ratio must be >= 1")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _key(row: BaselineResultRecord) -> tuple[int, int, str]:
    return (int(row.seed), int(row.milestone), str(row.track))


def _paired_bootstrap(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    if values.size == 0:
        return (float("nan"), float("nan"))
    if values.size == 1:
        value = float(values[0])
        return (value, value)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True).mean(axis=1)
    return (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))


def compare_external_baseline(
    aether: Iterable[BaselineResultRecord],
    baseline: Iterable[BaselineResultRecord],
    protocol: MatchedBaselineProtocol | None = None,
) -> dict[str, Any]:
    cfg = protocol or MatchedBaselineProtocol()
    aether_map = {_key(row): row for row in aether}
    baseline_map = {_key(row): row for row in baseline}
    common = sorted(set(aether_map) & set(baseline_map))
    rows: list[dict[str, Any]] = []
    violations: list[str] = []
    for key in common:
        left = aether_map[key]
        right = baseline_map[key]
        if cfg.require_equal_transitions and left.transitions != right.transitions:
            violations.append(f"transition_mismatch:{key}:{left.transitions}!={right.transitions}")
        denom = max(float(right.accelerator_seconds), 1e-12)
        accelerator_ratio = float(left.accelerator_seconds) / denom
        if cfg.max_accelerator_ratio is not None and accelerator_ratio > cfg.max_accelerator_ratio:
            violations.append(f"accelerator_budget_exceeded:{key}:{accelerator_ratio:.6f}")
        rows.append(
            {
                "seed": key[0],
                "milestone": key[1],
                "track": key[2],
                "aether_system": left.system,
                "baseline_system": right.system,
                "transitions": left.transitions,
                "success_delta": float(left.success_rate - right.success_rate),
                "return_delta": float(left.mean_return - right.mean_return),
                "accelerator_ratio": accelerator_ratio,
                "aether_artifact_sha256": left.artifact_sha256,
                "baseline_artifact_sha256": right.artifact_sha256,
            }
        )

    missing_aether = sorted(set(baseline_map) - set(aether_map))
    missing_baseline = sorted(set(aether_map) - set(baseline_map))
    if missing_aether:
        violations.append(f"missing_aether_pairs:{len(missing_aether)}")
    if missing_baseline:
        violations.append(f"missing_baseline_pairs:{len(missing_baseline)}")
    if len(common) < cfg.required_pairs:
        violations.append(f"insufficient_pairs:{len(common)}<{cfg.required_pairs}")

    success = np.asarray([row["success_delta"] for row in rows], dtype=np.float64)
    returns = np.asarray([row["return_delta"] for row in rows], dtype=np.float64)
    success_ci = _paired_bootstrap(success, cfg.bootstrap_samples, cfg.bootstrap_seed)
    return_ci = _paired_bootstrap(returns, cfg.bootstrap_samples, cfg.bootstrap_seed + 1)
    status = "QUALIFIED_COMPARISON" if not violations else "INSUFFICIENT_OR_UNMATCHED_EVIDENCE"
    return {
        "format": "awa-v2.36-external-baseline-comparison-v1",
        "status": status,
        "protocol": cfg.to_dict(),
        "pair_count": len(common),
        "violations": violations,
        "mean_success_delta": float(success.mean()) if success.size else None,
        "success_delta_ci95": list(success_ci) if success.size else None,
        "mean_return_delta": float(returns.mean()) if returns.size else None,
        "return_delta_ci95": list(return_ci) if returns.size else None,
        "pairs": rows,
        "claim_boundary": (
            "A qualified comparison means the supplied records satisfy this matching contract. "
            "It is not an automatic claim that Aether is superior; interpretation must use the reported paired effects and intervals."
        ),
    }


def load_baseline_records(path: str | Path) -> list[BaselineResultRecord]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("records") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError("baseline record file must contain a list or {'records': [...]} object")
    return [BaselineResultRecord.from_dict(dict(row)) for row in rows]


def write_baseline_comparison(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)
    return target
