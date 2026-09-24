from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


STATUS_ORDER = {
    "UNEXECUTED": 0,
    "IMPLEMENTED": 1,
    "SOFTWARE_VALIDATED": 2,
    "HARDWARE_VALIDATED": 3,
    "EMPIRICALLY_QUALIFIED": 4,
}


@dataclass(frozen=True)
class QualificationRow:
    tier: str
    capability: str
    status: str
    evidence: tuple[str, ...]
    boundary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _real_vizdoom_receipts(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in root.rglob("real_vizdoom_qualification.json"):
        payload = _read_json(path)
        if payload and bool(payload.get("backend", {}).get("real_vizdoom")):
            rows.append((path, payload))
    return rows


def _real_vizdoom_training_receipts(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows = []
    for path in root.rglob("real_vizdoom_training_receipt.json"):
        payload = _read_json(path)
        if payload and bool(payload.get("backend", {}).get("real_vizdoom")):
            rows.append((path, payload))
    return rows


def build_empirical_status(evidence_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(evidence_root).resolve() if evidence_root is not None else None
    rows: list[QualificationRow] = [
        QualificationRow(
            "T0",
            "continuous-point mathematical sanity",
            "SOFTWARE_VALIDATED",
            ("source tests and smoke coverage",),
            "Useful for mathematical/controller sanity only; not evidence of complex embodied control.",
        ),
        QualificationRow(
            "T1",
            "procedural 2-D arena integration",
            "SOFTWARE_VALIDATED",
            ("native procedural tests and campaign smokes",),
            "The shipped release does not include the canonical 25K/100K five-seed empirical result.",
        ),
        QualificationRow(
            "T2",
            "real ViZDoom structured-state control",
            "UNEXECUTED",
            (),
            "Requires a qualification receipt produced by the non-injected real ViZDoom path.",
        ),
        QualificationRow(
            "T3",
            "real ViZDoom RGB control",
            "UNEXECUTED",
            (),
            "Requires a real ViZDoom pixel-track qualification receipt.",
        ),
        QualificationRow(
            "T4",
            "real V-JEPA visual representation path",
            "UNEXECUTED",
            (),
            "Requires a non-injected V-JEPA model and real ViZDoom RGB frames.",
        ),
        QualificationRow(
            "T5",
            "unseen ViZDoom scenario/map transfer",
            "UNEXECUTED",
            (),
            "Requires a heldout real-ViZDoom transfer evaluation; no shipped result is assumed.",
        ),
        QualificationRow(
            "T6",
            "compositional adaptation/generalization",
            "IMPLEMENTED",
            ("factorized compositional benchmark generator",),
            "Task generation is implemented; a complete multi-seed result is not shipped.",
        ),
    ]

    if root is not None and root.exists():
        # Promote T1 only when the full ablation report is genuinely complete.
        reports = list(root.rglob("ablation_report.json"))
        for report_path in reports:
            report = _read_json(report_path)
            if report and report.get("status") == "QUALIFIED" and bool(report.get("complete", True)):
                rows[1] = QualificationRow(
                    "T1",
                    rows[1].capability,
                    "EMPIRICALLY_QUALIFIED",
                    (str(report_path),),
                    "Qualification applies only to the bound procedural benchmark/protocol.",
                )
                break

        for path, payload in _real_vizdoom_receipts(root):
            track = str(payload.get("track", ""))
            transitions = int(payload.get("collection", {}).get("transitions", 0))
            if transitions <= 0:
                continue
            if track == "structured":
                rows[2] = QualificationRow(
                    "T2",
                    rows[2].capability,
                    "HARDWARE_VALIDATED",
                    (str(path),),
                    "A real backend ran; benchmark-quality performance still requires a matched evaluation protocol.",
                )
            if track == "pixel":
                rows[3] = QualificationRow(
                    "T3",
                    rows[3].capability,
                    "HARDWARE_VALIDATED",
                    (str(path),),
                    "A real RGB environment path ran; this alone is not evidence of learned visual competence.",
                )
                if bool(payload.get("vjepa", {}).get("real_backbone")):
                    rows[4] = QualificationRow(
                        "T4",
                        rows[4].capability,
                        "HARDWARE_VALIDATED",
                        (str(path),),
                        "Real V-JEPA inference was observed; downstream benchmark performance remains separate evidence.",
                    )

        for path, payload in _real_vizdoom_training_receipts(root):
            track = str(payload.get("track", ""))
            transitions = int(payload.get("transitions", 0))
            if transitions <= 0:
                continue
            if track == "structured":
                rows[2] = QualificationRow(
                    "T2",
                    rows[2].capability,
                    "HARDWARE_VALIDATED",
                    (str(path),),
                    "Real ViZDoom collection, training and evaluation executed; multi-seed benchmark qualification remains separate.",
                )
            if track == "pixel":
                rows[3] = QualificationRow(
                    "T3",
                    rows[3].capability,
                    "HARDWARE_VALIDATED",
                    (str(path),),
                    "Real RGB collection, training and evaluation executed; this is not yet a multi-seed competence claim.",
                )
                if bool(payload.get("vjepa", {}).get("real_backbone")):
                    rows[4] = QualificationRow(
                        "T4",
                        rows[4].capability,
                        "HARDWARE_VALIDATED",
                        (str(path),),
                        "Real V-JEPA-backed downstream training/evaluation executed; external or multi-seed qualification remains separate.",
                    )

        for path in root.rglob("compositional_report.json"):
            report = _read_json(path)
            if report and report.get("status") == "QUALIFIED" and int(report.get("seeds", 0)) >= 5:
                rows[6] = QualificationRow(
                    "T6",
                    rows[6].capability,
                    "EMPIRICALLY_QUALIFIED",
                    (str(path),),
                    "Applies to the bound factorized benchmark only.",
                )
                break

    return {
        "format": "awa-v2.35-empirical-status-v2",
        "evidence_root": str(root) if root is not None else None,
        "tiers": [row.to_dict() for row in rows],
        "highest_status_rank": max(STATUS_ORDER[row.status] for row in rows),
        "claim_boundary": (
            "Implementation, unit tests, software smokes, hardware execution and empirical qualification are separate states. "
            "The status report never promotes a tier without corresponding evidence artifacts."
        ),
    }


def empirical_status_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Aether Empirical Status",
        "",
        "This file separates implementation maturity from empirical evidence. A passing unit test or synthetic smoke is not treated as proof of real-world control performance.",
        "",
        "| Tier | Capability | Status | Evidence / boundary |",
        "|---|---|---|---|",
    ]
    for row in payload["tiers"]:
        evidence = "; ".join(row.get("evidence", [])) or "none"
        boundary = str(row.get("boundary", ""))
        lines.append(
            f"| {row['tier']} | {row['capability']} | **{row['status']}** | {evidence}. {boundary} |"
        )
    lines.extend(["", f"Boundary: {payload['claim_boundary']}", ""])
    return "\n".join(lines)


def write_empirical_status(
    output_json: str | Path,
    output_markdown: str | Path,
    *,
    evidence_root: str | Path | None = None,
) -> tuple[Path, Path]:
    payload = build_empirical_status(evidence_root)
    json_path = Path(output_json)
    md_path = Path(output_markdown)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md_path.write_text(empirical_status_markdown(payload), encoding="utf-8")
    return json_path, md_path
