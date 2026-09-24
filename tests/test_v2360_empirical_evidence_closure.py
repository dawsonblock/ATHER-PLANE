from __future__ import annotations

import hashlib
import json
from pathlib import Path

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.research_os.external_baseline import (
    BaselineResultRecord,
    MatchedBaselineProtocol,
    compare_external_baseline,
)
from awa.v2.research_os.real_campaign_evidence import (
    EvidenceBundleConfig,
    build_real_execution_evidence_bundle,
)
from awa.v2.system_split import validate_split_packages

ROOT = Path(__file__).resolve().parents[1]


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_v236_version_and_split():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"
    assert validate_split_packages(ROOT / "src") == []


def test_evidence_bundle_fails_closed_when_required_artifacts_missing():
    report = build_real_execution_evidence_bundle()
    assert report["qualified"] is False
    assert set(report["required_failures"]) == {"preflight", "real_training"}
    assert len(report["bundle_sha256"]) == 64


def test_evidence_bundle_binds_real_training_and_checkpoint_hashes(tmp_path: Path):
    preflight = tmp_path / "execution_preflight.json"
    preflight.write_text(
        json.dumps({
            "ready": True,
            "required_failures": [],
            "environment_fingerprint_sha256": _digest("env"),
        }),
        encoding="utf-8",
    )
    world = tmp_path / "world.pt"
    actor = tmp_path / "actor.pt"
    world.write_bytes(b"world")
    actor.write_bytes(b"actor")
    world_sha = hashlib.sha256(world.read_bytes()).hexdigest()
    actor_sha = hashlib.sha256(actor.read_bytes()).hexdigest()
    training = tmp_path / "real_vizdoom_training_receipt.json"
    training.write_text(
        json.dumps({
            "track": "structured",
            "backend": {"real_vizdoom": True},
            "vjepa": {"requested": False, "real_backbone": False},
            "transitions": 2000,
            "stable_checkpoint": {
                "world": str(world),
                "actor": str(actor),
                "world_sha256": world_sha,
                "actor_sha256": actor_sha,
            },
        }),
        encoding="utf-8",
    )
    report = build_real_execution_evidence_bundle(
        preflight_path=preflight,
        training_receipt_path=training,
    )
    assert report["qualified"] is True
    assert report["required_failures"] == []
    assert report["artifacts"]["stable_checkpoints"]["world"]["digest_match"] is True
    assert report["artifacts"]["stable_checkpoints"]["actor"]["digest_match"] is True


def test_evidence_bundle_rejects_fake_pixel_backbone(tmp_path: Path):
    preflight = tmp_path / "p.json"
    preflight.write_text(json.dumps({"ready": True, "required_failures": []}), encoding="utf-8")
    world = tmp_path / "w.pt"
    actor = tmp_path / "a.pt"
    world.write_bytes(b"w")
    actor.write_bytes(b"a")
    training = tmp_path / "t.json"
    training.write_text(
        json.dumps({
            "track": "pixel",
            "backend": {"real_vizdoom": True},
            "vjepa": {"real_backbone": False},
            "transitions": 1000,
            "stable_checkpoint": {
                "world": str(world),
                "actor": str(actor),
                "world_sha256": hashlib.sha256(world.read_bytes()).hexdigest(),
                "actor_sha256": hashlib.sha256(actor.read_bytes()).hexdigest(),
            },
        }),
        encoding="utf-8",
    )
    report = build_real_execution_evidence_bundle(preflight_path=preflight, training_receipt_path=training)
    assert report["qualified"] is False
    row = {x["name"]: x for x in report["checks"]}["real_training"]
    assert row["details"]["real_vjepa"] is False


def _record(system: str, seed: int, success: float, seconds: float = 100.0) -> BaselineResultRecord:
    return BaselineResultRecord(
        system=system,
        seed=seed,
        milestone=25_000,
        track="structured",
        transitions=25_000,
        success_rate=success,
        mean_return=success * 10,
        accelerator_seconds=seconds,
        artifact_sha256=_digest(f"{system}-{seed}"),
    )


def test_matched_external_baseline_comparison_is_paired_and_qualified():
    aether = [_record("aether", seed, 0.70 + seed * 0.001) for seed in range(5)]
    baseline = [_record("baseline", seed, 0.60 + seed * 0.001) for seed in range(5)]
    report = compare_external_baseline(
        aether,
        baseline,
        MatchedBaselineProtocol(required_pairs=5, bootstrap_samples=200, max_accelerator_ratio=1.25),
    )
    assert report["status"] == "QUALIFIED_COMPARISON"
    assert report["pair_count"] == 5
    assert abs(report["mean_success_delta"] - 0.10) < 1e-9
    assert report["success_delta_ci95"][0] > 0


def test_matched_external_baseline_fails_on_unmatched_compute_budget():
    aether = [_record("aether", seed, 0.7, seconds=200.0) for seed in range(5)]
    baseline = [_record("baseline", seed, 0.6, seconds=100.0) for seed in range(5)]
    report = compare_external_baseline(
        aether,
        baseline,
        MatchedBaselineProtocol(required_pairs=5, bootstrap_samples=50, max_accelerator_ratio=1.25),
    )
    assert report["status"] == "INSUFFICIENT_OR_UNMATCHED_EVIDENCE"
    assert any(x.startswith("accelerator_budget_exceeded") for x in report["violations"])


def test_new_cli_entries_are_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert "awa-v2-real-evidence-bundle" in pyproject
    assert "awa-v2-external-baseline-compare" in pyproject
