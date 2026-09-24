from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import tempfile
import time

import numpy as np
import torch
import yaml

from .ablation import EMPIRICAL_ABLATIONS, build_empirical_ablation_protocol
from .empirical_closure import RunRecord, file_sha256
from .evidence_integrity import RunProvenance, canonical_sha256
from .experiment_protocol import build_protocol_receipt
from .game.collectors import CoverageArenaPolicy
from .game.dataset import collect_game_dataset
from .game.variant_runtime import VARIANT_BY_ID, evaluate_trained_variant, train_and_evaluate_variant
from .curriculum import ProceduralTaskFactory
from .meta_exploration import DeclarativeExplorationPolicy
from .native_campaign import ExplorationPolicyCurriculumAdapter
from .scaling import resolve_device


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp-{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _sha(value: Any) -> str:
    return canonical_sha256(value)


def _tree_sha256(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        h.update(path.relative_to(root).as_posix().encode("utf-8"))
        h.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


class AtomicFileLock:
    """Small cross-process lock for shared cloud/network volumes.

    It uses O_EXCL, records owner/timestamp, and only breaks a lock after a long
    stale interval. The protected artifact is still committed with os.replace,
    so a crash cannot expose a half-written JSON or NPZ file.
    """

    def __init__(self, path: str | Path, *, timeout_seconds: float = 3600.0,
                 stale_seconds: float = 12 * 3600.0, poll_seconds: float = 0.25):
        self.path = Path(path)
        self.timeout_seconds = float(timeout_seconds)
        self.stale_seconds = float(stale_seconds)
        self.poll_seconds = float(poll_seconds)
        self.acquired = False

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        payload = json.dumps({"pid": os.getpid(), "created_at": time.time()}).encode("utf-8")
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                try:
                    os.write(fd, payload)
                finally:
                    os.close(fd)
                self.acquired = True
                return self
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                    if age > self.stale_seconds:
                        self.path.unlink(missing_ok=True)
                        continue
                except FileNotFoundError:
                    continue
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for lock: {self.path}")
                time.sleep(self.poll_seconds)

    def __exit__(self, exc_type, exc, tb):
        if self.acquired:
            self.path.unlink(missing_ok=True)
        self.acquired = False


@dataclass(frozen=True)
class AblationCampaignConfig:
    seeds: tuple[int, ...] = (1701, 1702, 1703, 1704, 1705)
    milestones: tuple[int, ...] = (25_000, 100_000)
    splits: tuple[str, ...] = ("heldout", "transfer")
    systems: tuple[str, ...] = tuple(s.ablation_id for s in EMPIRICAL_ABLATIONS)
    device: str = "auto"
    horizon: int = 100
    hidden: int = 64
    sequence_length: int = 5
    world_epochs: int = 2
    actor_epochs: int = 4
    representation_epochs: int = 2
    calibration_epochs: int = 2
    batch_size: int = 64
    planner_world_batch_size: int = 0
    voc_epochs: int = 40
    train_difficulty: float = 0.45
    heldout_difficulty: float = 0.65
    transfer_difficulty: float = 0.80
    eval_tasks_per_split: int = 6
    minimum_seeds: int = 5
    bootstrap_resamples: int = 4000
    minimum_final_success_gain: float = 0.03
    minimum_relative_curve_gain: float = 0.10
    noninferiority_margin: float = 0.01
    meaningful_latency_reduction: float = 0.20
    meaningful_planner_call_reduction: float = 0.20
    meaningful_physical_forward_reduction: float = 0.20
    meaningful_logical_work_reduction: float = 0.20

    def __post_init__(self):
        if not self.seeds or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be non-empty and unique")
        if not self.milestones or tuple(sorted(self.milestones)) != self.milestones:
            raise ValueError("milestones must be sorted and non-empty")
        if any(int(x) <= 0 for x in self.milestones):
            raise ValueError("milestones must be positive")
        if set(self.splits) != {"heldout", "transfer"}:
            raise ValueError("v2.30 ablation campaign requires heldout and transfer splits")
        known = {s.ablation_id for s in EMPIRICAL_ABLATIONS}
        if not self.systems or any(s not in known for s in self.systems):
            raise ValueError("systems must be known executable ablation variants")
        canonical = [s.ablation_id for s in EMPIRICAL_ABLATIONS]
        if [s for s in canonical if s in self.systems] != list(self.systems):
            raise ValueError("systems must preserve canonical ladder order")
        if self.minimum_seeds < 2 or len(self.seeds) < self.minimum_seeds:
            raise ValueError("minimum_seeds must be >=2 and available in seeds")
        if int(self.planner_world_batch_size) < 0:
            raise ValueError("planner_world_batch_size must be >= 0 (0 means maximally batched)")
        for name in ("train_difficulty", "heldout_difficulty", "transfer_difficulty"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> "AblationCampaignConfig":
        section = dict(raw.get("ablation_campaign", raw))
        tuple_fields = {"seeds", "milestones", "splits", "systems"}
        for key in tuple_fields:
            if key in section:
                section[key] = tuple(section[key])
        return cls(**section)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AblationCampaignConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError("campaign YAML must contain a mapping")
        return cls.from_mapping(raw)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for key in ("seeds", "milestones", "splits", "systems"):
            d[key] = list(d[key])
        return d

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())


@dataclass(frozen=True)
class AblationTrainingJob:
    system: str
    seed: int
    milestone: int

    @property
    def key(self) -> str:
        return f"{self.system}__s{self.seed}__m{self.milestone}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AblationCampaignPlan:
    protocol_sha256: str
    config_sha256: str
    jobs: tuple[AblationTrainingJob, ...]
    expected_result_cells: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "awa-v2.30-ablation-campaign-plan-v1",
            "protocol_sha256": self.protocol_sha256,
            "config_sha256": self.config_sha256,
            "training_jobs": [j.to_dict() for j in self.jobs],
            "training_job_count": len(self.jobs),
            "expected_result_cells": int(self.expected_result_cells),
        }

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())


def build_campaign_plan(config: AblationCampaignConfig) -> tuple[Any, AblationCampaignPlan]:
    protocol = build_empirical_ablation_protocol(
        seeds=config.seeds,
        splits=config.splits,
        milestones=config.milestones,
        minimum_seeds=config.minimum_seeds,
    )
    # A subset run is useful for engineering, but a qualification plan must retain
    # a protocol that contains exactly the selected systems.
    if tuple(protocol.systems) != tuple(config.systems):
        from .experiment_protocol import ExperimentProtocol
        bound = dict(protocol.system_config_sha256)
        protocol = ExperimentProtocol(
            protocol_id="aether-v2.31-executable-component-ablation-v1",
            systems=tuple(config.systems), seeds=tuple(config.seeds), tasks=("procedural",),
            splits=tuple(config.splits), milestones=tuple(config.milestones),
            primary_metric="success_rate", minimum_seeds=int(config.minimum_seeds),
            system_config_sha256=tuple((s, bound[s]) for s in config.systems),
        )
    jobs = tuple(
        AblationTrainingJob(system, int(seed), int(milestone))
        for system in config.systems
        for seed in config.seeds
        for milestone in config.milestones
    )
    plan = AblationCampaignPlan(protocol.sha256, config.sha256, jobs, len(protocol.cells()))
    return protocol, plan


def _fixed_tasks(seed: int, difficulty: float, *, label: str):
    factory = ProceduralTaskFactory(seed)
    return [factory.make(stage, difficulty, stage, label) for stage in range(1, 13)]


def _adaptive_tasks(seed: int, difficulty: float, *, label: str):
    factory = ProceduralTaskFactory(seed)
    policy = DeclarativeExplorationPolicy("v229-full-curriculum")
    allocation = ExplorationPolicyCurriculumAdapter().allocate(policy, tuple(range(1, 13)), 12)
    rows = []
    idx = 0
    for stage, count in sorted(allocation.items()):
        for _ in range(int(count)):
            rows.append(factory.make(stage, difficulty, 10_000 + idx, label))
            idx += 1
    return rows


def _evaluation_tasks(seed: int, split: str, difficulty: float, count: int):
    if split not in {"heldout", "transfer"}:
        raise ValueError("evaluation split must be heldout/transfer")
    factory = ProceduralTaskFactory(seed + (700_000 if split == "transfer" else 500_000))
    if split == "heldout":
        stages = (2, 4, 6, 9, 11, 12)
        base = 50_000
    else:
        stages = (4, 7, 9, 10, 11, 12)
        base = 70_000
    rows = []
    for i in range(int(count)):
        stage = stages[i % len(stages)]
        rows.append(factory.make(stage, difficulty, base + i, f"v229-{split}"))
    return rows


def _dataset_mode(system: str) -> str:
    return "adaptive" if VARIANT_BY_ID[system].adaptive_curriculum else "fixed"


def _truncate_npz(source: Path, target: Path, n: int) -> None:
    with np.load(source, allow_pickle=False) as z:
        arrays = {k: np.asarray(z[k])[: int(n)] for k in z.files}
    tmp = target.with_name(target.name + f".tmp-{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)


def _ensure_dataset_series(root: Path, config: AblationCampaignConfig, *, seed: int, mode: str) -> dict[int, Path]:
    if mode not in {"fixed", "adaptive"}:
        raise ValueError("dataset mode must be fixed/adaptive")
    base = root / "datasets" / mode / f"seed-{seed}"
    base.mkdir(parents=True, exist_ok=True)
    wanted = {int(m): base / f"m{int(m)}.npz" for m in config.milestones}
    if all(p.exists() and p.with_suffix(".json").exists() for p in wanted.values()):
        return wanted
    lock = base / ".dataset.lock"
    with AtomicFileLock(lock):
        if all(p.exists() and p.with_suffix(".json").exists() for p in wanted.values()):
            return wanted
        max_m = max(config.milestones)
        tasks = (
            _adaptive_tasks(seed, config.train_difficulty, label="v229-train-adaptive")
            if mode == "adaptive" else
            _fixed_tasks(seed, config.train_difficulty, label="v229-train-fixed")
        )
        # Start from a horizon-based estimate and increase deterministically until
        # the collected real transition count covers the largest milestone.
        episodes = max(1, math.ceil(max_m / max(1, len(tasks) * config.horizon)))
        raw = base / "raw-max.npz"
        while True:
            collect_game_dataset(
                tasks, raw, episodes_per_task=episodes, horizon=config.horizon,
                policy=CoverageArenaPolicy(seed=seed + (991 if mode == "adaptive" else 313)),
            )
            with np.load(raw, allow_pickle=False) as z:
                available = int(len(z["observations"]))
            if available >= max_m:
                break
            scale = max(episodes + 1, math.ceil(episodes * max_m / max(1, available) * 1.10))
            episodes = int(scale)
        raw_meta = json.loads(raw.with_suffix(".json").read_text(encoding="utf-8"))
        previous_hash = ""
        for milestone in config.milestones:
            target = wanted[int(milestone)]
            _truncate_npz(raw, target, int(milestone))
            digest = file_sha256(target)
            meta = {
                "format": "awa-v2.30-ablation-dataset-v1",
                "seed": int(seed), "mode": mode, "target_transitions": int(milestone),
                "available_before_prefix": int(available), "dataset_sha256": digest,
                "parent_dataset_sha256": previous_hash,
                "collector": raw_meta.get("collector", {}),
                "task_ids": [t.task_id for t in tasks],
                "source_collection_report": raw_meta.get("report", {}),
            }
            _atomic_json(target.with_suffix(".json"), meta)
            previous_hash = digest
        raw.unlink(missing_ok=True)
        raw.with_suffix(".json").unlink(missing_ok=True)
    return wanted


def _environment_sha(tasks: Iterable[Any]) -> str:
    return _sha([t.to_dict() for t in tasks])


def _dependency_environment_sha() -> str:
    versions = {}
    for name in ("numpy", "torch", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "missing"
    return _sha({
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": versions,
    })


def _checkpoint_sha(training_checkpoint: str | Path) -> str:
    return file_sha256(training_checkpoint)


def _record_from_eval(*, system: str, seed: int, milestone: int, split: str,
                      ev: Any, checkpoint_sha256: str, config_sha256: str,
                      training_report: Any) -> RunRecord:
    metrics = {
        "success_rate": float(ev.success_rate),
        "episode_return": float(ev.mean_return),
        "constraint_violations": float(ev.constraint_violations),
        "planner_calls_per_episode": float(ev.planner_calls_per_episode),
        # Legacy logical count is retained for backward compatibility. v2.30 makes
        # the execution schedule explicit with physical forwards and batch size.
        "world_model_calls_per_episode": float(ev.world_model_calls_per_episode),
        "logical_world_model_transitions_per_episode": float(getattr(ev, "logical_world_model_transitions_per_episode", ev.world_model_calls_per_episode)),
        "physical_world_model_forwards_per_episode": float(getattr(ev, "physical_world_model_forwards_per_episode", 0.0)),
        "mean_world_model_batch_size": float(getattr(ev, "mean_world_model_batch_size", 0.0)),
        "logical_transitions_per_forward": float(getattr(ev, "logical_transitions_per_forward", 0.0)),
        "inference_latency_ms": float(ev.inference_latency_ms),
        "wall_clock_seconds": float(training_report.training_seconds),
        "accelerator_wall_hours": float(getattr(training_report, "accelerator_wall_hours", 0.0)),
        "peak_cuda_memory_bytes": float(getattr(training_report, "peak_cuda_memory_bytes", 0)),
        "source_transitions": float(getattr(training_report, "source_transitions", milestone)),
        "effective_training_rows": float(getattr(training_report, "effective_training_rows", milestone)),
    }
    return RunRecord(
        system=system, seed=int(seed), task="procedural", split=split,
        transitions=int(milestone), metrics=metrics,
        checkpoint_sha256=checkpoint_sha256, config_sha256=config_sha256,
    )


def _run_id(record: RunRecord) -> str:
    return "|".join(map(str, (record.system, record.seed, record.task, record.split, record.transitions)))


class AblationCampaignRunner:
    """Resumable, shardable execution of the preregistered v2.30 ablation matrix.

    One training job serves both heldout and transfer evaluation cells, preventing
    accidental train-twice-per-split compute inflation. Every completed job commits
    an immutable receipt only after both split evaluations and hashes exist.
    """

    def __init__(self, config: AblationCampaignConfig, out_dir: str | Path):
        self.config = config
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.protocol, self.plan = build_campaign_plan(config)
        self.plan_path = self.out_dir / "campaign_plan.json"
        self.protocol_path = self.out_dir / "protocol.json"
        self.source_sha256 = _tree_sha256(Path(__file__).resolve().parents[1])
        self.dependency_sha256 = _dependency_environment_sha()
        self._freeze_plan()

    def _freeze_plan(self) -> None:
        payload = self.plan.to_dict() | {"plan_sha256": self.plan.sha256}
        if self.plan_path.exists():
            old = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if old.get("plan_sha256") != self.plan.sha256:
                raise ValueError("existing campaign plan differs from requested config; use a new out-dir")
        else:
            _atomic_json(self.plan_path, payload)
        protocol_payload = {
            "format": "awa-v2.30-ablation-protocol-binding-v1",
            "protocol": self.protocol.to_dict(), "protocol_sha256": self.protocol.sha256,
            "config": self.config.to_dict(), "config_sha256": self.config.sha256,
        }
        if self.protocol_path.exists():
            old = json.loads(self.protocol_path.read_text(encoding="utf-8"))
            if old.get("protocol_sha256") != self.protocol.sha256 or old.get("config_sha256") != self.config.sha256:
                raise ValueError("existing protocol binding differs from requested config")
        else:
            _atomic_json(self.protocol_path, protocol_payload)

    def _job_dir(self, job: AblationTrainingJob) -> Path:
        return self.out_dir / "jobs" / job.system / f"seed-{job.seed}" / f"m{job.milestone}"

    def _receipt_path(self, job: AblationTrainingJob) -> Path:
        return self._job_dir(job) / "job_receipt.json"

    def _valid_receipt(self, job: AblationTrainingJob) -> bool:
        path = self._receipt_path(job)
        if not path.exists():
            return False
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("format") != "awa-v2.30-ablation-job-receipt-v1":
                return False
            if payload.get("job") != job.to_dict():
                return False
            if payload.get("protocol_sha256") != self.protocol.sha256:
                return False
            if payload.get("config_sha256") != self.config.sha256:
                return False
            if len(payload.get("records", [])) != len(self.config.splits):
                return False
            ck = Path(payload["training"]["checkpoint"])
            return ck.exists() and file_sha256(ck) == payload.get("checkpoint_sha256")
        except Exception:
            return False

    def status(self) -> dict[str, Any]:
        completed = [j.key for j in self.plan.jobs if self._valid_receipt(j)]
        pending = [j.key for j in self.plan.jobs if j.key not in set(completed)]
        return {
            "format": "awa-v2.30-ablation-campaign-status-v1",
            "plan_sha256": self.plan.sha256,
            "protocol_sha256": self.protocol.sha256,
            "training_jobs": len(self.plan.jobs),
            "completed_jobs": len(completed), "pending_jobs": len(pending),
            "completed": completed, "pending": pending,
        }

    def _execute_job(self, job: AblationTrainingJob) -> dict[str, Any]:
        final_dir = self._job_dir(job)
        receipt = self._receipt_path(job)
        if self._valid_receipt(job):
            return json.loads(receipt.read_text(encoding="utf-8"))
        lock = final_dir.with_suffix(".lock")
        with AtomicFileLock(lock):
            if self._valid_receipt(job):
                return json.loads(receipt.read_text(encoding="utf-8"))
            mode = _dataset_mode(job.system)
            datasets = _ensure_dataset_series(self.out_dir, self.config, seed=job.seed, mode=mode)
            dataset = datasets[job.milestone]
            dataset_sha = file_sha256(dataset)
            variant = VARIANT_BY_ID[job.system]
            train_tasks = (
                _adaptive_tasks(job.seed, self.config.train_difficulty, label="v229-train-adaptive")
                if mode == "adaptive" else
                _fixed_tasks(job.seed, self.config.train_difficulty, label="v229-train-fixed")
            )
            eval_tasks = {
                "heldout": _evaluation_tasks(job.seed, "heldout", self.config.heldout_difficulty, self.config.eval_tasks_per_split),
                "transfer": _evaluation_tasks(job.seed, "transfer", self.config.transfer_difficulty, self.config.eval_tasks_per_split),
            }
            device = str(resolve_device(self.config.device))
            attempt_parent = final_dir.parent
            attempt_parent.mkdir(parents=True, exist_ok=True)
            attempt = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.attempt-", dir=attempt_parent))
            try:
                train_dir = attempt / "trained"
                tr, heldout = train_and_evaluate_variant(
                    job.system, dataset, train_tasks, eval_tasks["heldout"], train_dir,
                    device=device, sequence_length=self.config.sequence_length, hidden=self.config.hidden,
                    world_epochs=self.config.world_epochs, actor_epochs=self.config.actor_epochs,
                    representation_epochs=self.config.representation_epochs,
                    calibration_epochs=self.config.calibration_epochs, batch_size=self.config.batch_size,
                    seed=job.seed, horizon=self.config.horizon, voc_epochs=self.config.voc_epochs,
                    curriculum_mode=mode,
                    planner_world_batch_size=(None if self.config.planner_world_batch_size == 0 else self.config.planner_world_batch_size),
                )
                transfer = evaluate_trained_variant(
                    job.system, train_dir, eval_tasks["transfer"], device=device,
                    horizon=self.config.horizon, seed_offset=53,
                    voc_tasks=train_tasks, voc_epochs=self.config.voc_epochs,
                    planner_world_batch_size=(None if self.config.planner_world_batch_size == 0 else self.config.planner_world_batch_size),
                )
                checkpoint = Path(tr.checkpoint)
                ck_sha = _checkpoint_sha(checkpoint)
                spec_sha = next(s.sha256 for s in EMPIRICAL_ABLATIONS if s.ablation_id == job.system)
                records = [
                    _record_from_eval(system=job.system, seed=job.seed, milestone=job.milestone,
                                      split="heldout", ev=heldout, checkpoint_sha256=ck_sha,
                                      config_sha256=spec_sha, training_report=tr),
                    _record_from_eval(system=job.system, seed=job.seed, milestone=job.milestone,
                                      split="transfer", ev=transfer, checkpoint_sha256=ck_sha,
                                      config_sha256=spec_sha, training_report=tr),
                ]
                prov = {}
                for record in records:
                    split_tasks = eval_tasks[record.split]
                    prov[_run_id(record)] = RunProvenance(
                        run_id=_run_id(record), source_sha256=self.source_sha256,
                        dataset_sha256=dataset_sha,
                        environment_sha256=_environment_sha(split_tasks),
                        dependency_lock_sha256=self.dependency_sha256,
                        scenario_ids=tuple(t.task_id for t in split_tasks),
                    )
                payload = {
                    "format": "awa-v2.30-ablation-job-receipt-v1",
                    "job": job.to_dict(), "variant": variant.to_dict(),
                    "protocol_sha256": self.protocol.sha256, "config_sha256": self.config.sha256,
                    "dataset": str(dataset), "dataset_sha256": dataset_sha,
                    "checkpoint_sha256": ck_sha,
                    "training": tr.to_dict(),
                    "execution_fingerprint": {
                        "format": "awa-v2.30-execution-fingerprint-v1",
                        "logical_data_budget_transitions": int(job.milestone),
                        "effective_training_rows": int(getattr(tr, "effective_training_rows", job.milestone)),
                        "planner_world_batch_size": int(self.config.planner_world_batch_size),
                        "planner_world_batch_size_semantics": "0=maximally_batched",
                        "heldout": heldout.to_dict(),
                        "transfer": transfer.to_dict(),
                        "accelerator_energy_joules": None,
                        "energy_measurement": "not_measured",
                    },
                    "records": [r.to_dict() for r in records],
                    "provenance": {k: v.to_dict() for k, v in prov.items()},
                }
                # Paths created inside the attempt directory must remain valid after commit.
                old_prefix = str(attempt)
                new_prefix = str(final_dir)
                def rewrite(obj):
                    if isinstance(obj, dict): return {k: rewrite(v) for k, v in obj.items()}
                    if isinstance(obj, list): return [rewrite(v) for v in obj]
                    if isinstance(obj, str) and obj.startswith(old_prefix): return new_prefix + obj[len(old_prefix):]
                    return obj
                payload = rewrite(payload)
                _atomic_json(attempt / "job_receipt.json", payload)
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                os.replace(attempt, final_dir)
                return payload
            except Exception:
                shutil.rmtree(attempt, ignore_errors=True)
                raise

    def execute(self, *, max_jobs: int | None = None, worker_index: int = 0, worker_count: int = 1) -> dict[str, Any]:
        if worker_count < 1 or not 0 <= worker_index < worker_count:
            raise ValueError("worker_index must be in [0, worker_count)")
        selected = [j for i, j in enumerate(self.plan.jobs) if i % worker_count == worker_index]
        completed_now = 0
        failures = []
        for job in selected:
            if self._valid_receipt(job):
                continue
            if max_jobs is not None and completed_now >= int(max_jobs):
                break
            try:
                self._execute_job(job)
                completed_now += 1
            except TimeoutError as exc:
                failures.append({"job": job.to_dict(), "error": str(exc), "kind": "lock_timeout"})
            except Exception as exc:
                failures.append({"job": job.to_dict(), "error": f"{type(exc).__name__}: {exc}", "kind": "execution"})
                _atomic_json(self.out_dir / "failures" / f"{job.key}.json", failures[-1])
        consolidated = self.consolidate()
        return {
            "status": "ok" if not failures else "partial_failure",
            "worker_index": int(worker_index), "worker_count": int(worker_count),
            "completed_now": int(completed_now), "failures": failures,
            "campaign": self.status(), "consolidated": consolidated,
        }

    def consolidate(self) -> dict[str, Any]:
        records: list[RunRecord] = []
        provenance: dict[str, RunProvenance] = {}
        for job in self.plan.jobs:
            if not self._valid_receipt(job):
                continue
            payload = json.loads(self._receipt_path(job).read_text(encoding="utf-8"))
            for raw in payload["records"]:
                record = RunRecord(**raw)
                records.append(record)
            for key, raw in payload["provenance"].items():
                provenance[key] = RunProvenance(
                    run_id=raw["run_id"], source_sha256=raw["source_sha256"],
                    dataset_sha256=raw["dataset_sha256"], environment_sha256=raw["environment_sha256"],
                    dependency_lock_sha256=raw["dependency_lock_sha256"], scenario_ids=tuple(raw["scenario_ids"]),
                )
        records.sort(key=lambda r: (r.system, r.seed, r.transitions, r.split))
        records_path = self.out_dir / "records.jsonl"
        tmp = records_path.with_name(records_path.name + f".tmp-{os.getpid()}")
        tmp.write_text("".join(json.dumps(r.to_dict(), sort_keys=True) + "\n" for r in records), encoding="utf-8")
        os.replace(tmp, records_path)
        _atomic_json(self.out_dir / "provenance.json", {k: v.to_dict() for k, v in sorted(provenance.items())})
        if records:
            receipt = build_protocol_receipt(self.protocol, records, provenance, baseline=self.config.systems[-1])
        else:
            receipt = {
                "format": "awa-v2.30-empty-protocol-receipt-v1", "status": "FAIL",
                "failures": ["no completed records"], "protocol_sha256": self.protocol.sha256,
            }
        _atomic_json(self.out_dir / "protocol_receipt.json", receipt)
        from .ablation_report import build_ablation_decision_report
        report = build_ablation_decision_report(records, self.protocol, self.config)
        from .ablation_report import build_compute_schedule_report
        compute_report = build_compute_schedule_report(records, self.protocol, self.config)
        _atomic_json(self.out_dir / "ablation_report.json", report)
        _atomic_json(self.out_dir / "compute_schedule_report.json", compute_report)
        (self.out_dir / "ablation_report.md").write_text(render_ablation_report_markdown(report), encoding="utf-8")
        (self.out_dir / "compute_schedule_report.md").write_text(render_compute_schedule_report_markdown(compute_report), encoding="utf-8")
        return {
            "records": len(records), "protocol_status": receipt.get("status"),
            "report_status": report.get("status"), "compute_report_status": compute_report.get("status"),
        }


def render_ablation_report_markdown(report: dict[str, Any]) -> str:
    lines = ["# Aether v2.30 Ablation Decision Report", "", f"Status: **{report.get('status','UNKNOWN')}**", ""]
    if report.get("reason"):
        lines += [report["reason"], ""]
    lines += ["| Added mechanism | Baseline → Candidate | Decision | Final success Δ | Curve Δ | Physical forwards Δ | Logical work Δ |", "|---|---|---:|---:|---:|---:|---:|"]
    for row in report.get("marginal_components", []):
        lines.append(
            f"| {row['mechanism']} | {row['baseline']} → {row['candidate']} | {row['decision']} | "
            f"{row.get('final_success_delta_mean', float('nan')):.4f} | {row.get('relative_curve_gain_mean', float('nan')):.4f} | "
            f"{row.get('physical_forward_reduction_fraction', 0.0):.4f} | {row.get('logical_world_model_work_reduction_fraction', 0.0):.4f} |"
        )
    lines += ["", "Decisions are generated only after the preregistered matrix is complete. `REMOVE_CANDIDATE` means the mechanism failed the preregistered value thresholds in this benchmark; it is not a universal claim about the mechanism.", ""]
    return "\n".join(lines)


def render_compute_schedule_report_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Aether v2.30 Compute Schedule Report", "",
        f"Status: **{report.get('status', 'UNKNOWN')}**", "",
        "| System | Logical WM transitions/episode | Physical WM forwards/episode | Effective batch | Inference ms | Training accelerator-hours | Peak CUDA bytes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.get("systems", []):
        lines.append(
            f"| {row['system']} | {row.get('mean_logical_world_model_transitions_per_episode',0.0):.3f} | "
            f"{row.get('mean_physical_world_model_forwards_per_episode',0.0):.3f} | "
            f"{row.get('logical_transitions_per_physical_forward',0.0):.3f} | "
            f"{row.get('mean_inference_latency_ms',0.0):.3f} | "
            f"{row.get('mean_training_accelerator_wall_hours',0.0):.6f} | "
            f"{row.get('max_peak_cuda_memory_bytes',0)} |"
        )
    lines += [
        "",
        "Energy is not estimated. `accelerator_energy_joules` remains null unless a hardware-backed meter records it.",
        "",
    ]
    return "\n".join(lines)
