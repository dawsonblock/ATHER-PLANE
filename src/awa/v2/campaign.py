from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from typing import Any

import numpy as np
import torch

from .experiment_ledger import ExperimentLedger
from .replay import ShardedReplayStore
from .curriculum import ProceduralTaskFactory
from .game import LogicalArenaTeacher, RandomArenaPolicy, collect_game_dataset, train_game_stack
from .telemetry import TelemetryRecorder


def _sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _system_ram_gb() -> float | None:
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return float(pages * page_size) / (1024**3)
    except Exception:
        return None


@dataclass(frozen=True)
class ResourceProfile:
    name: str
    device: str
    precision: str
    detected_vram_gb: float | None
    detected_system_ram_gb: float | None
    max_batch_size: int
    max_sequence_length: int
    recommended_ensemble_members: int
    recommended_planner_budget: int
    replay_shard_size: int

    def to_dict(self):
        return asdict(self)


def detect_resource_profile(
    device: str = "auto",
    *,
    vram_gb: float | None = None,
    system_ram_gb: float | None = None,
) -> ResourceProfile:
    """Resolve a conservative local training profile.

    Explicit ``vram_gb`` is supported for planning/tests without requiring that the
    target accelerator be installed in the current process.
    """
    dev = str(device).lower()
    if dev == "auto":
        if torch.cuda.is_available():
            dev = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            dev = "mps"
        else:
            dev = "cpu"
    if dev not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda or mps")
    detected_vram = vram_gb
    if detected_vram is None and dev == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        detected_vram = float(props.total_memory) / (1024**3)
    ram = float(system_ram_gb) if system_ram_gb is not None else _system_ram_gb()
    if dev == "cpu":
        return ResourceProfile("cpu", dev, "fp32", None, ram, 32, 8, 1, 16, 25_000)
    if dev == "mps":
        return ResourceProfile("apple-silicon", dev, "fp32", detected_vram, ram, 64, 12, 1, 24, 50_000)
    gb = float(detected_vram or 0.0)
    bf16 = bool(torch.cuda.is_available() and getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    precision = "bf16" if bf16 else "fp16"
    if gb <= 13.0:
        return ResourceProfile("consumer-12gb", dev, precision, detected_vram, ram, 64, 16, 1, 32, 50_000)
    if gb <= 28.0:
        return ResourceProfile("consumer-24gb", dev, precision, detected_vram, ram, 128, 24, 3, 64, 100_000)
    return ResourceProfile("accelerator-48gb-plus", dev, precision, detected_vram, ram, 256, 32, 5, 128, 200_000)


@dataclass(frozen=True)
class CampaignStage:
    name: str
    target_transitions: int
    curriculum_stages: tuple[int, ...]
    difficulty: float
    sequence_length: int
    hidden: int
    world_epochs: int
    actor_epochs: int
    calibration_epochs: int
    batch_size: int
    episodes_per_task: int = 1
    tasks_per_stage: int = 2
    horizon: int = 100
    teacher_fraction: float = 0.8

    def __post_init__(self):
        if not self.name:
            raise ValueError("campaign stage name cannot be empty")
        if int(self.target_transitions) <= 0:
            raise ValueError("target_transitions must be > 0")
        if not self.curriculum_stages or any(int(s) < 1 or int(s) > 12 for s in self.curriculum_stages):
            raise ValueError("curriculum_stages must contain stages in [1,12]")
        if not 0.0 <= float(self.difficulty) <= 1.0:
            raise ValueError("difficulty must be in [0,1]")
        if min(self.sequence_length, self.hidden, self.world_epochs, self.actor_epochs, self.batch_size, self.horizon) <= 0:
            raise ValueError("stage numeric training parameters must be > 0")
        if int(self.calibration_epochs) < 0 or int(self.episodes_per_task) <= 0 or int(self.tasks_per_stage) <= 0:
            raise ValueError("stage episode/task/calibration values are invalid")
        if not 0.0 <= float(self.teacher_fraction) <= 1.0:
            raise ValueError("teacher_fraction must be in [0,1]")

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "CampaignStage":
        return cls(
            name=str(raw["name"]),
            target_transitions=int(raw["target_transitions"]),
            curriculum_stages=tuple(int(x) for x in raw["curriculum_stages"]),
            difficulty=float(raw.get("difficulty", 0.35)),
            sequence_length=int(raw.get("sequence_length", 8)),
            hidden=int(raw.get("hidden", 96)),
            world_epochs=int(raw.get("world_epochs", 2)),
            actor_epochs=int(raw.get("actor_epochs", 4)),
            calibration_epochs=int(raw.get("calibration_epochs", 2)),
            batch_size=int(raw.get("batch_size", 64)),
            episodes_per_task=int(raw.get("episodes_per_task", 1)),
            tasks_per_stage=int(raw.get("tasks_per_stage", 2)),
            horizon=int(raw.get("horizon", 100)),
            teacher_fraction=float(raw.get("teacher_fraction", 0.8)),
        )

    def to_dict(self):
        data = asdict(self)
        data["curriculum_stages"] = list(self.curriculum_stages)
        return data


@dataclass(frozen=True)
class PromotionPolicy:
    min_actor_success: float | None = None
    min_planner_success: float | None = None
    max_risk_brier: float | None = None
    max_horizon_rmse: float | None = None
    max_actor_success_regression: float = 0.15
    max_planner_success_regression: float = 0.10

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "PromotionPolicy":
        raw = raw or {}
        return cls(**{k: raw.get(k) for k in cls.__dataclass_fields__ if k in raw})

    def assess(self, candidate: dict[str, Any], baseline: dict[str, Any] | None = None) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        numeric = [candidate.get("actor_success_rate"), candidate.get("planner_success_rate"), candidate.get("actor_mean_return"), candidate.get("planner_mean_return")]
        if any(v is None or not math.isfinite(float(v)) for v in numeric):
            reasons.append("candidate contains non-finite core metrics")
        actor = float(candidate.get("actor_success_rate", 0.0))
        planner = float(candidate.get("planner_success_rate", 0.0))
        if self.min_actor_success is not None and actor < float(self.min_actor_success):
            reasons.append(f"actor success {actor:.4f} below {float(self.min_actor_success):.4f}")
        if self.min_planner_success is not None and planner < float(self.min_planner_success):
            reasons.append(f"planner success {planner:.4f} below {float(self.min_planner_success):.4f}")
        risk = candidate.get("risk_brier")
        if self.max_risk_brier is not None and risk is not None and float(risk) > float(self.max_risk_brier):
            reasons.append(f"risk Brier {float(risk):.4f} exceeds {float(self.max_risk_brier):.4f}")
        if self.max_horizon_rmse is not None:
            vals = [float(v) for v in (candidate.get("horizon_rmse") or {}).values()]
            if vals and max(vals) > float(self.max_horizon_rmse):
                reasons.append(f"horizon RMSE {max(vals):.4f} exceeds {float(self.max_horizon_rmse):.4f}")
        if baseline:
            ba = float(baseline.get("actor_success_rate", actor))
            bp = float(baseline.get("planner_success_rate", planner))
            if actor + float(self.max_actor_success_regression) < ba:
                reasons.append("actor success regressed beyond allowed margin")
            if planner + float(self.max_planner_success_regression) < bp:
                reasons.append("planner success regressed beyond allowed margin")
        return not reasons, reasons


@dataclass(frozen=True)
class CampaignComputeEstimate:
    final_target_transitions: int
    stage_count: int
    approximate_world_updates: int
    approximate_actor_updates: int
    approximate_sequence_transition_passes: int
    estimated_gpu_hours: float | None

    def to_dict(self):
        return asdict(self)


def estimate_campaign_compute(stages: list[CampaignStage], *, samples_per_second: float | None = None) -> CampaignComputeEstimate:
    world_updates = 0
    actor_updates = 0
    sequence_passes = 0
    total_sample_passes = 0
    for stage in stages:
        sequences = max(1, stage.target_transitions - stage.sequence_length + 1)
        world_updates += math.ceil(sequences / stage.batch_size) * stage.world_epochs
        actor_updates += math.ceil(stage.target_transitions / stage.batch_size) * stage.actor_epochs
        sequence_passes += sequences * stage.world_epochs
        total_sample_passes += sequences * stage.world_epochs + stage.target_transitions * stage.actor_epochs
    hours = None
    if samples_per_second is not None:
        rate = float(samples_per_second)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("samples_per_second must be finite and > 0")
        hours = float(total_sample_passes / rate / 3600.0)
    return CampaignComputeEstimate(stages[-1].target_transitions, len(stages), world_updates, actor_updates, sequence_passes, hours)


class _MixedArenaPolicy:
    def __init__(self, teacher_fraction: float, seed: int):
        self.teacher = LogicalArenaTeacher()
        self.random = RandomArenaPolicy(seed)
        self.teacher_fraction = float(teacher_fraction)
        self.rng = np.random.default_rng(int(seed) + 91_337)

    def act(self, env, observation=None):
        source = self.teacher if self.rng.random() < self.teacher_fraction else self.random
        return source.act(env, observation)


class CheckpointRegistry:
    """Atomic stable-checkpoint registry with retained promotion history."""
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "registry.json"
        if self.path.exists():
            self.state = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.state = {"format": "awa-v2.13-checkpoint-registry-v1", "stable": None, "history": []}
            self._save()

    def _save(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    def current(self) -> dict[str, Any] | None:
        return self.state.get("stable")

    def promote(self, stage_name: str, world: str | Path, actor: str | Path, metrics: dict[str, Any]) -> dict[str, Any]:
        world = Path(world); actor = Path(actor)
        if not world.exists() or not actor.exists():
            raise FileNotFoundError("candidate world/actor checkpoint missing")
        stamp = len(self.state["history"])
        world_dst = self.root / f"stable-{stamp:03d}-{stage_name}-world.pt"
        actor_dst = self.root / f"stable-{stamp:03d}-{stage_name}-actor.pt"
        shutil.copy2(world, world_dst); shutil.copy2(actor, actor_dst)
        entry = {
            "stage": str(stage_name),
            "world": str(world_dst),
            "actor": str(actor_dst),
            "world_sha256": _sha256(world_dst),
            "actor_sha256": _sha256(actor_dst),
            "metrics": metrics,
            "promoted_at": time.time(),
        }
        self.state["history"].append(entry)
        self.state["stable"] = entry
        self._save()
        return entry

    def rollback(self) -> dict[str, Any] | None:
        if len(self.state["history"]) <= 1:
            return self.current()
        self.state["history"].pop()
        self.state["stable"] = self.state["history"][-1]
        self._save()
        return self.current()

    def verify(self) -> list[str]:
        bad = []
        for entry in self.state.get("history", []):
            for key in ("world", "actor"):
                path = Path(entry[key])
                expected = entry[f"{key}_sha256"]
                if not path.exists() or _sha256(path) != expected:
                    bad.append(str(path))
        return bad


class GameTrainingCampaign:
    """Resumable cumulative procedural-game training campaign.

    The campaign grows one append-only replay store. Each stage materializes the
    cumulative replay, trains the v2.10+ belief/world/actor stack, applies promotion
    gates, and keeps the last stable checkpoint if the candidate fails.
    """
    def __init__(self, config: dict[str, Any], out_dir: str | Path):
        self.config = json.loads(json.dumps(config))
        self.out = Path(out_dir); self.out.mkdir(parents=True, exist_ok=True)
        self.stages = [CampaignStage.from_dict(x) for x in self.config.get("stages", [])]
        if not self.stages:
            raise ValueError("campaign config requires stages")
        targets = [s.target_transitions for s in self.stages]
        if targets != sorted(targets) or len(set(targets)) != len(targets):
            raise ValueError("campaign target_transitions must be strictly increasing")
        runtime = self.config.get("runtime") or {}
        self.profile = detect_resource_profile(
            runtime.get("device", "auto"),
            vram_gb=runtime.get("vram_gb"),
            system_ram_gb=runtime.get("system_ram_gb"),
        )
        requested_precision = str(runtime.get("precision", self.profile.precision)).lower()
        if requested_precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32, bf16 or fp16")
        if requested_precision == "fp16" and self.profile.device != "cuda":
            raise ValueError("fp16 campaign training requires CUDA")
        self.precision = requested_precision
        self.strict_resources = bool(runtime.get("strict_resources", False))
        replay_cfg = self.config.get("replay") or {}
        self.replay = ShardedReplayStore(self.out / "replay", shard_size=int(replay_cfg.get("shard_size", self.profile.replay_shard_size)))
        ledger_config = json.loads(json.dumps(self.config))
        ledger_config["_resolved_runtime"] = {**self.profile.to_dict(), "effective_precision": self.precision}
        self.ledger = ExperimentLedger(self.out, ledger_config, experiment_id="training_campaign")
        self.registry = CheckpointRegistry(self.out / "checkpoints")
        self.promotion = PromotionPolicy.from_dict(self.config.get("promotion"))
        self.seed = int(self.config.get("seed", 213))
        self.factory = ProceduralTaskFactory(self.seed)
        telemetry_cfg = self.config.get("telemetry") or {}
        self.telemetry = TelemetryRecorder(self.out / "telemetry.jsonl", run_id=str(telemetry_cfg.get("run_id", "training_campaign"))) if bool(telemetry_cfg.get("enabled", True)) else None

    def _effective_stage(self, stage: CampaignStage) -> CampaignStage:
        if self.strict_resources and (stage.batch_size > self.profile.max_batch_size or stage.sequence_length > self.profile.max_sequence_length):
            raise ValueError(f"stage {stage.name!r} exceeds resource profile {self.profile.name}")
        return CampaignStage(
            **{
                **stage.to_dict(),
                "curriculum_stages": tuple(stage.curriculum_stages),
                "batch_size": min(stage.batch_size, self.profile.max_batch_size),
                "sequence_length": min(stage.sequence_length, self.profile.max_sequence_length),
            }
        )

    def _task_batch(self, stage: CampaignStage, batch_index: int):
        tasks = []
        for s in stage.curriculum_stages:
            for i in range(stage.tasks_per_stage):
                idx = batch_index * 10_000 + s * 100 + i
                tasks.append(self.factory.make(s, stage.difficulty, idx, f"campaign:{stage.name}"))
        return tasks

    def _collect_to_target(self, stage: CampaignStage) -> dict[str, Any]:
        batch_index = len(self.replay.shards)
        added = 0
        while len(self.replay) < stage.target_transitions:
            tasks = self._task_batch(stage, batch_index)
            policy = _MixedArenaPolicy(stage.teacher_fraction, self.seed + batch_index)
            with tempfile.TemporaryDirectory() as td:
                path = Path(td) / "batch.npz"
                report = collect_game_dataset(
                    tasks, path, episodes_per_task=stage.episodes_per_task,
                    horizon=stage.horizon, policy=policy,
                )
                with np.load(path, allow_pickle=False) as z:
                    arrays = {k: np.asarray(z[k]) for k in z.files}
                self.replay.append(arrays)
                added += int(report.transitions)
            batch_index += 1
        return {"target": stage.target_transitions, "transitions": len(self.replay), "added": added, "shards": len(self.replay.shards)}

    def _train_stage(self, stage: CampaignStage, stage_dir: Path) -> dict[str, Any]:
        dataset = self.replay.materialize(stage_dir / "campaign_replay.npz", max_transitions=stage.target_transitions)
        eval_tasks = self._task_batch(stage, 999_999)[: max(2, min(8, len(stage.curriculum_stages) * 2))]
        report = train_game_stack(
            dataset, eval_tasks, stage_dir / "train",
            sequence_length=stage.sequence_length,
            horizons=(1, 2, 4),
            world_epochs=stage.world_epochs,
            actor_epochs=stage.actor_epochs,
            calibration_epochs=stage.calibration_epochs,
            batch_size=stage.batch_size,
            hidden=stage.hidden,
            seed=self.seed + stage.target_transitions,
            device=self.profile.device,
            precision=self.precision,
            one_step_aux_epochs=1,
        )
        return {
            "dataset": str(dataset),
            "report": report.to_dict(),
            "world": str(stage_dir / "train" / "game_world.pt"),
            "actor": str(stage_dir / "train" / "game_actor.pt"),
        }

    def run(self, *, stop_after_stage: str | None = None) -> dict[str, Any]:
        stage_results = []
        for raw_stage in self.stages:
            stage = self._effective_stage(raw_stage)
            if self.telemetry is not None:
                self.telemetry.event("stage_start", step=len(self.replay), metadata={"stage": stage.name, "target_transitions": stage.target_transitions})
            stage_dir = self.out / "stages" / stage.name
            stage_dir.mkdir(parents=True, exist_ok=True)
            collect = self.ledger.run(f"{stage.name}:collect", lambda s=stage: self._collect_to_target(s))
            if self.telemetry is not None:
                self.telemetry.event("collection_complete", step=len(self.replay), metrics={"replay_transitions": len(self.replay), "added": collect.get("added", 0)}, metadata={"stage": stage.name})
            train = self.ledger.run(f"{stage.name}:train", lambda s=stage, d=stage_dir: self._train_stage(s, d))
            if self.telemetry is not None:
                tm = train.get("report", {})
                numeric = {k: v for k, v in tm.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}
                self.telemetry.event("training_complete", step=len(self.replay), metrics=numeric, metadata={"stage": stage.name})
            baseline = self.registry.current()
            baseline_metrics = None if baseline is None else baseline.get("metrics")
            def promote_phase():
                ok, reasons = self.promotion.assess(train["report"], baseline_metrics)
                entry = None
                if ok:
                    entry = self.registry.promote(stage.name, train["world"], train["actor"], train["report"])
                return {"promoted": ok, "reasons": reasons, "stable": entry or self.registry.current()}
            promotion = self.ledger.run(f"{stage.name}:promote", promote_phase)
            if self.telemetry is not None:
                self.telemetry.event("promotion", step=len(self.replay), metrics={"promoted": int(bool(promotion.get("promoted")))}, metadata={"stage": stage.name, "reasons": promotion.get("reasons", [])})
            result = {"stage": stage.name, "effective_stage": stage.to_dict(), "collection": collect, "training": train["report"], "promotion": promotion}
            stage_results.append(result)
            if not promotion["promoted"] and bool(self.config.get("stop_on_failed_promotion", True)):
                break
            if stop_after_stage is not None and stage.name == stop_after_stage:
                break
        summary = {
            "format": "awa-v2.14-training-campaign-v1",
            "resource_profile": self.profile.to_dict(),
            "precision": self.precision,
            "replay_transitions": len(self.replay),
            "replay_integrity_failures": self.replay.verify(),
            "checkpoint_integrity_failures": self.registry.verify(),
            "stable_checkpoint": self.registry.current(),
            "stages": stage_results,
            "compute_estimate": estimate_campaign_compute(self.stages).to_dict(),
            "telemetry": None if self.telemetry is None else self.telemetry.summarize().to_dict(),
        }
        path = self.out / "campaign_report.json"
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return summary


def campaign_plan(config: dict[str, Any]) -> dict[str, Any]:
    stages = [CampaignStage.from_dict(x) for x in config.get("stages", [])]
    if not stages:
        raise ValueError("campaign config requires stages")
    runtime = config.get("runtime") or {}
    profile = detect_resource_profile(runtime.get("device", "auto"), vram_gb=runtime.get("vram_gb"), system_ram_gb=runtime.get("system_ram_gb"))
    compute = config.get("compute") or {}
    estimate = estimate_campaign_compute(stages, samples_per_second=compute.get("measured_samples_per_second"))
    return {"format": "awa-v2.14-campaign-plan-v1", "resource_profile": profile.to_dict(), "stages": [s.to_dict() for s in stages], "compute_estimate": estimate.to_dict()}
