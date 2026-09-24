from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
import hashlib
import json
import math
import platform
import shutil
import tempfile
import time

import numpy as np
import torch

from .curriculum import ProceduralTaskFactory
from .empirical_closure import RunRecord
from .evidence_integrity import RunProvenance, canonical_sha256
from .evolving_campaign import CampaignCell, CampaignEvidence, CampaignRunner
from .meta_exploration import DeclarativeExplorationPolicy, EvidenceClass
from .game import LogicalArenaTeacher, collect_game_dataset, train_game_stack
from .game.collectors import CoverageArenaPolicy, FrozenAetherArenaPolicy, collector_receipt
from .game.teacher import RandomArenaPolicy
from .game.procedural_runtime import ProceduralActorEvaluation, evaluate_procedural_actor


def _file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_fingerprint() -> str:
    """Hash the installed Aether Python source tree, not the mutable run directory."""
    root = Path(__file__).resolve().parents[2]
    rows: list[tuple[str, str]] = []
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rows.append((path.relative_to(root).as_posix(), _file_sha256(path)))
    return canonical_sha256(rows)


def _dependency_fingerprint() -> str:
    return canonical_sha256({
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "mps_available": bool(hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    })


@dataclass(frozen=True)
class NativeProceduralRunnerConfig:
    """Concrete v2.23 runner for Aether's built-in procedural environment.

    The declarative exploration policy is translated into an explicit curriculum
    allocation; it never changes the evaluator, metric definitions, or evidence
    schema. Held-out and transfer task generation use disjoint deterministic index
    ranges so evaluation scenarios cannot enter the training dataset.
    """

    device: str = "auto"
    precision: str = "fp32"
    curriculum_stages: tuple[int, ...] = tuple(range(1, 13))
    tasks_per_batch: int = 8
    episodes_per_task: int = 1
    horizon: int = 100
    train_difficulty: float = 0.45
    eval_difficulty: float = 0.65
    eval_tasks: int = 6
    sequence_length: int = 5
    hidden: int = 64
    world_epochs: int = 2
    actor_epochs: int = 4
    calibration_epochs: int = 2
    batch_size: int = 64
    one_step_aux_epochs: int = 1
    max_collection_batches: int = 10_000
    minimum_transitions: int = 32
    reuse_training_artifacts: bool = True
    cumulative_milestone_training: bool = True
    run_planner_diagnostics: bool = False
    evaluation_seed_offset: int = 31
    collection_mode: str = "teacher"
    bootstrap_collection_mode: str = "coverage"
    collector_planner_budget: int = 16
    collector_planner_horizon: int = 4
    collector_planner_stride: int = 4

    def __post_init__(self):
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be auto/cpu/cuda/mps")
        if self.precision not in {"fp32", "bf16", "fp16"}:
            raise ValueError("precision must be fp32/bf16/fp16")
        if self.collection_mode not in {"teacher", "random", "coverage", "aether_actor", "aether_planner"}:
            raise ValueError("invalid collection_mode")
        if self.bootstrap_collection_mode not in {"teacher", "random", "coverage"}:
            raise ValueError("bootstrap_collection_mode must be teacher/random/coverage")
        if self.precision == "fp16" and self.device not in {"auto", "cuda"}:
            raise ValueError("fp16 native procedural training requires CUDA")
        if not self.curriculum_stages or any(int(s) < 1 or int(s) > 12 for s in self.curriculum_stages):
            raise ValueError("curriculum_stages must be within [1,12]")
        if len(set(self.curriculum_stages)) != len(self.curriculum_stages):
            raise ValueError("curriculum_stages must be unique")
        for name in (
            "tasks_per_batch", "episodes_per_task", "horizon", "eval_tasks", "sequence_length",
            "hidden", "world_epochs", "actor_epochs", "batch_size", "max_collection_batches",
            "minimum_transitions", "evaluation_seed_offset", "collector_planner_budget",
            "collector_planner_horizon", "collector_planner_stride",
        ):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.calibration_epochs < 0 or self.one_step_aux_epochs < 0:
            raise ValueError("calibration/aux epochs must be >= 0")
        for name in ("train_difficulty", "eval_difficulty"):
            if not 0.0 <= float(getattr(self, name)) <= 1.0:
                raise ValueError(f"{name} must lie in [0,1]")

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "NativeProceduralRunnerConfig":
        raw = dict(raw or {})
        if "curriculum_stages" in raw:
            raw["curriculum_stages"] = tuple(int(x) for x in raw["curriculum_stages"])
        return cls(**raw)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["curriculum_stages"] = list(self.curriculum_stages)
        return out

    def resolved_device(self) -> str:
        if self.device != "auto":
            return self.device
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"


class ExplorationPolicyCurriculumAdapter:
    """Deterministically map meta-policy weights to curriculum-stage allocations.

    The mapping is deliberately simple and inspectable. It prevents the environment
    adapter from hiding an arbitrary learned policy inside the evaluator.
    """

    GROUPS = {
        "foundation": (1, 2, 3, 5, 6, 9),
        "information": (4, 7, 8),
        "novelty": (10, 11),
        "transfer": (12,),
    }

    def group_weights(self, policy: DeclarativeExplorationPolicy) -> dict[str, float]:
        # Positive floor keeps every configured concept family reachable.
        return {
            "foundation": 0.05 + max(0.0, float(policy.quality_weight)),
            "information": 0.05 + max(0.0, float(policy.information_weight)) + max(0.0, float(policy.uncertainty_weight)),
            "novelty": 0.05 + max(0.0, float(policy.novelty_weight)) + float(policy.new_world_fraction),
            "transfer": 0.05 + max(0.0, float(policy.transfer_weight)) + float(policy.adversarial_fraction),
        }

    @staticmethod
    def _largest_remainder(weights: list[float], total: int) -> list[int]:
        w = np.asarray(weights, dtype=np.float64)
        if total < 1 or np.any(~np.isfinite(w)) or np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("invalid allocation weights")
        raw = w / w.sum() * int(total)
        base = np.floor(raw).astype(int)
        left = int(total) - int(base.sum())
        order = np.argsort(-(raw - base), kind="stable")
        for i in order[:left]:
            base[int(i)] += 1
        return base.tolist()

    def allocate(self, policy: DeclarativeExplorationPolicy, stages: Iterable[int], slots: int) -> dict[int, int]:
        stages = tuple(int(s) for s in stages)
        if not stages:
            raise ValueError("stages required")
        weights = self.group_weights(policy)
        stage_weights: list[float] = []
        for stage in stages:
            group = next((g for g, members in self.GROUPS.items() if stage in members), None)
            if group is None:
                raise ValueError(f"no curriculum group for stage {stage}")
            # Split a group's mass evenly over enabled member stages.
            enabled = sum(int(x in stages) for x in self.GROUPS[group])
            stage_weights.append(weights[group] / max(1, enabled))
        counts = self._largest_remainder(stage_weights, int(slots))
        return {stage: count for stage, count in zip(stages, counts) if count > 0}

    def receipt(self, policy: DeclarativeExplorationPolicy, stages: Iterable[int], slots: int) -> dict[str, Any]:
        counts = self.allocate(policy, stages, slots)
        body = {
            "format": "awa-v2.22-curriculum-allocation-v1",
            "policy_id": policy.policy_id,
            "policy_sha256": policy.sha256,
            "group_weights": self.group_weights(policy),
            "stage_counts": {str(k): int(v) for k, v in sorted(counts.items())},
            "slots": int(slots),
        }
        body["sha256"] = canonical_sha256(body)
        return body

    def execution_signature(
        self,
        policy: DeclarativeExplorationPolicy,
        stages: Iterable[int],
        slots: int,
    ) -> tuple[tuple[int, int], ...]:
        """Canonical real-execution behavior for candidate deduplication.

        Meta-policy fields can differ while producing the same integer curriculum
        allocation after largest-remainder rounding.  Those candidates are
        operationally equivalent for the native runner and must not consume replay
        evaluation or real GPU qualification as if they represented different
        training policies.
        """
        return tuple(sorted((int(k), int(v)) for k, v in self.allocate(policy, stages, slots).items()))


class NativeProceduralCampaignRunner(CampaignRunner):
    """Qualification-grade local runner for the built-in procedural game lab.

    Real environment interactions generate the training dataset, the normal Aether
    belief/world/actor stack is trained, and evaluation is run on deterministic
    held-out or transfer tasks. The resulting evidence is fully content-addressed.
    """

    SUPPORTED_TASKS = {"procedural", "aether-procedural", "game-lab"}

    def __init__(
        self,
        config: NativeProceduralRunnerConfig | None = None,
        *,
        allocator: ExplorationPolicyCurriculumAdapter | None = None,
        collect_fn: Callable[..., Any] = collect_game_dataset,
        train_fn: Callable[..., Any] = train_game_stack,
        eval_fn: Callable[..., Any] = evaluate_procedural_actor,
    ):
        self.config = config or NativeProceduralRunnerConfig()
        self.allocator = allocator or ExplorationPolicyCurriculumAdapter()
        self.collect_fn = collect_fn
        self.train_fn = train_fn
        self.eval_fn = eval_fn
        self._collector_cache: dict[tuple, Any] = {}
        self.source_sha256 = _source_fingerprint()
        self.dependency_sha256 = _dependency_fingerprint()

    @staticmethod
    def _run_id(cell: CampaignCell) -> str:
        return "|".join(map(str, cell.key))

    def _training_tasks(self, cell: CampaignCell, policy: DeclarativeExplorationPolicy, batch_index: int):
        cfg = self.config
        allocation = self.allocator.allocate(policy, cfg.curriculum_stages, cfg.tasks_per_batch)
        factory = ProceduralTaskFactory(int(cell.seed) + 17_000_003)
        tasks = []
        cursor = batch_index * 1_000_000
        for stage, count in sorted(allocation.items()):
            for j in range(int(count)):
                # Training indices are intentionally far from eval ranges.
                idx = cursor + stage * 10_000 + j
                tasks.append(factory.make(stage, cfg.train_difficulty, idx, "v222-train"))
        return tasks

    def _evaluation_tasks(self, cell: CampaignCell):
        cfg = self.config
        factory = ProceduralTaskFactory(int(cell.seed) + 31_000_019)
        if cell.split == "heldout":
            stages = (2, 4, 6, 9, 11, 12)
            offset = 50_000_000
            novelty = "v222-heldout"
        elif cell.split == "transfer":
            stages = (10, 11, 12)
            offset = 70_000_000
            novelty = "v222-transfer"
        else:
            raise ValueError("native procedural qualification supports heldout/transfer splits only")
        tasks = []
        for i in range(cfg.eval_tasks):
            stage = stages[i % len(stages)]
            tasks.append(factory.make(stage, cfg.eval_difficulty, offset + i, novelty))
        return tasks

    @staticmethod
    def _concat_and_trim(batch_paths: list[Path], target: int, output: Path) -> int:
        arrays: dict[str, list[np.ndarray]] = {}
        for path in batch_paths:
            with np.load(path, allow_pickle=False) as z:
                for key in z.files:
                    # task_ids are stored only in metadata sidecars, never in NPZ.
                    arrays.setdefault(key, []).append(np.asarray(z[key]))
        if not arrays:
            raise RuntimeError("dataset collection produced no arrays")
        merged = {k: np.concatenate(v, axis=0) for k, v in arrays.items()}
        n = min(int(target), int(next(iter(merged.values())).shape[0]))
        if n < 1:
            raise RuntimeError("dataset collection produced no transitions")
        merged = {k: v[:n] for k, v in merged.items()}
        # Mark the final retained row terminal so sequence traversal cannot leak into
        # a transition that was intentionally trimmed away.
        if "dones" in merged:
            merged["dones"] = merged["dones"].copy()
            merged["dones"][-1] = True
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, **merged)
        return n

    def _collection_policy(
        self,
        cell: CampaignCell,
        batch_index: int,
        *,
        prefix_artifact: dict[str, Any] | None,
    ):
        """Resolve the real data-collection policy for this cumulative milestone.

        Aether-controlled collection is only allowed from a prior grounded checkpoint.
        The first milestone therefore uses an explicit non-Aether bootstrap collector;
        the receipt records the actual collector so this transition cannot be hidden.
        """
        cfg = self.config
        requested = cfg.collection_mode
        mode = requested
        if requested in {"aether_actor", "aether_planner"} and prefix_artifact is None:
            mode = cfg.bootstrap_collection_mode
        seed = int(cell.seed) * 1_000_003 + int(batch_index) * 97
        if mode == "teacher":
            return LogicalArenaTeacher(), requested, mode
        if mode == "random":
            return RandomArenaPolicy(seed=seed), requested, mode
        if mode == "coverage":
            return CoverageArenaPolicy(seed=seed), requested, mode
        if prefix_artifact is None:
            raise RuntimeError("Aether collection requires a grounded parent checkpoint")
        root = Path(str(prefix_artifact["root"]))
        paths = self._artifact_paths(root)
        key = (
            str(paths["world"]), str(paths["actor"]), mode, cfg.resolved_device(),
            int(cfg.collector_planner_budget), int(cfg.collector_planner_horizon),
            int(cfg.collector_planner_stride),
        )
        collector = self._collector_cache.get(key)
        if collector is None:
            collector = FrozenAetherArenaPolicy(
                paths["world"], paths["actor"], device=cfg.resolved_device(),
                use_planner=(mode == "aether_planner"),
                planner_budget=cfg.collector_planner_budget,
                planner_horizon=cfg.collector_planner_horizon,
                planner_stride=cfg.collector_planner_stride,
            )
            self._collector_cache[key] = collector
        return collector, requested, mode

    def _collect_training_dataset(self, cell: CampaignCell, policy: DeclarativeExplorationPolicy, work_dir: Path) -> tuple[Path, dict[str, Any]]:
        dataset, _delta, receipt = self._collect_training_dataset_from_prefix(
            cell, policy, work_dir, prefix_artifact=None
        )
        return dataset, receipt

    def _collect_training_dataset_from_prefix(
        self,
        cell: CampaignCell,
        policy: DeclarativeExplorationPolicy,
        work_dir: Path,
        *,
        prefix_artifact: dict[str, Any] | None,
    ) -> tuple[Path, Path, dict[str, Any]]:
        cfg = self.config
        if int(cell.transitions) < int(cfg.minimum_transitions):
            raise ValueError(
                f"planned transition budget {cell.transitions} is below native minimum "
                f"{cfg.minimum_transitions}"
            )
        target = int(cell.transitions)
        prefix_retained = int((prefix_artifact or {}).get("retained_transitions", 0))
        if prefix_retained >= target:
            raise ValueError("cumulative parent must be smaller than requested milestone")
        incremental_target = target - prefix_retained
        batch_paths: list[Path] = []
        collected = 0
        parent_batches = int((prefix_artifact or {}).get("collection_batches_total", 0))
        batch_index = parent_batches
        collection_rows = []
        while collected < incremental_target:
            if batch_index >= cfg.max_collection_batches:
                raise RuntimeError("native procedural collection exceeded max_collection_batches")
            tasks = self._training_tasks(cell, policy, batch_index)
            batch_path = work_dir / "collection" / f"batch-{batch_index:05d}.npz"
            batch_path.parent.mkdir(parents=True, exist_ok=True)
            collector, requested_mode, actual_mode = self._collection_policy(
                cell, batch_index, prefix_artifact=prefix_artifact
            )
            report = self.collect_fn(
                tasks,
                batch_path,
                episodes_per_task=cfg.episodes_per_task,
                horizon=cfg.horizon,
                policy=collector,
            )
            n = int(getattr(report, "transitions", 0))
            if n <= 0:
                raise RuntimeError("native procedural collector made no progress")
            collected += n
            batch_paths.append(batch_path)
            collection_rows.append({
                "batch": batch_index, "transitions": n, "tasks": [t.task_id for t in tasks],
                "requested_collection_mode": requested_mode, "actual_collection_mode": actual_mode,
                "collector": collector_receipt(collector),
            })
            batch_index += 1
        delta_dataset = work_dir / "training_delta.npz"
        incremental_retained = self._concat_and_trim(batch_paths, incremental_target, delta_dataset)
        dataset = work_dir / "training_dataset.npz"
        if prefix_artifact is None:
            shutil.copy2(delta_dataset, dataset)
        else:
            parent_root = Path(str(prefix_artifact["root"]))
            parent_dataset = self._artifact_paths(parent_root)["dataset"]
            retained = self._concat_and_trim([parent_dataset, delta_dataset], target, dataset)
            if retained != target:
                raise RuntimeError("cumulative dataset did not reach requested milestone")
        retained = target
        receipt = {
            "format": "awa-v2.27-native-collection-v1",
            "requested_transitions": int(cell.transitions),
            "minimum_transitions": int(cfg.minimum_transitions),
            "retained_transitions": int(retained),
            "prefix_retained_transitions": int(prefix_retained),
            "incremental_retained_transitions": int(incremental_retained),
            "raw_collected_transitions": int(collected),
            "dataset_sha256": _file_sha256(dataset),
            "delta_dataset_sha256": _file_sha256(delta_dataset),
            "parent_training_identity_sha256": (prefix_artifact or {}).get("training_identity_sha256"),
            "parent_checkpoint_sha256": (prefix_artifact or {}).get("checkpoint_sha256"),
            "allocation": self.allocator.receipt(policy, cfg.curriculum_stages, cfg.tasks_per_batch),
            "collection_mode": cfg.collection_mode,
            "bootstrap_collection_mode": cfg.bootstrap_collection_mode,
            "batches": collection_rows,
            "collection_batches_total": int(parent_batches + len(collection_rows)),
        }
        (work_dir / "collection_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return dataset, delta_dataset, receipt

    def _diagnostic_tasks(self, cell: CampaignCell):
        """Training-only diagnostics from a namespace disjoint from qualification splits."""
        cfg = self.config
        factory = ProceduralTaskFactory(int(cell.seed) + 23_000_033)
        stages = tuple(int(x) for x in cfg.curriculum_stages)
        count = min(max(1, int(cfg.eval_tasks)), max(1, len(stages)))
        return [
            factory.make(stages[i % len(stages)], cfg.train_difficulty, 30_000_000 + i, "v223-train-diagnostic")
            for i in range(count)
        ]

    def _training_lineage(self, cell: CampaignCell, policy: DeclarativeExplorationPolicy) -> tuple[str, dict[str, Any]]:
        cfg = self.config.to_dict()
        # Qualification-only knobs must not change the identity of the trained model.
        for key in ("eval_difficulty", "eval_tasks", "evaluation_seed_offset", "reuse_training_artifacts"):
            cfg.pop(key, None)
        body = {
            "format": "awa-v2.27-native-training-lineage-v1",
            "source_sha256": self.source_sha256,
            "dependency_sha256": self.dependency_sha256,
            "runner_training_config": cfg,
            "policy": policy.to_dict(),
            "task": cell.task,
            "seed": int(cell.seed),
        }
        return canonical_sha256(body), body

    def _training_identity(self, cell: CampaignCell, policy: DeclarativeExplorationPolicy) -> tuple[str, dict[str, Any]]:
        lineage_sha, lineage_body = self._training_lineage(cell, policy)
        body = {
            "format": "awa-v2.27-native-training-identity-v1",
            "lineage_sha256": lineage_sha,
            "lineage": lineage_body,
            "transitions": int(cell.transitions),
        }
        return canonical_sha256(body), body

    @staticmethod
    def _artifact_paths(root: Path) -> dict[str, Path]:
        return {
            "receipt": root / "training_artifact_receipt.json",
            "dataset": root / "training_dataset.npz",
            "delta_dataset": root / "training_delta.npz",
            "collection": root / "collection_receipt.json",
            "world": root / "train" / "game_world.pt",
            "actor": root / "train" / "game_actor.pt",
            "training_report": root / "train" / "game_training_report.json",
        }

    def _load_training_artifact(self, root: Path, expected_identity: str) -> dict[str, Any]:
        paths = self._artifact_paths(root)
        if not paths["receipt"].exists():
            raise RuntimeError(f"incomplete native training artifact: {root}")
        receipt = json.loads(paths["receipt"].read_text(encoding="utf-8"))
        if receipt.get("format") not in {"awa-v2.23-native-training-artifact-v1", "awa-v2.25-native-training-artifact-v1", "awa-v2.26-native-training-artifact-v1", "awa-v2.27-native-training-artifact-v1"}:
            raise ValueError("unsupported native training artifact format")
        if receipt.get("training_identity_sha256") != expected_identity:
            raise ValueError("native training artifact identity mismatch")
        checks = {
            "dataset_sha256": paths["dataset"],
            "world_sha256": paths["world"],
            "actor_sha256": paths["actor"],
            "collection_receipt_sha256": paths["collection"],
            "training_report_sha256": paths["training_report"],
        }
        if receipt.get("delta_dataset_sha256") is not None:
            checks["delta_dataset_sha256"] = paths["delta_dataset"]
        for field, path in checks.items():
            if not path.exists():
                raise RuntimeError(f"native training artifact missing {path.name}")
            if receipt.get(field) != _file_sha256(path):
                raise RuntimeError(f"native training artifact hash mismatch: {field}")
        checkpoint_sha = canonical_sha256({
            "world": receipt["world_sha256"],
            "actor": receipt["actor_sha256"],
        })
        if checkpoint_sha != receipt.get("checkpoint_sha256"):
            raise RuntimeError("native training artifact checkpoint digest mismatch")
        receipt["root"] = str(root)
        return receipt

    @staticmethod
    def _shared_training_cache_root(work_dir: Path) -> Path:
        """Use one campaign-level cache across bootstrap and later iterations.

        v2.24 scoped caches to a single run directory, which forced the active
        baseline to retrain in later meta-iterations.  The orchestrator has stable
        ``bootstrap`` and ``iterations`` anchors, so we can safely locate the common
        campaign root without changing the CampaignRunner interface.
        """
        for parent in work_dir.parents:
            if parent.name in {"bootstrap", "iterations"}:
                return parent.parent / "_training_cache"
        return work_dir.parent / "_training_cache"

    def _find_parent_training_artifact(
        self,
        cache_root: Path,
        cell: CampaignCell,
        policy: DeclarativeExplorationPolicy,
    ) -> dict[str, Any] | None:
        if not self.config.cumulative_milestone_training or not cache_root.exists():
            return None
        lineage_sha, _ = self._training_lineage(cell, policy)
        target = int(cell.transitions)
        best: tuple[int, dict[str, Any]] | None = None
        for receipt_path in cache_root.glob("*/training_artifact_receipt.json"):
            try:
                raw = json.loads(receipt_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if raw.get("lineage_sha256") != lineage_sha:
                continue
            retained = int(raw.get("retained_transitions", 0))
            if retained <= 0 or retained >= target:
                continue
            # A matching lineage is part of the trusted cumulative chain.  If it
            # is corrupt, fail closed rather than silently discarding provenance
            # and retraining a different chain under the same campaign.
            verified = self._load_training_artifact(
                receipt_path.parent, str(raw["training_identity_sha256"])
            )
            if best is None or retained > best[0]:
                best = (retained, verified)
        return None if best is None else best[1]

    def _prepare_training_artifact(
        self,
        cell: CampaignCell,
        policy: DeclarativeExplorationPolicy,
        work_dir: Path,
        device: str,
    ) -> tuple[dict[str, Any], bool]:
        cfg = self.config
        identity, identity_body = self._training_identity(cell, policy)
        if cfg.reuse_training_artifacts:
            cache_root = self._shared_training_cache_root(work_dir)
            artifact_root = cache_root / identity
            if artifact_root.exists():
                return self._load_training_artifact(artifact_root, identity), True
            artifact_root.parent.mkdir(parents=True, exist_ok=True)
            temp_parent = artifact_root.parent
        else:
            artifact_root = work_dir / "training_artifact"
            if artifact_root.exists():
                shutil.rmtree(artifact_root)
            artifact_root.parent.mkdir(parents=True, exist_ok=True)
            temp_parent = artifact_root.parent

        tmp = Path(tempfile.mkdtemp(prefix=f"{identity[:16]}.tmp-", dir=temp_parent))
        try:
            parent_artifact = (
                self._find_parent_training_artifact(artifact_root.parent, cell, policy)
                if cfg.reuse_training_artifacts else None
            )
            t0 = time.perf_counter()
            dataset, delta_dataset, collection = self._collect_training_dataset_from_prefix(
                cell, policy, tmp, prefix_artifact=parent_artifact
            )
            train_out = tmp / "train"
            report = self.train_fn(
                delta_dataset,
                self._diagnostic_tasks(cell),
                train_out,
                sequence_length=min(cfg.sequence_length, max(1, collection["incremental_retained_transitions"])),
                world_epochs=cfg.world_epochs,
                actor_epochs=cfg.actor_epochs,
                calibration_epochs=cfg.calibration_epochs,
                batch_size=cfg.batch_size,
                hidden=cfg.hidden,
                seed=int(cell.seed),
                device=device,
                precision=cfg.precision,
                one_step_aux_epochs=cfg.one_step_aux_epochs,
                resume_world_checkpoint=(
                    self._artifact_paths(Path(parent_artifact["root"]))["world"] if parent_artifact else None
                ),
                resume_actor_checkpoint=(
                    self._artifact_paths(Path(parent_artifact["root"]))["actor"] if parent_artifact else None
                ),
                run_planner_diagnostics=cfg.run_planner_diagnostics,
            )
            incremental_training_elapsed = time.perf_counter() - t0
            parent_cumulative_training = float((parent_artifact or {}).get("training_wall_clock_seconds", 0.0))
            cumulative_training_elapsed = parent_cumulative_training + incremental_training_elapsed
            paths = self._artifact_paths(tmp)
            if not paths["world"].exists() or not paths["actor"].exists():
                raise RuntimeError("native procedural training did not produce checkpoints")
            if not paths["training_report"].exists():
                payload = report.to_dict() if hasattr(report, "to_dict") else dict(report)
                paths["training_report"].parent.mkdir(parents=True, exist_ok=True)
                paths["training_report"].write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            receipt = {
                "format": "awa-v2.27-native-training-artifact-v1",
                "training_identity_sha256": identity,
                "training_identity": identity_body,
                "lineage_sha256": identity_body["lineage_sha256"],
                "retained_transitions": int(collection["retained_transitions"]),
                "incremental_retained_transitions": int(collection["incremental_retained_transitions"]),
                "collection_batches_total": int(collection["collection_batches_total"]),
                "parent_training_identity_sha256": collection.get("parent_training_identity_sha256"),
                "parent_checkpoint_sha256": collection.get("parent_checkpoint_sha256"),
                "incremental_training_wall_clock_seconds": float(incremental_training_elapsed),
                "training_wall_clock_seconds": float(cumulative_training_elapsed),
                "dataset_sha256": _file_sha256(paths["dataset"]),
                "delta_dataset_sha256": _file_sha256(paths["delta_dataset"]),
                "world_sha256": _file_sha256(paths["world"]),
                "actor_sha256": _file_sha256(paths["actor"]),
                "collection_receipt_sha256": _file_sha256(paths["collection"]),
                "training_report_sha256": _file_sha256(paths["training_report"]),
            }
            receipt["checkpoint_sha256"] = canonical_sha256({
                "world": receipt["world_sha256"], "actor": receipt["actor_sha256"]
            })
            paths["receipt"].write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            # Raw collection shards are no longer needed once the exact merged dataset is committed.
            shutil.rmtree(tmp / "collection", ignore_errors=True)
            if artifact_root.exists():
                # A concurrent writer may have won the race. Trust only a fully verified artifact.
                existing = self._load_training_artifact(artifact_root, identity)
                shutil.rmtree(tmp, ignore_errors=True)
                return existing, True
            tmp.rename(artifact_root)
            return self._load_training_artifact(artifact_root, identity), False
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise

    @staticmethod
    def _normalize_eval_report(report: Any) -> dict[str, float]:
        raw = report.to_dict() if hasattr(report, "to_dict") else dict(report)
        aliases = {
            "success_rate": ("success_rate", "actor_success_rate"),
            "mean_return": ("mean_return", "actor_mean_return"),
            "constraint_violations": ("constraint_violations", "actor_constraint_violations"),
            "inference_latency_ms": ("inference_latency_ms", "actor_inference_latency_ms"),
        }
        out: dict[str, float] = {}
        for target, names in aliases.items():
            value = next((raw[name] for name in names if name in raw), None)
            if value is None:
                raise ValueError(f"evaluation report missing {target}")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"evaluation report contains non-finite {target}")
            out[target] = value
        return out

    def run(self, cell: CampaignCell, policy: DeclarativeExplorationPolicy, work_dir: Path) -> CampaignEvidence:
        if cell.task not in self.SUPPORTED_TASKS:
            raise ValueError(f"unsupported native procedural task {cell.task!r}")
        if cell.split not in {"heldout", "transfer"}:
            raise ValueError("native procedural runner requires heldout or transfer split")
        work_dir = Path(work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        cfg = self.config
        device = cfg.resolved_device()
        if cfg.precision == "fp16" and device != "cuda":
            raise ValueError("fp16 native procedural training requires CUDA after device resolution")

        artifact, reused = self._prepare_training_artifact(cell, policy, work_dir, device)
        artifact_root = Path(artifact["root"])
        paths = self._artifact_paths(artifact_root)
        # Bind each evaluation cell to the exact cached training identity while keeping
        # its evaluator/scenario configuration split-specific.
        config_receipt = {
            "format": "awa-v2.23-native-runner-config-v1",
            "runner": cfg.to_dict(),
            "policy": policy.to_dict(),
            "cell": cell.to_dict(),
            "training_identity_sha256": artifact["training_identity_sha256"],
            "checkpoint_sha256": artifact["checkpoint_sha256"],
        }
        config_sha = canonical_sha256(config_receipt)
        (work_dir / "native_runner_config.json").write_text(
            json.dumps(config_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        # Make the training receipt visible from the cell without copying model bytes.
        (work_dir / "training_artifact_receipt.json").write_text(
            json.dumps({k: v for k, v in artifact.items() if k != "root"}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        shutil.copy2(paths["collection"], work_dir / "collection_receipt.json")

        eval_tasks = self._evaluation_tasks(cell)
        eval_t0 = time.perf_counter()
        eval_report = self.eval_fn(
            paths["world"], paths["actor"], eval_tasks,
            device=device, horizon=cfg.horizon, seed_offset=cfg.evaluation_seed_offset,
        )
        evaluation_elapsed = time.perf_counter() - eval_t0
        e = self._normalize_eval_report(eval_report)
        training_elapsed = float(artifact["training_wall_clock_seconds"])
        incremental_training_elapsed = float(artifact.get("incremental_training_wall_clock_seconds", training_elapsed))
        logical_elapsed = training_elapsed + evaluation_elapsed
        physical_elapsed = evaluation_elapsed + (0.0 if reused else incremental_training_elapsed)
        metrics = {
            "success_rate": e["success_rate"],
            "episode_return": e["mean_return"],
            "constraint_violations": e["constraint_violations"],
            "planner_calls_per_episode": 0.0,
            "world_model_calls_per_episode": 0.0,
            "inference_latency_ms": e["inference_latency_ms"],
            # wall_clock_seconds remains a stable logical cell cost so results do not
            # depend on whether heldout or transfer happened to execute first.
            "wall_clock_seconds": float(logical_elapsed),
            "training_wall_clock_seconds": training_elapsed,
            "incremental_training_wall_clock_seconds": incremental_training_elapsed,
            "evaluation_wall_clock_seconds": float(evaluation_elapsed),
            "physical_wall_clock_seconds": float(physical_elapsed),
            "training_cache_reused": 1.0 if reused else 0.0,
            "normalized_compute_cost": float(logical_elapsed / max(1.0, float(cell.transitions))),
            "novelty": float(policy.new_world_fraction),
            "information_gain": float(max(0.0, policy.information_weight)),
            "uncertainty_reduction": float(max(0.0, policy.uncertainty_weight)),
        }
        if not all(math.isfinite(v) for v in metrics.values()):
            raise RuntimeError("native procedural runner produced non-finite metrics")

        scenario_ids = tuple(t.task_id for t in eval_tasks)
        environment_sha = canonical_sha256([t.to_dict() for t in eval_tasks])
        evaluation_receipt = {
            "format": "awa-v2.23-native-evaluation-v1",
            "split": cell.split,
            "scenario_ids": list(scenario_ids),
            "environment_sha256": environment_sha,
            "checkpoint_sha256": artifact["checkpoint_sha256"],
            "dataset_sha256": artifact["dataset_sha256"],
            "metrics": e,
            "evaluation_wall_clock_seconds": float(evaluation_elapsed),
            "training_cache_reused": bool(reused),
        }
        (work_dir / "evaluation_receipt.json").write_text(
            json.dumps(evaluation_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        run_id = self._run_id(cell)
        record = RunRecord(
            system=cell.policy_id,
            seed=int(cell.seed),
            task=cell.task,
            split=cell.split,
            transitions=int(cell.transitions),
            metrics=metrics,
            checkpoint_sha256=str(artifact["checkpoint_sha256"]),
            config_sha256=config_sha,
        )
        provenance = RunProvenance(
            run_id=run_id,
            source_sha256=self.source_sha256,
            dataset_sha256=str(artifact["dataset_sha256"]),
            environment_sha256=environment_sha,
            dependency_lock_sha256=self.dependency_sha256,
            scenario_ids=scenario_ids,
        )
        evidence = CampaignEvidence(record, provenance, EvidenceClass.VALIDATED)
        (work_dir / "evidence.json").write_text(
            json.dumps(evidence.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return evidence


def native_runner_from_config(config: dict[str, Any]) -> NativeProceduralCampaignRunner:
    runner = config.get("runner") or {}
    if str(runner.get("type", "native_procedural")) != "native_procedural":
        raise ValueError("runner.type must be native_procedural")
    return NativeProceduralCampaignRunner(NativeProceduralRunnerConfig.from_dict(runner.get("config") or {}))
