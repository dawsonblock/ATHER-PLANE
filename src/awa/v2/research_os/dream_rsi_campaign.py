"""Paired empirical campaign for fixed exploration vs grounded DREAM-RSI.

The campaign is intentionally outside the action-time agent.  It executes the same
training-allocation interface for both arms, keeps hard transition/worker ceilings,
continues each arm from its best grounded checkpoint, and treats replay-selected
policies as trial deployments until the paired multi-seed report is complete.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import hashlib
import json
import math
import statistics
from typing import Iterable, Sequence

import numpy as np
import yaml

from awa.v2.learning_system.meta_allocation import (
    DeclarativeTrainingAllocationPolicy,
    DeclarativeTrainingPolicyMutator,
    TrainingAllocationDecision,
    TrainingAllocationOption,
)
from awa.v2.research_os.dream_rsi import (
    DreamRSIContract,
    DreamRSIOptimizer,
    ReplayObjective,
    SupportAwareReplayPool,
    SupportPolicy,
    SupportStatus,
)
from awa.v2.research_os.dream_rsi_loop import (
    RealOnlineRoundConfig,
    RealOnlineRoundReceipt,
    execute_policy_online_world,
    load_online_world,
)


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_device_name(requested: str) -> str:
    value = str(requested)
    if value != "auto":
        return value
    import torch
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


@dataclass(frozen=True)
class MenuOptionSpec:
    factors: tuple[str, ...]
    capability_deficit: float
    uncertainty: float = 0.0
    transfer_gap: float = 0.0
    novelty: float = 0.0
    expected_cost: float = 0.0
    collector: str = "coverage"
    planner_budget: int = 0
    parallel_worlds: int = 1

    @classmethod
    def from_dict(cls, payload: dict) -> "MenuOptionSpec":
        row = dict(payload)
        row["factors"] = tuple(sorted(set(row.pop("factor_signature", row.get("factors", ())))))
        return cls(**row)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["factors"] = list(self.factors)
        return payload


@dataclass(frozen=True)
class DreamRSICampaignConfig:
    seeds: tuple[int, ...]
    rounds: int
    round_transition_budget: int
    option_transition_budget: int
    worker_budget: int
    max_parallel: int
    revision_rounds: int
    replay_max_rounds: int
    mutation_step: float
    menu: tuple[MenuOptionSpec, ...]
    initial_policy: DeclarativeTrainingAllocationPolicy
    support_policy: SupportPolicy
    objective: ReplayObjective
    runtime: RealOnlineRoundConfig
    minimum_final_mean_gain: float = 0.0
    minimum_curve_mean_gain: float = 0.0
    max_seed_regression: float = 0.05
    bootstrap_samples: int = 10000
    final_eval_replicates: int = 2
    final_eval_base_seed: int = 934000
    final_heldout_difficulty: float = 0.70
    final_transfer_difficulty: float = 0.82

    def __post_init__(self) -> None:
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError("campaign requires at least two unique paired seeds")
        if self.rounds < 2:
            raise ValueError("campaign requires at least two recursive rounds")
        if self.round_transition_budget < 1 or self.option_transition_budget < 1:
            raise ValueError("transition budgets must be positive")
        if self.option_transition_budget > self.round_transition_budget:
            raise ValueError("option transition budget cannot exceed round budget")
        if self.worker_budget < 1 or self.max_parallel < 1:
            raise ValueError("worker/max_parallel budgets must be positive")
        if not self.menu:
            raise ValueError("campaign menu cannot be empty")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be >= 100")
        if self.final_eval_replicates < 1:
            raise ValueError("final_eval_replicates must be >= 1")
        if not 0.0 <= self.final_heldout_difficulty <= 1.0 or not 0.0 <= self.final_transfer_difficulty <= 1.0:
            raise ValueError("final evaluation difficulties must lie in [0,1]")
        if self.initial_policy.worker_budget != self.worker_budget:
            raise ValueError("initial policy worker_budget must equal campaign worker_budget")
        if self.initial_policy.max_parallel != self.max_parallel:
            raise ValueError("initial policy max_parallel must equal campaign max_parallel")
        if self.initial_policy.transition_budget_cap != self.round_transition_budget:
            raise ValueError("initial policy transition_budget_cap must equal round_transition_budget")

    @property
    def sha256(self) -> str:
        return _sha(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "seeds": list(self.seeds),
            "rounds": self.rounds,
            "round_transition_budget": self.round_transition_budget,
            "option_transition_budget": self.option_transition_budget,
            "worker_budget": self.worker_budget,
            "max_parallel": self.max_parallel,
            "revision_rounds": self.revision_rounds,
            "replay_max_rounds": self.replay_max_rounds,
            "mutation_step": self.mutation_step,
            "menu": [row.to_dict() for row in self.menu],
            "initial_policy": self.initial_policy.to_dict(),
            "support_policy": asdict(self.support_policy),
            "objective": self.objective.to_dict(),
            "runtime": asdict(self.runtime),
            "minimum_final_mean_gain": self.minimum_final_mean_gain,
            "minimum_curve_mean_gain": self.minimum_curve_mean_gain,
            "max_seed_regression": self.max_seed_regression,
            "bootstrap_samples": self.bootstrap_samples,
            "final_eval_replicates": self.final_eval_replicates,
            "final_eval_base_seed": self.final_eval_base_seed,
            "final_heldout_difficulty": self.final_heldout_difficulty,
            "final_transfer_difficulty": self.final_transfer_difficulty,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "DreamRSICampaignConfig":
        row = dict(payload)
        row.pop("release", None)
        row.pop("purpose", None)
        row["seeds"] = tuple(int(x) for x in row["seeds"])
        row["menu"] = tuple(MenuOptionSpec.from_dict(x) for x in row["menu"])
        initial = dict(row["initial_policy"])
        initial.setdefault("transition_budget_cap", int(row["round_transition_budget"]))
        row["initial_policy"] = DeclarativeTrainingAllocationPolicy.from_dict(initial)
        row["support_policy"] = SupportPolicy(**row.get("support_policy", {}))
        row["objective"] = ReplayObjective.from_dict(row.get("objective", {}))
        row["runtime"] = RealOnlineRoundConfig(**row.get("runtime", {}))
        return cls(**row)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "DreamRSICampaignConfig":
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("campaign YAML must decode to a mapping")
        return cls.from_dict(payload)


@dataclass(frozen=True)
class ArmRoundRecord:
    seed: int
    round_index: int
    arm: str
    policy_id: str
    policy_sha256: str
    menu_sha256: str
    selected_execution_keys: tuple[str, ...]
    score: float
    best_quality: float
    exact_transitions: int
    wall_seconds: float
    world_sha256: str
    world_checkpoint_sha256: str | None
    actor_checkpoint_sha256: str | None
    proposal_status: str | None = None
    proposal_policy_id: str | None = None
    proposal_replay_gain: float | None = None
    proposal_support: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SeedComparisonSummary:
    seed: int
    fixed_final_score: float
    dream_final_score: float
    final_gain: float
    fixed_heldout_success: float
    dream_heldout_success: float
    fixed_transfer_success: float
    dream_transfer_success: float
    fixed_curve_mean: float
    dream_curve_mean: float
    curve_gain: float
    fixed_transitions: int
    dream_transitions: int
    fixed_wall_seconds: float
    dream_wall_seconds: float
    dream_policy_changes: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FixedVsDreamCampaignReport:
    status: str
    config_sha256: str
    contract_sha256: str
    seeds: tuple[int, ...]
    completed_rounds: int
    expected_rounds: int
    mean_final_gain: float
    final_gain_ci95: tuple[float, float]
    worst_final_gain: float
    mean_curve_gain: float
    curve_gain_ci95: tuple[float, float]
    fixed_total_transitions: int
    dream_total_transitions: int
    fixed_total_wall_seconds: float
    dream_total_wall_seconds: float
    failures: tuple[str, ...]
    seed_summaries: tuple[SeedComparisonSummary, ...]

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["final_gain_ci95"] = list(self.final_gain_ci95)
        payload["curve_gain_ci95"] = list(self.curve_gain_ci95)
        payload["seed_summaries"] = [row.to_dict() for row in self.seed_summaries]
        return payload


def build_menu(config: DreamRSICampaignConfig, *, round_index: int) -> tuple[TrainingAllocationOption, ...]:
    rows: list[TrainingAllocationOption] = []
    for spec in config.menu:
        decision = TrainingAllocationDecision(
            factor_signature=tuple(sorted(spec.factors)),
            collector=spec.collector,
            transition_budget=int(config.option_transition_budget),
            planner_budget=int(spec.planner_budget),
            parallel_worlds=int(spec.parallel_worlds),
        )
        rows.append(
            TrainingAllocationOption(
                decision=decision,
                capability_deficit=float(spec.capability_deficit),
                uncertainty=float(spec.uncertainty),
                transfer_gap=float(spec.transfer_gap),
                novelty=float(spec.novelty),
                expected_cost=float(spec.expected_cost),
                historical_support=0,
                age_iterations=int(round_index),
            )
        )
    return tuple(rows)


def _menu_sha(options: Sequence[TrainingAllocationOption]) -> str:
    return _sha([row.to_dict() for row in options])


def _paired_bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return (float("nan"), float("nan"))
    if arr.size == 1:
        return (float(arr[0]), float(arr[0]))
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, arr.size, size=(int(samples), arr.size))
    means = arr[indices].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def _policy_for_campaign(config: DreamRSICampaignConfig) -> DeclarativeTrainingAllocationPolicy:
    # Rebuild from the frozen payload so both arms truly start byte-identically.
    return DeclarativeTrainingAllocationPolicy.from_dict(config.initial_policy.to_dict())


def _lineage_from_receipt(receipt: RealOnlineRoundReceipt) -> tuple[str | None, str | None, str | None, str | None]:
    best = receipt.best_artifact
    if best is None:
        return None, None, None, None
    return (
        best.world_checkpoint_path,
        best.actor_checkpoint_path,
        best.world_checkpoint_sha256,
        best.actor_checkpoint_sha256,
    )


def _final_generalization_score(
    config: DreamRSICampaignConfig,
    *,
    seed: int,
    world_checkpoint: str,
    actor_checkpoint: str,
) -> tuple[float, float, float]:
    from awa.v2.curriculum.environment_factory import FactorizedEnvironmentFactory
    from awa.v2.game.procedural_runtime import evaluate_procedural_actor

    factory = FactorizedEnvironmentFactory(base_seed=int(config.final_eval_base_seed) + int(seed) * 1009)
    benchmark = factory.build_canonical_benchmark(
        train_replicates=1,
        eval_replicates=int(config.final_eval_replicates),
        train_difficulty=float(config.runtime.difficulty),
        heldout_difficulty=float(config.final_heldout_difficulty),
        transfer_difficulty=float(config.final_transfer_difficulty),
    )
    heldout = evaluate_procedural_actor(
        world_checkpoint, actor_checkpoint, benchmark.heldout,
        device=_resolve_device_name(config.runtime.device), horizon=int(config.runtime.horizon), seed_offset=181,
    )
    transfer = evaluate_procedural_actor(
        world_checkpoint, actor_checkpoint, benchmark.transfer,
        device=_resolve_device_name(config.runtime.device), horizon=int(config.runtime.horizon), seed_offset=211,
    )
    combined = 0.5 * (float(heldout.success_rate) + float(transfer.success_rate))
    return combined, float(heldout.success_rate), float(transfer.success_rate)


class FixedVsDreamCampaignRunner:
    """Run a matched recursive fixed-policy vs DREAM-RSI campaign.

    Every seed is an independent replicate.  Both arms receive the same offered menu,
    environment seed, transition ceiling, worker ceiling, underlying Aether agent, and
    evaluator contract.  The DREAM arm may trial-deploy a replay-selected policy in the
    *next* real round; this trial deployment is not itself labeled empirical promotion.
    """

    def __init__(
        self,
        config: DreamRSICampaignConfig,
        contract: DreamRSIContract,
        out_dir: str | Path,
    ) -> None:
        self.config = config
        self.contract = contract
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.plan_path = self.out_dir / "campaign_plan.json"
        self.records_path = self.out_dir / "records.jsonl"
        self.report_path = self.out_dir / "fixed_vs_dream_report.json"

    def freeze_plan(self) -> dict:
        payload = {
            "format": "awa-v2.34-fixed-vs-dream-plan-v1",
            "config_sha256": self.config.sha256,
            "contract_sha256": self.contract.sha256,
            "execution_device": _resolve_device_name(self.config.runtime.device),
            "config": self.config.to_dict(),
            "invariants": {
                "same_initial_policy": True,
                "same_offered_menu_per_paired_round": True,
                "same_environment_seed_per_paired_round": True,
                "same_transition_ceiling": True,
                "same_worker_ceiling": True,
                "same_agent_evaluator_interface_contract": True,
                "replay_selected_policy_is_trial_deployment_not_empirical_promotion": True,
            },
        }
        payload["sha256"] = _sha(payload)
        if self.plan_path.exists():
            existing = json.loads(self.plan_path.read_text(encoding="utf-8"))
            if existing.get("sha256") != payload["sha256"]:
                raise RuntimeError("campaign plan changed after freeze")
        else:
            _atomic_json(self.plan_path, payload)
        return payload

    def _round_dir(self, seed: int, round_index: int, arm: str) -> Path:
        return self.out_dir / f"seed-{seed:06d}" / f"round-{round_index:03d}" / arm

    def _load_or_execute(
        self,
        *,
        policy: DeclarativeTrainingAllocationPolicy,
        menu: Sequence[TrainingAllocationOption],
        seed: int,
        round_index: int,
        arm: str,
        world_checkpoint: str | None,
        actor_checkpoint: str | None,
    ):
        out = self._round_dir(seed, round_index, arm)
        if (out / "online_round_receipt.json").exists() and (out / "grounded_replay_world.json").exists():
            world, receipt = load_online_world(out, self.contract)
            if receipt.policy_id != policy.policy_id:
                raise RuntimeError("resumed round policy mismatch")
            expected_menu = tuple(option.decision.execution_key for option in menu)
            if receipt.offered_option_keys != expected_menu:
                raise RuntimeError("resumed round decision menu mismatch")
            expected_world_parent = None if world_checkpoint is None else _sha_file(world_checkpoint)
            expected_actor_parent = None if actor_checkpoint is None else _sha_file(actor_checkpoint)
            if receipt.parent_world_checkpoint_sha256 != expected_world_parent:
                raise RuntimeError("resumed round world-checkpoint lineage mismatch")
            if receipt.parent_actor_checkpoint_sha256 != expected_actor_parent:
                raise RuntimeError("resumed round actor-checkpoint lineage mismatch")
            return world, receipt
        out.mkdir(parents=True, exist_ok=True)
        return execute_policy_online_world(
            policy,
            menu,
            self.contract,
            out,
            iteration=int(round_index),
            seed=int(seed) * 1000 + int(round_index),
            config=replace(self.config.runtime, device=_resolve_device_name(self.config.runtime.device)),
            objective=self.config.objective,
            world_checkpoint=world_checkpoint,
            actor_checkpoint=actor_checkpoint,
        )

    def _record(
        self,
        *,
        world,
        receipt: RealOnlineRoundReceipt,
        seed: int,
        round_index: int,
        arm: str,
        policy: DeclarativeTrainingAllocationPolicy,
        menu_sha256: str,
        proposal=None,
    ) -> ArmRoundRecord:
        trace = world.realized_trace(policy.policy_id)
        best = receipt.best_artifact
        best_quality = float(best.quality) if best is not None else 0.0
        return ArmRoundRecord(
            seed=int(seed),
            round_index=int(round_index),
            arm=str(arm),
            policy_id=policy.policy_id,
            policy_sha256=policy.sha256,
            menu_sha256=str(menu_sha256),
            selected_execution_keys=tuple(receipt.selected_decision_keys),
            score=float(trace.score),
            best_quality=best_quality,
            exact_transitions=int(receipt.exact_transitions),
            wall_seconds=float(trace.latency_seconds),
            world_sha256=receipt.world_sha256,
            world_checkpoint_sha256=None if best is None else best.world_checkpoint_sha256,
            actor_checkpoint_sha256=None if best is None else best.actor_checkpoint_sha256,
            proposal_status=None if proposal is None else proposal.status,
            proposal_policy_id=None if proposal is None else proposal.candidate_policy.policy_id,
            proposal_replay_gain=None if proposal is None else float(proposal.replay_gain),
            proposal_support=None if proposal is None else proposal.candidate_support.value,
        )

    def run(self) -> FixedVsDreamCampaignReport:
        self.freeze_plan()
        all_records: list[ArmRoundRecord] = []
        summaries: list[SeedComparisonSummary] = []

        for seed in self.config.seeds:
            fixed_policy = _policy_for_campaign(self.config)
            dream_policy = _policy_for_campaign(self.config)
            if fixed_policy.sha256 != dream_policy.sha256:
                raise RuntimeError("paired arms do not start from identical policies")

            fixed_world_ckpt = fixed_actor_ckpt = None
            dream_world_ckpt = dream_actor_ckpt = None
            fixed_rows: list[ArmRoundRecord] = []
            dream_rows: list[ArmRoundRecord] = []
            dream_worlds = []
            policy_changes = 0

            for round_index in range(self.config.rounds):
                menu = build_menu(self.config, round_index=round_index)
                menu_sha = _menu_sha(menu)

                executed_dream_policy = dream_policy

                fixed_world, fixed_receipt = self._load_or_execute(
                    policy=fixed_policy,
                    menu=menu,
                    seed=seed,
                    round_index=round_index,
                    arm="fixed",
                    world_checkpoint=fixed_world_ckpt,
                    actor_checkpoint=fixed_actor_ckpt,
                )
                dream_world, dream_receipt = self._load_or_execute(
                    policy=executed_dream_policy,
                    menu=menu,
                    seed=seed,
                    round_index=round_index,
                    arm="dream",
                    world_checkpoint=dream_world_ckpt,
                    actor_checkpoint=dream_actor_ckpt,
                )
                if fixed_receipt.offered_option_keys != dream_receipt.offered_option_keys:
                    raise RuntimeError("paired arms saw different offered decision menus")
                if fixed_receipt.exact_transitions > self.config.round_transition_budget:
                    raise RuntimeError("fixed arm exceeded frozen transition ceiling")
                if dream_receipt.exact_transitions > self.config.round_transition_budget:
                    raise RuntimeError("DREAM arm exceeded frozen transition ceiling")

                fixed_world_ckpt, fixed_actor_ckpt, _, _ = _lineage_from_receipt(fixed_receipt)
                dream_world_ckpt, dream_actor_ckpt, _, _ = _lineage_from_receipt(dream_receipt)
                dream_worlds.append(dream_world)

                proposal = None
                if len(dream_worlds) >= self.config.support_policy.min_worlds:
                    pool = SupportAwareReplayPool(dream_worlds, self.config.support_policy)
                    optimizer = DreamRSIOptimizer(
                        pool,
                        DeclarativeTrainingPolicyMutator(step=self.config.mutation_step),
                    )
                    proposal = optimizer.propose(
                        executed_dream_policy,
                        max_rounds=self.config.replay_max_rounds,
                        revision_rounds=self.config.revision_rounds,
                    )
                    if (
                        proposal.candidate_policy.policy_id != executed_dream_policy.policy_id
                        and proposal.replay_gain > 0
                        and proposal.candidate_support is not SupportStatus.UNSUPPORTED
                    ):
                        dream_policy = replace(
                            proposal.candidate_policy,
                            worker_budget=self.config.worker_budget,
                            max_parallel=self.config.max_parallel,
                            transition_budget_cap=self.config.round_transition_budget,
                        )
                        policy_changes += 1

                fixed_row = self._record(
                    world=fixed_world,
                    receipt=fixed_receipt,
                    seed=seed,
                    round_index=round_index,
                    arm="fixed",
                    policy=fixed_policy,
                    menu_sha256=menu_sha,
                )
                dream_row = self._record(
                    world=dream_world,
                    receipt=dream_receipt,
                    seed=seed,
                    round_index=round_index,
                    arm="dream",
                    policy=executed_dream_policy,
                    menu_sha256=menu_sha,
                    proposal=proposal,
                )
                _atomic_json(
                    self._round_dir(seed, round_index, "dream") / "executed_policy.json",
                    executed_dream_policy.to_dict(),
                )

                fixed_rows.append(fixed_row)
                dream_rows.append(dream_row)
                all_records.extend((fixed_row, dream_row))

                state_payload = {
                    "format": "awa-v2.34-dream-seed-state-v1",
                    "seed": seed,
                    "round_completed": round_index,
                    "next_dream_policy": dream_policy.to_dict(),
                    "fixed_lineage": {
                        "world_checkpoint": fixed_world_ckpt,
                        "actor_checkpoint": fixed_actor_ckpt,
                    },
                    "dream_lineage": {
                        "world_checkpoint": dream_world_ckpt,
                        "actor_checkpoint": dream_actor_ckpt,
                    },
                    "policy_changes": policy_changes,
                }
                _atomic_json(self.out_dir / f"seed-{seed:06d}" / "state.json", state_payload)

            fixed_scores = [row.score for row in fixed_rows]
            dream_scores = [row.score for row in dream_rows]
            if not fixed_world_ckpt or not fixed_actor_ckpt or not dream_world_ckpt or not dream_actor_ckpt:
                raise RuntimeError("campaign ended without final grounded checkpoints")
            fixed_final, fixed_heldout, fixed_transfer = _final_generalization_score(
                self.config, seed=seed, world_checkpoint=fixed_world_ckpt, actor_checkpoint=fixed_actor_ckpt
            )
            dream_final, dream_heldout, dream_transfer = _final_generalization_score(
                self.config, seed=seed, world_checkpoint=dream_world_ckpt, actor_checkpoint=dream_actor_ckpt
            )
            summaries.append(
                SeedComparisonSummary(
                    seed=int(seed),
                    fixed_final_score=float(fixed_final),
                    dream_final_score=float(dream_final),
                    final_gain=float(dream_final - fixed_final),
                    fixed_heldout_success=float(fixed_heldout),
                    dream_heldout_success=float(dream_heldout),
                    fixed_transfer_success=float(fixed_transfer),
                    dream_transfer_success=float(dream_transfer),
                    fixed_curve_mean=float(statistics.fmean(fixed_scores)),
                    dream_curve_mean=float(statistics.fmean(dream_scores)),
                    curve_gain=float(statistics.fmean(dream_scores) - statistics.fmean(fixed_scores)),
                    fixed_transitions=sum(row.exact_transitions for row in fixed_rows),
                    dream_transitions=sum(row.exact_transitions for row in dream_rows),
                    fixed_wall_seconds=sum(row.wall_seconds for row in fixed_rows),
                    dream_wall_seconds=sum(row.wall_seconds for row in dream_rows),
                    dream_policy_changes=int(policy_changes),
                )
            )

        # Rewrite records deterministically on every complete run rather than append duplicates on resume.
        self.records_path.write_text(
            "".join(json.dumps(row.to_dict(), sort_keys=True) + "\n" for row in all_records),
            encoding="utf-8",
        )
        report = build_campaign_report(self.config, self.contract, summaries)
        _atomic_json(self.report_path, report.to_dict())
        return report


def build_campaign_report(
    config: DreamRSICampaignConfig,
    contract: DreamRSIContract,
    summaries: Iterable[SeedComparisonSummary],
) -> FixedVsDreamCampaignReport:
    rows = tuple(sorted(summaries, key=lambda row: row.seed))
    failures: list[str] = []
    if len(rows) != len(config.seeds):
        failures.append(f"completed paired seeds {len(rows)} != expected {len(config.seeds)}")
    if tuple(row.seed for row in rows) != tuple(sorted(config.seeds)):
        failures.append("paired seed set mismatch")
    final_gains = [row.final_gain for row in rows]
    curve_gains = [row.curve_gain for row in rows]
    mean_final = float(statistics.fmean(final_gains)) if final_gains else float("nan")
    mean_curve = float(statistics.fmean(curve_gains)) if curve_gains else float("nan")
    worst_final = float(min(final_gains)) if final_gains else float("nan")
    final_ci = _paired_bootstrap_ci(final_gains, samples=config.bootstrap_samples, seed=23401)
    curve_ci = _paired_bootstrap_ci(curve_gains, samples=config.bootstrap_samples, seed=23402)

    if rows:
        if mean_final <= config.minimum_final_mean_gain and mean_curve <= config.minimum_curve_mean_gain:
            failures.append("DREAM arm did not clear final-score or learning-curve gain threshold")
        if worst_final < -abs(config.max_seed_regression):
            failures.append("DREAM arm exceeded allowed per-seed final regression")
    status = "PASS" if not failures else "FAIL"
    if len(rows) != len(config.seeds):
        status = "INSUFFICIENT_EVIDENCE"

    return FixedVsDreamCampaignReport(
        status=status,
        config_sha256=config.sha256,
        contract_sha256=contract.sha256,
        seeds=tuple(row.seed for row in rows),
        completed_rounds=len(rows) * config.rounds,
        expected_rounds=len(config.seeds) * config.rounds,
        mean_final_gain=mean_final,
        final_gain_ci95=final_ci,
        worst_final_gain=worst_final,
        mean_curve_gain=mean_curve,
        curve_gain_ci95=curve_ci,
        fixed_total_transitions=sum(row.fixed_transitions for row in rows),
        dream_total_transitions=sum(row.dream_transitions for row in rows),
        fixed_total_wall_seconds=sum(row.fixed_wall_seconds for row in rows),
        dream_total_wall_seconds=sum(row.dream_wall_seconds for row in rows),
        failures=tuple(failures),
        seed_summaries=rows,
    )
