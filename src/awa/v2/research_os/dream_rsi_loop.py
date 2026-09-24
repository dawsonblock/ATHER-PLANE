"""Operational DREAM-RSI loop over Aether training allocation.

This module closes the v2.32 wiring gap: the same declarative menu is used by
real exact-transition execution and grounded replay, completed online rounds are
appended to the replay pool, iterative dreaming runs over the expanded history,
and promotion remains gated by paired grounded evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
from typing import Sequence

from awa.v2.learning_system.dream_execution import execute_training_allocation
from awa.v2.learning_system.meta_allocation import (
    DeclarativeTrainingAllocationPolicy,
    TrainingAllocationDecision,
    TrainingAllocationOption,
)
from awa.v2.research_os.dream_rsi import (
    DreamRSIContract,
    DreamRSIMetaController,
    DreamRSIProposal,
    GroundedReplayWorld,
    GroundedTrainingOutcome,
    OnlineMetaPolicyResult,
    OnlinePromotionReceipt,
    ReplayObjective,
    ground_execution_receipt,
)


def _sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class RealOnlineRoundConfig:
    base_seed: int = 2330
    difficulty: float = 0.50
    horizon: int = 60
    device: str = "cpu"
    precision: str = "fp32"
    sequence_length: int = 3
    hidden: int = 32
    world_epochs: int = 1
    actor_epochs: int = 1
    calibration_epochs: int = 0
    batch_size: int = 32
    one_step_aux_epochs: int = 0
    evaluate_generalization: bool = False
    evaluation_replicates: int = 1


@dataclass(frozen=True)
class OnlineDecisionArtifact:
    outcome_id: str
    execution_key: str
    output_dir: str
    world_checkpoint_path: str
    actor_checkpoint_path: str
    world_checkpoint_sha256: str
    actor_checkpoint_sha256: str
    quality: float
    exact_transitions: int
    wall_seconds: float

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict) -> "OnlineDecisionArtifact":
        return cls(**dict(payload))


@dataclass(frozen=True)
class RealOnlineRoundReceipt:
    world_id: str
    policy_id: str
    iteration: int
    seed: int
    offered_option_keys: tuple[str, ...]
    selected_decision_keys: tuple[str, ...]
    exact_transitions: int
    outcome_ids: tuple[str, ...]
    world_sha256: str
    artifacts: tuple[OnlineDecisionArtifact, ...] = ()
    parent_world_checkpoint_sha256: str | None = None
    parent_actor_checkpoint_sha256: str | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["artifacts"] = [artifact.to_dict() for artifact in self.artifacts]
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> "RealOnlineRoundReceipt":
        row = dict(payload)
        row["offered_option_keys"] = tuple(row.get("offered_option_keys", ()))
        row["selected_decision_keys"] = tuple(row.get("selected_decision_keys", ()))
        row["outcome_ids"] = tuple(row.get("outcome_ids", ()))
        row["artifacts"] = tuple(OnlineDecisionArtifact.from_dict(item) for item in row.get("artifacts", ()))
        return cls(**row)

    @property
    def best_artifact(self) -> OnlineDecisionArtifact | None:
        if not self.artifacts:
            return None
        return max(self.artifacts, key=lambda row: (float(row.quality), -float(row.wall_seconds), row.execution_key))


@dataclass(frozen=True)
class RecursiveRoundReceipt:
    online: RealOnlineRoundReceipt
    proposal: DreamRSIProposal
    replay_world_count: int
    active_policy_id_after_online: str


@dataclass(frozen=True)
class FixedVsDreamComparisonPlan:
    """Controlled baseline plan matching DREAM-RSI's fixed-exploration comparison."""

    fixed_policy_sha256: str
    adaptive_initial_policy_sha256: str
    contract_sha256: str
    seeds: tuple[int, ...]
    rounds: int
    transition_budget_per_round: int
    identical_initial_policy_required: bool = True
    identical_agent_evaluator_interface_required: bool = True
    identical_resource_budget_required: bool = True

    def __post_init__(self) -> None:
        if len(self.seeds) < 2 or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("comparison plan requires at least two unique paired seeds")
        if self.rounds < 1 or self.transition_budget_per_round < 1:
            raise ValueError("rounds and transition budget must be positive")
        if self.fixed_policy_sha256 != self.adaptive_initial_policy_sha256:
            raise ValueError("fixed and adaptive arms must start from the identical policy")

    @property
    def sha256(self) -> str:
        return _sha(asdict(self))


def execute_policy_online_world(
    policy: DeclarativeTrainingAllocationPolicy,
    offered_options: Sequence[TrainingAllocationOption],
    contract: DreamRSIContract,
    out_dir: str | Path,
    *,
    iteration: int,
    seed: int,
    config: RealOnlineRoundConfig | None = None,
    objective: ReplayObjective | None = None,
    world_checkpoint: str | Path | None = None,
    actor_checkpoint: str | Path | None = None,
) -> tuple[GroundedReplayWorld, RealOnlineRoundReceipt]:
    """Execute one real meta-policy round and convert it into a grounded replay world."""
    config = config or RealOnlineRoundConfig()
    options = tuple(offered_options)
    if not options:
        raise ValueError("online round requires a non-empty complete decision menu")
    option_keys = [option.decision.execution_key for option in options]
    if len(option_keys) != len(set(option_keys)):
        raise ValueError("online decision menu contains duplicate execution keys")

    decisions = policy.choose_batch(options)
    if len(decisions) == 1 and decisions[0].stop:
        raise RuntimeError("active meta-policy stopped; no online world can be generated")
    selected = tuple(decision for decision in decisions if not decision.stop)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    outcomes: list[GroundedTrainingOutcome] = []
    artifacts: list[OnlineDecisionArtifact] = []
    exact_transitions = 0
    for index, decision in enumerate(selected):
        option = next(option for option in options if option.decision.execution_key == decision.execution_key)
        decision_dir = out / f"decision-{index:03d}-{decision.execution_key[:12]}"
        execution = execute_training_allocation(
            decision,
            options,
            decision_dir,
            seed=int(seed) + index * 1009,
            iteration=int(iteration),
            base_seed=int(config.base_seed) + index * 17,
            difficulty=float(config.difficulty),
            horizon=int(config.horizon),
            device=str(config.device),
            precision=str(config.precision),
            world_checkpoint=world_checkpoint,
            actor_checkpoint=actor_checkpoint,
            sequence_length=int(config.sequence_length),
            hidden=int(config.hidden),
            world_epochs=int(config.world_epochs),
            actor_epochs=int(config.actor_epochs),
            calibration_epochs=int(config.calibration_epochs),
            batch_size=int(config.batch_size),
            one_step_aux_epochs=int(config.one_step_aux_epochs),
            evaluate_generalization=bool(config.evaluate_generalization),
            evaluation_replicates=int(config.evaluation_replicates),
        )
        outcome_id = f"iter{int(iteration):04d}-seed{int(seed):06d}-decision{index:03d}"
        grounded = ground_execution_receipt(
            execution,
            outcome_id=outcome_id,
            parent_id=None,
            option=option,
            contract=contract,
        )
        outcomes.append(grounded)
        training_dir = Path(execution.output_dir) / "trained"
        artifacts.append(
            OnlineDecisionArtifact(
                outcome_id=outcome_id,
                execution_key=decision.execution_key,
                output_dir=str(execution.output_dir),
                world_checkpoint_path=str(training_dir / "game_world.pt"),
                actor_checkpoint_path=str(training_dir / "game_actor.pt"),
                world_checkpoint_sha256=execution.world_checkpoint_sha256,
                actor_checkpoint_sha256=execution.actor_checkpoint_sha256,
                quality=float(execution.actor_success_rate),
                exact_transitions=int(execution.exact_transitions),
                wall_seconds=float(execution.wall_seconds),
            )
        )
        exact_transitions += int(execution.exact_transitions)

    world_id = f"online-{int(iteration):04d}-{int(seed):06d}-{policy.sha256[:12]}"
    world = GroundedReplayWorld(world_id, outcomes, contract=contract, objective=objective)
    world_payload = world.to_dict() | {
        "policy_id": policy.policy_id,
        "iteration": int(iteration),
        "seed": int(seed),
    }
    world_sha256 = _sha(world_payload)
    (out / "grounded_replay_world.json").write_text(
        json.dumps(world_payload | {"sha256": world_sha256}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt = RealOnlineRoundReceipt(
        world_id=world_id,
        policy_id=policy.policy_id,
        iteration=int(iteration),
        seed=int(seed),
        offered_option_keys=tuple(option_keys),
        selected_decision_keys=tuple(decision.execution_key for decision in selected),
        exact_transitions=int(exact_transitions),
        outcome_ids=tuple(row.outcome_id for row in outcomes),
        world_sha256=world_sha256,
        artifacts=tuple(artifacts),
        parent_world_checkpoint_sha256=(
            None if world_checkpoint is None else _sha_file(Path(world_checkpoint))
        ),
        parent_actor_checkpoint_sha256=(
            None if actor_checkpoint is None else _sha_file(Path(actor_checkpoint))
        ),
    )
    (out / "online_round_receipt.json").write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return world, receipt


def load_online_world(out_dir: str | Path, contract: DreamRSIContract) -> tuple[GroundedReplayWorld, RealOnlineRoundReceipt]:
    out = Path(out_dir)
    world_payload = json.loads((out / "grounded_replay_world.json").read_text(encoding="utf-8"))
    expected_world_sha = str(world_payload.pop("sha256"))
    if _sha(world_payload) != expected_world_sha:
        raise ValueError("grounded replay world digest mismatch")
    world = GroundedReplayWorld.from_dict(world_payload, contract=contract)
    receipt_payload = json.loads((out / "online_round_receipt.json").read_text(encoding="utf-8"))
    receipt = RealOnlineRoundReceipt.from_dict(receipt_payload)
    if receipt.world_sha256 != expected_world_sha:
        raise ValueError("online round receipt does not bind grounded replay world")
    for artifact in receipt.artifacts:
        world_path = Path(artifact.world_checkpoint_path)
        actor_path = Path(artifact.actor_checkpoint_path)
        if not world_path.exists() or not actor_path.exists():
            raise FileNotFoundError("online round artifact checkpoint missing")
        if _sha_file(world_path) != artifact.world_checkpoint_sha256:
            raise ValueError("world checkpoint digest mismatch")
        if _sha_file(actor_path) != artifact.actor_checkpoint_sha256:
            raise ValueError("actor checkpoint digest mismatch")
    return world, receipt


class DreamRSIRecursiveExperiment:
    """Stateful recursive loop: online -> append history -> dream -> qualify -> redeploy."""

    def __init__(self, controller: DreamRSIMetaController):
        self.controller = controller
        self.iteration = len(controller.pool.worlds)
        self.round_history: list[RecursiveRoundReceipt] = []

    def run_online_and_dream(
        self,
        offered_options: Sequence[TrainingAllocationOption],
        out_dir: str | Path,
        *,
        seed: int,
        config: RealOnlineRoundConfig | None = None,
        objective: ReplayObjective | None = None,
        max_replay_rounds: int = 64,
        revision_rounds: int = 4,
        world_checkpoint: str | Path | None = None,
        actor_checkpoint: str | Path | None = None,
    ) -> RecursiveRoundReceipt:
        policy = self.controller.active_policy
        world, online = execute_policy_online_world(
            policy,
            offered_options,
            self.controller.contract,
            out_dir,
            iteration=self.iteration,
            seed=int(seed),
            config=config,
            objective=objective,
            world_checkpoint=world_checkpoint,
            actor_checkpoint=actor_checkpoint,
        )
        self.controller.ingest_online_world(world)
        proposal = self.controller.dream(
            max_rounds=int(max_replay_rounds),
            revision_rounds=int(revision_rounds),
        )
        receipt = RecursiveRoundReceipt(
            online=online,
            proposal=proposal,
            replay_world_count=len(self.controller.pool.worlds),
            active_policy_id_after_online=self.controller.active_policy.policy_id,
        )
        self.round_history.append(receipt)
        self.iteration += 1
        return receipt

    def qualify_and_redeploy(
        self,
        results: Sequence[OnlineMetaPolicyResult],
    ) -> OnlinePromotionReceipt:
        """Promote only from paired grounded online evidence; next round uses winner."""
        return self.controller.validate_and_promote(results)
