"""Real online execution for DREAM-RSI training-allocation decisions.

The learning layer executes exactly the same declarative decisions consumed by
historical replay.  It intentionally knows nothing about promotion, provenance
classes, or research-OS decisions.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import time
from typing import Sequence

import numpy as np

from awa.v2.game.goal_program import GOAL_DIM
from awa.v2.game.procedural_arena import ACTION_DIM, OBS_DIM, ProceduralArenaEnv
from awa.v2.learning_system.meta_allocation import (
    TrainingAllocationDecision,
    TrainingAllocationOption,
    build_collector_from_decision,
    materialize_training_allocation,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ExactAllocationExecutionReceipt:
    """Groundable result of one real training-allocation execution.

    The receipt is deliberately evidence-neutral.  The research OS is responsible
    for deciding whether the receipt is OBSERVED/VALIDATED and binding it to a
    fixed DREAM-RSI agent/evaluator/interface contract.
    """

    decision: TrainingAllocationDecision
    offered_options: tuple[TrainingAllocationOption, ...]
    seed: int
    iteration: int
    exact_transitions: int
    collection_success_rate: float
    collection_mean_return: float
    actor_success_rate: float
    actor_mean_return: float
    world_loss: float
    wall_seconds: float
    dataset_sha256: str
    world_checkpoint_sha256: str
    actor_checkpoint_sha256: str
    receipt_sha256: str
    output_dir: str
    generalization_evaluated: bool = False
    heldout_success_rate: float = 0.0
    heldout_mean_return: float = 0.0
    transfer_success_rate: float = 0.0
    transfer_mean_return: float = 0.0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["decision"] = self.decision.to_dict()
        payload["offered_options"] = [option.to_dict() for option in self.offered_options]
        return payload


def collect_exact_transition_dataset(
    decision: TrainingAllocationDecision,
    output: str | Path,
    *,
    base_seed: int,
    difficulty: float,
    seed: int,
    horizon: int = 120,
    split: str = "train",
    world_checkpoint: str | Path | None = None,
    actor_checkpoint: str | Path | None = None,
    device: str = "cpu",
) -> tuple[dict, tuple[object, ...]]:
    """Collect exactly ``decision.transition_budget`` grounded transitions.

    Unlike the older episode-count collector this function stops at the exact
    transition budget.  Short final episodes are explicitly terminated by the
    environment horizon so sequence boundaries remain valid.
    """
    if decision.stop:
        raise ValueError("cannot execute a stop decision")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    materialized = materialize_training_allocation(
        decision,
        base_seed=int(base_seed),
        difficulty=float(difficulty),
        split=str(split),
        validate_tasks=True,
    )
    tasks = tuple(materialized.tasks)
    policy = build_collector_from_decision(
        decision,
        seed=int(seed),
        world_checkpoint=None if world_checkpoint is None else str(world_checkpoint),
        actor_checkpoint=None if actor_checkpoint is None else str(actor_checkpoint),
        device=str(device),
    )

    observations: list[np.ndarray] = []
    next_observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    dones: list[bool] = []
    constraints: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    next_goals: list[np.ndarray] = []
    episode_ids: list[int] = []
    stages: list[int] = []
    task_ids: list[str] = []

    returns: list[float] = []
    successes = 0
    episode_counter = 0
    task_cursor = 0
    budget = int(decision.transition_budget)

    while len(observations) < budget:
        task = tasks[task_cursor % len(tasks)]
        task_cursor += 1
        remaining = budget - len(observations)
        episode_horizon = min(int(horizon), int(remaining))
        env = ProceduralArenaEnv(task, horizon=episode_horizon)
        obs = env.reset(seed=int(task.seed) + episode_counter * 7919 + int(seed))
        if hasattr(policy, "reset_episode"):
            policy.reset_episode(env)
        done = False
        total_return = 0.0
        info: dict = {}
        while not done and len(observations) < budget:
            goal = env.goal_vector().copy()
            action = np.asarray(policy.act(env, obs), dtype=np.float32).reshape(ACTION_DIM)
            nxt, reward, done, info = env.step(action)
            next_goal = env.goal_vector().copy()
            observations.append(obs.copy())
            next_observations.append(nxt.copy())
            actions.append(action.copy())
            rewards.append(float(reward))
            dones.append(bool(done))
            constraints.append(env.constraint_labels())
            goals.append(goal)
            next_goals.append(next_goal)
            episode_ids.append(episode_counter)
            stages.append(int(task.stage))
            task_ids.append(task.task_id)
            total_return += float(reward)
            obs = nxt
        returns.append(total_return)
        successes += int(bool(info.get("success", False)))
        episode_counter += 1

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        observations=np.asarray(observations, dtype=np.float32),
        goals=np.asarray(goals, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        next_goals=np.asarray(next_goals, dtype=np.float32),
        dones=np.asarray(dones, dtype=np.bool_),
        constraints=np.asarray(constraints, dtype=np.float32),
        episode_ids=np.asarray(episode_ids, dtype=np.int64),
        stages=np.asarray(stages, dtype=np.int16),
    )
    report = {
        "format": "awa-v2.34-exact-allocation-dataset-v1",
        "exact_transition_budget": budget,
        "transitions": len(observations),
        "episodes": episode_counter,
        "successes": successes,
        "success_rate": float(successes / max(1, episode_counter)),
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "observation_dim": OBS_DIM,
        "goal_dim": GOAL_DIM,
        "action_dim": ACTION_DIM,
        "task_ids": task_ids,
        "tasks": [task.to_dict() for task in tasks],
        "decision": decision.to_dict(),
    }
    if report["transitions"] != budget:
        raise RuntimeError("exact transition collector violated requested budget")
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report, tasks


def execute_training_allocation(
    decision: TrainingAllocationDecision,
    offered_options: Sequence[TrainingAllocationOption],
    out_dir: str | Path,
    *,
    seed: int,
    iteration: int,
    base_seed: int = 2330,
    difficulty: float = 0.5,
    horizon: int = 120,
    device: str = "cpu",
    precision: str = "fp32",
    world_checkpoint: str | Path | None = None,
    actor_checkpoint: str | Path | None = None,
    sequence_length: int = 3,
    hidden: int = 32,
    world_epochs: int = 1,
    actor_epochs: int = 1,
    calibration_epochs: int = 0,
    batch_size: int = 32,
    one_step_aux_epochs: int = 0,
    evaluate_generalization: bool = False,
    evaluation_replicates: int = 1,
) -> ExactAllocationExecutionReceipt:
    """Execute one real decision: exact collection -> train -> immutable receipt."""
    from awa.v2.game.training import train_game_stack

    if decision.stop:
        raise ValueError("cannot train a stop decision")
    option_keys = {option.decision.execution_key for option in offered_options}
    if decision.execution_key not in option_keys:
        raise ValueError("executed decision must appear in the offered decision menu")

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    dataset_path = out / "allocation_dataset.npz"
    collection, tasks = collect_exact_transition_dataset(
        decision,
        dataset_path,
        base_seed=int(base_seed),
        difficulty=float(difficulty),
        seed=int(seed),
        horizon=int(horizon),
        world_checkpoint=world_checkpoint,
        actor_checkpoint=actor_checkpoint,
        device=str(device),
    )
    training_dir = out / "trained"
    report = train_game_stack(
        dataset_path,
        tasks,
        training_dir,
        sequence_length=int(sequence_length),
        world_epochs=int(world_epochs),
        actor_epochs=int(actor_epochs),
        calibration_epochs=int(calibration_epochs),
        batch_size=int(batch_size),
        hidden=int(hidden),
        seed=int(seed),
        device=str(device),
        precision=str(precision),
        one_step_aux_epochs=int(one_step_aux_epochs),
        resume_world_checkpoint=world_checkpoint,
        resume_actor_checkpoint=actor_checkpoint,
        run_planner_diagnostics=False,
    )
    wall_seconds = float(time.perf_counter() - started)
    world_path = training_dir / "game_world.pt"
    actor_path = training_dir / "game_actor.pt"
    heldout_success_rate = 0.0
    heldout_mean_return = 0.0
    transfer_success_rate = 0.0
    transfer_mean_return = 0.0
    if evaluate_generalization:
        from awa.v2.curriculum.environment_factory import FactorizedEnvironmentFactory
        from awa.v2.game.procedural_runtime import evaluate_procedural_actor

        eval_factory = FactorizedEnvironmentFactory(
            base_seed=int(base_seed) + 100_000 + int(seed) * 37 + int(iteration) * 101
        )
        benchmark = eval_factory.build_canonical_benchmark(
            train_replicates=1,
            eval_replicates=int(evaluation_replicates),
            train_difficulty=float(difficulty),
            heldout_difficulty=min(1.0, float(difficulty) + 0.15),
            transfer_difficulty=min(1.0, float(difficulty) + 0.25),
        )
        heldout_eval = evaluate_procedural_actor(
            world_path, actor_path, benchmark.heldout,
            device=str(device), horizon=int(horizon), seed_offset=71,
        )
        transfer_eval = evaluate_procedural_actor(
            world_path, actor_path, benchmark.transfer,
            device=str(device), horizon=int(horizon), seed_offset=97,
        )
        heldout_success_rate = float(heldout_eval.success_rate)
        heldout_mean_return = float(heldout_eval.mean_return)
        transfer_success_rate = float(transfer_eval.success_rate)
        transfer_mean_return = float(transfer_eval.mean_return)
    core = {
        "decision": decision.to_dict(),
        "offered_options": [option.to_dict() for option in offered_options],
        "seed": int(seed),
        "iteration": int(iteration),
        "exact_transitions": int(collection["transitions"]),
        "collection_success_rate": float(collection["success_rate"]),
        "collection_mean_return": float(collection["mean_return"]),
        "actor_success_rate": float(report.actor_success_rate),
        "actor_mean_return": float(report.actor_mean_return),
        "world_loss": float(report.world_loss),
        "wall_seconds": wall_seconds,
        "dataset_sha256": _sha256_file(dataset_path),
        "world_checkpoint_sha256": _sha256_file(world_path),
        "actor_checkpoint_sha256": _sha256_file(actor_path),
        "output_dir": str(out),
        "generalization_evaluated": bool(evaluate_generalization),
        "heldout_success_rate": heldout_success_rate,
        "heldout_mean_return": heldout_mean_return,
        "transfer_success_rate": transfer_success_rate,
        "transfer_mean_return": transfer_mean_return,
    }
    receipt_sha256 = _sha256_json(core)
    receipt = ExactAllocationExecutionReceipt(
        decision=decision,
        offered_options=tuple(offered_options),
        seed=int(seed),
        iteration=int(iteration),
        exact_transitions=int(collection["transitions"]),
        collection_success_rate=float(collection["success_rate"]),
        collection_mean_return=float(collection["mean_return"]),
        actor_success_rate=float(report.actor_success_rate),
        actor_mean_return=float(report.actor_mean_return),
        world_loss=float(report.world_loss),
        wall_seconds=wall_seconds,
        dataset_sha256=core["dataset_sha256"],
        world_checkpoint_sha256=core["world_checkpoint_sha256"],
        actor_checkpoint_sha256=core["actor_checkpoint_sha256"],
        receipt_sha256=receipt_sha256,
        output_dir=str(out),
        generalization_evaluated=bool(evaluate_generalization),
        heldout_success_rate=heldout_success_rate,
        heldout_mean_return=heldout_mean_return,
        transfer_success_rate=transfer_success_rate,
        transfer_mean_return=transfer_mean_return,
    )
    (out / "allocation_execution_receipt.json").write_text(
        json.dumps({"format": "awa-v2.34-exact-allocation-execution-v1", **receipt.to_dict()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return receipt
