from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import json
import tempfile
import numpy as np
import torch

from awa.v2.curriculum import TaskSpec, TaskOutcome, ProceduralTaskFactory, LearningFrontierScheduler, ConceptCoverageTracker, AdaptiveTaskGenerator
from awa.v2.curriculum.verifiers import TaskVerifierRegistry
from awa.v2.experience import ExperienceAnalyzer, ExperienceRecord, RunningNovelty
from awa.v2.replay import PrioritizedExperienceBuffer, StructuralPriority, PriorityWeights
from awa.v2.reasoning import ReasoningEffortController
from awa.v2.distill import TeacherPool, DistillationBuffer
from awa.v2.skills import (
    SkillDiscovery, SkillRegistry, SkillSpec, SkillPolicyBuffer,
    DistilledContinuousSkill, SkillPolicyTrainer, ExecutableSkillLibrary,
    SkillContractModel,
)
from awa.v2.engram import EngramLite
from awa.v2.goals import HindsightGoalRelabeler
from .modes import TrainingMode, TrainingModeController, IntrinsicRewardComposer


@dataclass(frozen=True)
class ReusableLearningConfig:
    replay_capacity: int = 100_000
    novelty_capacity: int = 4096
    surprise_threshold: float = 0.25
    surprise_repeat_window: int = 4
    skill_min_examples: int = 5
    engram_capacity: int = 100_000
    her_future_goals: int = 4
    seed: int = 0

    def __post_init__(self):
        if int(self.replay_capacity) <= 0:
            raise ValueError("replay_capacity must be > 0")
        if int(self.novelty_capacity) <= 0:
            raise ValueError("novelty_capacity must be > 0")
        if not np.isfinite(float(self.surprise_threshold)) or float(self.surprise_threshold) < 0:
            raise ValueError("surprise_threshold must be finite and >= 0")
        if int(self.surprise_repeat_window) <= 0:
            raise ValueError("surprise_repeat_window must be > 0")
        if int(self.skill_min_examples) <= 0:
            raise ValueError("skill_min_examples must be > 0")
        if int(self.engram_capacity) <= 0:
            raise ValueError("engram_capacity must be > 0")
        if int(self.her_future_goals) <= 0:
            raise ValueError("her_future_goals must be > 0")


@dataclass(frozen=True)
class StepResult:
    record: ExperienceRecord
    shaped_reward: float
    mode: str
    verifier_success: bool


def _json_safe(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(v) for v in value]
    return str(value)


def _module_state_to_json(module: torch.nn.Module) -> dict:
    return {k: v.detach().cpu().tolist() for k, v in module.state_dict().items()}


def _load_module_state_from_json(module: torch.nn.Module, raw: dict) -> None:
    current = module.state_dict()
    state = {}
    for key, value in raw.items():
        if key not in current:
            raise KeyError(f"unexpected module state key: {key}")
        state[key] = torch.as_tensor(value, dtype=current[key].dtype, device=current[key].device)
    missing = set(current) - set(state)
    if missing:
        raise KeyError(f"missing module state keys: {sorted(missing)}")
    module.load_state_dict(state, strict=True)


class ReusableLearningEngine:
    """Closed-loop coordinator for Aether's reusable-learning components.

    This class intentionally does not own neural-network optimizers.  It turns live
    experience into consistent curriculum, replay, distillation, skill, memory and
    adaptation state that world-model/actor trainers can consume independently.
    """

    def __init__(self, config: ReusableLearningConfig | None = None):
        self.config = config or ReusableLearningConfig()
        c = self.config
        self.factory = ProceduralTaskFactory(c.seed)
        self.scheduler = LearningFrontierScheduler(seed=c.seed)
        self.coverage = ConceptCoverageTracker()
        self.adaptive_generator = AdaptiveTaskGenerator(self.factory, self.coverage)
        self.analyzer = ExperienceAnalyzer(c.surprise_threshold, c.surprise_repeat_window)
        self.novelty = RunningNovelty(c.novelty_capacity)
        self.replay = PrioritizedExperienceBuffer(c.replay_capacity, seed=c.seed)
        self.effort = ReasoningEffortController.default()
        self.teacher_pool = TeacherPool()
        self.distillation = DistillationBuffer()
        self.skill_discovery = SkillDiscovery(min_examples=c.skill_min_examples)
        self.skills = SkillRegistry()
        self.skill_policy_buffer = SkillPolicyBuffer()
        self.executable_skills = ExecutableSkillLibrary(self.skills)
        self.engram = EngramLite(capacity=c.engram_capacity, seed=c.seed)
        self.verifiers = TaskVerifierRegistry()
        self.goal_relabeler = HindsightGoalRelabeler(c.her_future_goals, seed=c.seed)
        self.mode_controller = TrainingModeController()
        self.reward_composer = IntrinsicRewardComposer()
        self.tasks: dict[str, TaskSpec] = {}
        self.total_steps = 0
        self.total_episodes = 0

    def register_tasks(self, tasks: list[TaskSpec]) -> None:
        for task in tasks:
            self.tasks[task.task_id] = task

    def task(self, task_id: str) -> TaskSpec:
        return self.tasks[task_id]

    def record_step(
        self,
        task: TaskSpec,
        *,
        state,
        goal,
        action,
        predicted_next,
        actual_next,
        task_reward: float,
        uncertainty: float,
        risk: float,
        td_error: float = 0.0,
        task_importance: float = 0.0,
        information_value: float = 0.0,
        success: bool | None = None,
        verifier_context: dict | None = None,
        mode: TrainingMode | None = None,
        metadata: dict | None = None,
    ) -> StepResult:
        self.register_tasks([task])
        if success is None:
            if verifier_context is None:
                raise ValueError("record_step requires success or verifier_context")
            success = self.verifiers.verify(task, verifier_context).success
        novelty = self.novelty.observe(np.concatenate([np.asarray(actual_next).reshape(-1), np.asarray(goal).reshape(-1)]))
        pred_error = self.analyzer.prediction_error(predicted_next, actual_next)
        current_mode = mode if mode is not None else self.mode_controller.step()
        reward = self.reward_composer.compose(
            task_reward,
            novelty=novelty,
            information=information_value,
            surprise=min(1.0, pred_error),
            mode=current_mode,
        )
        md = dict(metadata or {})
        md.update({"task_id": task.task_id, "training_mode": current_mode.value})
        rec = self.analyzer.make(
            state=state,
            goal=goal,
            action=action,
            predicted_next=predicted_next,
            actual_next=actual_next,
            reward=reward.total,
            success=success,
            novelty=novelty,
            uncertainty=uncertainty,
            risk=risk,
            td_error=td_error,
            task_importance=task_importance,
            information_value=information_value,
            metadata=md,
        )
        self.replay.add(rec)
        if rec.repeated_surprise:
            self.engram.put(
                "repeated_surprise",
                np.concatenate([rec.state.reshape(-1), rec.action.reshape(-1)]),
                {"task_id": task.task_id, "prediction_error": rec.prediction_error, "mode": current_mode.value},
                score=min(1.0, rec.prediction_error),
            )
        self.total_steps += 1
        return StepResult(rec, reward.total, current_mode.value, bool(success))

    def finish_episode(
        self,
        task: TaskSpec,
        records: list[ExperienceRecord],
        *,
        success: bool,
        reward: float,
        planner_actions: int = 0,
        total_actions: int | None = None,
        states=None,
        actions=None,
    ) -> dict:
        total = int(total_actions if total_actions is not None else max(1, len(records)))
        planner_actions = int(planner_actions)
        if total <= 0:
            raise ValueError("total_actions must be > 0")
        if planner_actions < 0 or planner_actions > total:
            raise ValueError("planner_actions must be in [0,total_actions]")
        planner_dependency = float(planner_actions) / total
        mean_error = float(np.mean([r.prediction_error for r in records])) if records else 0.0
        mean_novelty = float(np.mean([r.novelty for r in records])) if records else 0.0
        outcome = TaskOutcome(task.task_id, bool(success), float(reward), mean_error, mean_novelty, planner_dependency)
        self.scheduler.record(outcome)
        self.coverage.record(task, outcome)
        promoted = []
        if states is not None and actions is not None:
            sig = self.skill_discovery.observe(states, actions, success)
            skill_name = f"auto_{sig.key}"
            if success:
                goal = records[0].goal if records else None
                self.skill_policy_buffer.add_trajectory(skill_name, states, actions, goal=goal)
            for candidate in self.skill_discovery.candidates():
                was_known = self.skills.get(candidate.name) is not None
                if self.skills.register(candidate) and not was_known:
                    promoted.append(candidate.name)
        if success and states is not None:
            states_np = np.asarray(states, dtype=np.float32)
            if len(states_np):
                self.engram.put(
                    "successful_tactic",
                    states_np[0],
                    {"task_id": task.task_id, "planner_dependency": planner_dependency, "reward": float(reward)},
                    score=max(0.0, min(1.0, float(reward) if np.isfinite(reward) else 0.0)),
                )
        self.total_episodes += 1
        return {
            "task_id": task.task_id,
            "success": bool(success),
            "planner_dependency": planner_dependency,
            "prediction_error": mean_error,
            "novelty": mean_novelty,
            "promoted_skills": promoted,
        }

    def add_teacher_example(self, state, goal, candidates, *, student_value: float = 0.0, metadata: dict | None = None):
        ex = self.teacher_pool.example(state, goal, candidates, student_value, metadata)
        self.distillation.add(ex)
        return ex

    def relabel_episode(self, achieved_goals, actions, desired_goal, **kwargs):
        return self.goal_relabeler.relabel_episode(achieved_goals, actions, desired_goal, **kwargs)

    def augment_with_hindsight(self, records, achieved_goals, actions, desired_goal, *, add_to_replay: bool = True):
        relabeled = self.relabel_episode(achieved_goals, actions, desired_goal)
        out = []
        for row in relabeled:
            base = records[row.transition_index]
            md = dict(base.metadata or {})
            md.update({
                "hindsight": True,
                "original_goal": np.asarray(row.original_goal).tolist(),
                "source_future_index": int(row.source_future_index),
            })
            clone = ExperienceRecord(
                np.asarray(base.state, dtype=np.float32),
                np.asarray(row.relabeled_goal, dtype=np.float32),
                np.asarray(base.action, dtype=np.float32),
                np.asarray(base.predicted_next, dtype=np.float32),
                np.asarray(base.actual_next, dtype=np.float32),
                float(row.reward),
                bool(row.success),
                float(base.prediction_error),
                float(base.novelty),
                float(base.uncertainty),
                float(base.risk),
                float(base.td_error),
                float(base.task_importance),
                float(base.information_value),
                bool(base.repeated_surprise),
                md,
            )
            out.append(clone)
            if add_to_replay:
                self.replay.add(clone)
        return out

    def generate_adaptive_tasks(self, count: int = 4):
        tasks = self.adaptive_generator.generate(count)
        self.register_tasks(tasks)
        return tasks

    def train_skill_policy(self, skill_name: str, low, high, *, hidden: int = 128, epochs: int = 20, batch_size: int = 128, lr: float = 3e-4, seed: int = 0):
        states, actions, goals, weights = self.skill_policy_buffer.arrays(skill_name)
        goal_dim = 0 if goals is None else int(goals.shape[-1])
        model = DistilledContinuousSkill(states.shape[-1], actions.shape[-1], low, high, goal_dim=goal_dim, hidden=hidden)
        report = SkillPolicyTrainer(model, lr=lr).fit(states, actions, goals, weights, epochs=epochs, batch_size=batch_size, seed=seed)
        self.executable_skills.attach_policy(skill_name, model)
        return model, report

    def summary(self) -> dict:
        return {
            "steps": self.total_steps,
            "episodes": self.total_episodes,
            "tasks": len(self.tasks),
            "replay": len(self.replay),
            "distillation_examples": len(self.distillation),
            "skills": len(self.skills),
            "skill_policy_examples": len(self.skill_policy_buffer),
            "executable_skills": len(self.executable_skills.policies),
            "engram_entries": len(self.engram),
            "mode": self.mode_controller.mode.value,
            "weakest_concepts": [x.to_dict() for x in self.coverage.weakest(5)],
        }

    @staticmethod
    def _tuple_tree(value):
        if isinstance(value, list):
            return tuple(ReusableLearningEngine._tuple_tree(v) for v in value)
        if isinstance(value, dict):
            return {k: ReusableLearningEngine._tuple_tree(v) for k, v in value.items()}
        return value

    def state_dict(self) -> dict:
        scheduler_rows = {
            task_id: [asdict(row) for row in rows]
            for task_id, rows in self.scheduler._rows.items()
        }
        discovery = {}
        for key, rows in self.skill_discovery.groups.items():
            discovery[key] = [
                {
                    "key": r.key,
                    "start": r.start.tolist(),
                    "end": r.end.tolist(),
                    "mean_action": r.mean_action.tolist(),
                    "length": r.length,
                    "success": r.success,
                }
                for r in rows
            ]
        engrams = []
        seen = set()
        for bucket, key in self.engram._order:
            for entry in self.engram._by_bucket.get(bucket, []):
                if entry.key == key and entry.key not in seen:
                    seen.add(entry.key); engrams.append(entry.to_dict())
        return {
            "schema": "aether-reusable-learning-v2",
            "config": asdict(self.config),
            "tasks": [t.to_dict() for t in self.tasks.values()],
            "scheduler_rows": scheduler_rows,
            "scheduler_rng": self.scheduler.rng.getstate(),
            "coverage_rows": {k: list(v) for k, v in self.coverage._rows.items()},
            "adaptive_counter": int(self.adaptive_generator.counter),
            "novelty_items": [np.asarray(x).tolist() for x in self.novelty.items],
            "analyzer_recent": list(self.analyzer._recent),
            "replay": [r.to_dict() for r in self.replay.items],
            "replay_priorities": list(map(float, self.replay.priorities)),
            "replay_rng": self.replay.rng.bit_generator.state,
            "replay_priority": {
                "weights": asdict(self.replay.priority.weights),
                "clip": float(self.replay.priority.clip),
                "epsilon": float(self.replay.priority.epsilon),
            },
            "skills": [s.to_dict() for s in self.skills._skills.values()],
            "skill_policy_buffer": {
                name: [
                    {"state": x.state.tolist(), "action": x.action.tolist(), "goal": None if x.goal is None else x.goal.tolist(), "weight": x.weight}
                    for x in rows
                ]
                for name, rows in self.skill_policy_buffer._items.items()
            },
            "skill_discovery": discovery,
            "skill_discovery_config": {
                "round_decimals": int(self.skill_discovery.round_decimals),
                "min_examples": int(self.skill_discovery.min_examples),
            },
            "executable_skill_policies": {
                name: {
                    "state_dim": int(model.state_dim),
                    "action_dim": int(model.action_dim),
                    "goal_dim": int(model.goal_dim),
                    "hidden": int(model.hidden),
                    "low": model.low.detach().cpu().tolist(),
                    "high": model.high.detach().cpu().tolist(),
                    "state_dict": _module_state_to_json(model),
                }
                for name, model in self.executable_skills.policies.items()
            },
            "skill_contracts": {
                name: {
                    "state_dim": int(model.state_dim),
                    "goal_dim": int(model.goal_dim),
                    "hidden": int(model.hidden),
                    "state_dict": _module_state_to_json(model),
                }
                for name, model in self.executable_skills.contracts._models.items()
            },
            "engrams": engrams,
            "teacher_pool": {"min_margin": float(self.teacher_pool.min_margin)},
            "distillation": [
                {
                    "state": x.state.tolist(), "goal": x.goal.tolist(), "action": x.action.tolist(),
                    "teacher": x.teacher, "utility": x.utility, "weight": x.weight, "metadata": x.metadata,
                }
                for x in self.distillation.items
            ],
            "mode_controller": self.mode_controller.state_dict(),
            "effort": {
                "levels": [asdict(x) for x in self.effort.levels],
                "cost_lambda": self.effort.cost_lambda,
            },
            "reward_composer": {
                "novelty_weight": self.reward_composer.novelty_weight,
                "information_weight": self.reward_composer.information_weight,
                "surprise_weight": self.reward_composer.surprise_weight,
                "intrinsic_cap": self.reward_composer.intrinsic_cap,
            },
            "goal_relabeler": {
                "k_future": self.goal_relabeler.k_future,
                "tolerance": self.goal_relabeler.tolerance,
                "rng_state": self.goal_relabeler.rng.bit_generator.state,
            },
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }

    def load_state_dict(self, state: dict) -> None:
        schema = state.get("schema")
        if schema not in {"aether-reusable-learning-v1", "aether-reusable-learning-v2"}:
            raise ValueError("unsupported reusable-learning checkpoint schema")
        from awa.v2.curriculum.task_spec import TaskOutcome
        from awa.v2.experience.analyzer import ExperienceRecord
        from awa.v2.skills.discovery import TrajectorySignature
        from awa.v2.distill.teacher_pool import DistillationExample

        self.tasks.clear()
        for raw in state.get("tasks", []):
            raw = dict(raw); raw["concepts"] = tuple(raw.get("concepts", ()))
            self.tasks[raw["task_id"]] = TaskSpec(**raw)
        self.scheduler._rows.clear()
        for task_id, rows in state.get("scheduler_rows", {}).items():
            for raw in rows:
                self.scheduler.record(TaskOutcome(**raw))
        if state.get("scheduler_rng"):
            self.scheduler.rng.setstate(self._tuple_tree(state["scheduler_rng"]))
        self.coverage._rows.clear()
        for concept, rows in state.get("coverage_rows", {}).items():
            for value in rows:
                self.coverage._rows[concept].append(bool(value))
        self.adaptive_generator.counter = int(state.get("adaptive_counter", 0))
        self.novelty.items.clear()
        for x in state.get("novelty_items", []):
            self.novelty.items.append(np.asarray(x, dtype=np.float32))
        self.analyzer._recent = list(map(float, state.get("analyzer_recent", [])))
        self.replay.items.clear(); self.replay.priorities.clear()
        for raw in state.get("replay", []):
            raw = dict(raw)
            for k in ("state", "goal", "action", "predicted_next", "actual_next"):
                raw[k] = np.asarray(raw[k], dtype=np.float32)
            self.replay.items.append(ExperienceRecord(**raw))
        self.replay.priorities.extend(map(float, state.get("replay_priorities", [])))
        if len(self.replay.priorities) != len(self.replay.items):
            raise ValueError("checkpoint replay/priorities length mismatch")
        priority = state.get("replay_priority")
        if priority:
            self.replay.priority = StructuralPriority(
                PriorityWeights(**priority.get("weights", {})),
                clip=float(priority.get("clip", 5.0)),
                epsilon=float(priority.get("epsilon", 1e-4)),
            )
        if state.get("replay_rng") is not None:
            self.replay.rng.bit_generator.state = state["replay_rng"]
        self.skills._skills.clear()
        for raw in state.get("skills", []):
            raw = dict(raw); raw["preconditions"] = frozenset(raw.get("preconditions", [])); raw["effects"] = frozenset(raw.get("effects", []))
            self.skills._skills[raw["name"]] = SkillSpec(**raw)
        self.skill_policy_buffer._items.clear()
        for name, rows in state.get("skill_policy_buffer", {}).items():
            for raw in rows:
                self.skill_policy_buffer.add(name, raw["state"], raw["action"], raw.get("goal"), raw.get("weight", 1.0))
        skill_cfg = state.get("skill_discovery_config")
        if skill_cfg:
            self.skill_discovery.round_decimals = int(skill_cfg.get("round_decimals", self.skill_discovery.round_decimals))
            self.skill_discovery.min_examples = int(skill_cfg.get("min_examples", self.skill_discovery.min_examples))
        self.skill_discovery.groups.clear()
        for key, rows in state.get("skill_discovery", {}).items():
            self.skill_discovery.groups[key] = [TrajectorySignature(r["key"], np.asarray(r["start"], dtype=np.float32), np.asarray(r["end"], dtype=np.float32), np.asarray(r["mean_action"], dtype=np.float32), int(r["length"]), bool(r["success"])) for r in rows]
        self.executable_skills.policies.clear()
        self.executable_skills.contracts._models.clear()
        for name, raw in state.get("executable_skill_policies", {}).items():
            model = DistilledContinuousSkill(
                int(raw["state_dim"]), int(raw["action_dim"]), raw["low"], raw["high"],
                goal_dim=int(raw.get("goal_dim", 0)), hidden=int(raw.get("hidden", 128)),
            )
            _load_module_state_from_json(model, raw["state_dict"])
            self.executable_skills.attach_policy(name, model)
        for name, raw in state.get("skill_contracts", {}).items():
            model = SkillContractModel(
                int(raw["state_dim"]), goal_dim=int(raw.get("goal_dim", 0)), hidden=int(raw.get("hidden", 64))
            )
            _load_module_state_from_json(model, raw["state_dict"])
            self.executable_skills.contracts.attach(name, model)
        self.engram._by_bucket.clear(); self.engram._order.clear()
        for raw in state.get("engrams", []):
            entry = self.engram.put(raw["pattern"], raw["vector"], raw["payload"], score=raw.get("score", 1.0), key=raw["key"])
            entry.uses = int(raw.get("uses", 0))
        teacher = state.get("teacher_pool")
        if teacher:
            self.teacher_pool.min_margin = float(teacher.get("min_margin", self.teacher_pool.min_margin))
        self.distillation.items.clear()
        for raw in state.get("distillation", []):
            self.distillation.add(DistillationExample(np.asarray(raw["state"], dtype=np.float32), np.asarray(raw["goal"], dtype=np.float32), np.asarray(raw["action"], dtype=np.float32), raw["teacher"], float(raw["utility"]), float(raw["weight"]), raw.get("metadata")))
        self.mode_controller.load_state_dict(state.get("mode_controller", self.mode_controller.state_dict()))
        effort = state.get("effort")
        if effort:
            from awa.v2.reasoning.effort import EffortLevel
            self.effort = ReasoningEffortController([EffortLevel(**x) for x in effort["levels"]], cost_lambda=float(effort["cost_lambda"]))
        reward = state.get("reward_composer")
        if reward:
            self.reward_composer = IntrinsicRewardComposer(**reward)
        relabel = state.get("goal_relabeler")
        if relabel:
            self.goal_relabeler = HindsightGoalRelabeler(int(relabel["k_future"]), float(relabel["tolerance"]), seed=self.config.seed)
            self.goal_relabeler.rng.bit_generator.state = relabel["rng_state"]
        self.total_steps = int(state.get("total_steps", 0)); self.total_episodes = int(state.get("total_episodes", 0))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(_json_safe(self.state_dict()), sort_keys=True, separators=(",", ":"))
        with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as f:
            f.write(payload); tmp = Path(f.name)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ReusableLearningEngine":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        engine = cls(ReusableLearningConfig(**state["config"]))
        engine.load_state_dict(state)
        return engine
