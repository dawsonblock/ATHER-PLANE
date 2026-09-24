from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from awa.v2.training import ReusableLearningEngine, ReusableLearningConfig, TrainingMode
from awa.v2.curriculum.task_spec import TaskSpec
from .procedural_arena import ProceduralArenaEnv
from .teacher import LogicalArenaTeacher
from .her import augment_game_hindsight
from .counterfactual import add_counterfactual_branches


@dataclass(frozen=True)
class GameLoopReport:
    episodes: int
    steps: int
    success_rate: float
    mean_return: float
    hindsight_examples: int
    replay_size: int
    skills: int
    executable_skills: int
    weakest_concepts: list[dict]

    def to_dict(self): return asdict(self)


class ReusableGameLoop:
    """Connect the procedural game directly to ReusableLearningEngine.

    This layer is deliberately model-agnostic. A prediction callback may supply
    world-model next-state predictions/uncertainty; before a learned model exists,
    the loop uses a persistence prediction so surprise remains informative.
    """

    def __init__(self, engine: ReusableLearningEngine | None = None, *, horizon: int = 120):
        self.engine = engine or ReusableLearningEngine(ReusableLearningConfig())
        self.horizon = int(horizon)

    def run_episode(
        self,
        task: TaskSpec,
        policy=None,
        *,
        seed_offset: int = 0,
        prediction_fn=None,
        mode: TrainingMode | None = None,
        hindsight: bool = True,
        counterfactual_branches: int = 0,
    ) -> dict:
        policy = policy or LogicalArenaTeacher()
        env = ProceduralArenaEnv(task, horizon=self.horizon)
        obs = env.reset(seed=task.seed + int(seed_offset))
        records=[]; states=[]; actions=[]; achieved=[]; total=0.0; planner_actions=0
        done=False
        while not done:
            states.append(obs.copy())
            out = policy.act(env, obs) if hasattr(policy, "act") else policy(obs)
            used_planner=False
            if isinstance(out, tuple):
                action, used_planner = out
            else:
                action = out
            action=np.asarray(action,dtype=np.float32)
            goal=env.goal_vector().copy()
            if int(counterfactual_branches)>0:
                add_counterfactual_branches(
                    self.engine,env,task,obs,goal,action,count=int(counterfactual_branches),
                    episode_id=f"{task.task_id}:{seed_offset}",step=len(records),
                )
            nxt,reward,done,info=env.step(action)
            if prediction_fn is None:
                pred=obs.copy(); uncertainty=float(np.linalg.norm(nxt-obs))
            else:
                pred, uncertainty = prediction_fn(obs, action)
            constraint_labels=env.constraint_labels().copy()
            step=self.engine.record_step(
                task,state=obs,goal=goal,action=action,predicted_next=pred,actual_next=nxt,
                task_reward=reward,uncertainty=max(0.0,float(uncertainty)),risk=float(np.max(constraint_labels)),
                task_importance=task.difficulty,information_value=float(np.linalg.norm(nxt-obs)),
                success=bool(info.get("success",False)),mode=mode,
                metadata={"episode_id":f"{task.task_id}:{seed_offset}","step":len(records),"done":bool(done),"episode_seed_offset":int(seed_offset),"stage":task.stage,"active_objective":env.active_objective,"constraint_labels":constraint_labels.tolist()},
            )
            records.append(step.record); actions.append(action.copy()); achieved.append(nxt[0:2].copy())
            planner_actions += int(bool(used_planner)); total += float(reward); obs=nxt
        summary=self.engine.finish_episode(
            task,records,success=bool(info.get("success",False)),reward=total,
            planner_actions=planner_actions,total_actions=len(records),states=np.asarray(states),actions=np.asarray(actions),
        )
        her=[]
        if hindsight and records and not info.get("success",False):
            her=augment_game_hindsight(self.engine,records,k_future=min(2,self.engine.config.her_future_goals),seed=task.seed+int(seed_offset))
        return {**summary,"return":float(total),"steps":len(records),"hindsight_examples":len(her)}

    def run_tasks(self, tasks: list[TaskSpec], *, episodes_per_task: int = 1, policy=None, mode: TrainingMode | None = None, counterfactual_branches: int = 0) -> GameLoopReport:
        rows=[]
        for task in tasks:
            for i in range(int(episodes_per_task)):
                rows.append(self.run_episode(task,policy,seed_offset=i*7919,mode=mode,counterfactual_branches=counterfactual_branches))
        return GameLoopReport(
            episodes=len(rows),steps=int(sum(r["steps"] for r in rows)),
            success_rate=float(np.mean([r["success"] for r in rows])) if rows else 0.0,
            mean_return=float(np.mean([r["return"] for r in rows])) if rows else 0.0,
            hindsight_examples=int(sum(r["hindsight_examples"] for r in rows)),
            replay_size=len(self.engine.replay),skills=len(self.engine.skills),
            executable_skills=len(self.engine.executable_skills.policies),
            weakest_concepts=[x.to_dict() for x in self.engine.coverage.weakest(5)],
        )

    def save(self, path: str | Path):
        return self.engine.save(path)

    @classmethod
    def load(cls, path: str | Path, *, horizon: int = 120):
        return cls(ReusableLearningEngine.load(path),horizon=horizon)
