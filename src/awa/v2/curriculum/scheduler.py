from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass
import math, random
from .task_spec import TaskSpec, TaskOutcome

@dataclass(frozen=True)
class TaskStats:
    attempts:int
    success_rate:float
    learning_progress:float
    mean_prediction_error:float
    mean_novelty:float
    planner_dependency:float

class LearningFrontierScheduler:
    """Adaptive curriculum centered on learning progress rather than raw failure rate."""
    def __init__(self, history=50, target_low=.30, target_high=.80, seed=0):
        self.history=int(history); self.target_low=float(target_low); self.target_high=float(target_high)
        self._rows=defaultdict(lambda: deque(maxlen=self.history)); self.rng=random.Random(seed)

    def record(self, outcome:TaskOutcome): self._rows[outcome.task_id].append(outcome)

    def stats(self, task_id:str) -> TaskStats:
        rows=list(self._rows[task_id])
        if not rows: return TaskStats(0,.5,1.0,0.0,1.0,1.0)
        sr=sum(r.success for r in rows)/len(rows)
        half=max(1,len(rows)//2); old=rows[:half]; new=rows[half:] or rows
        old_sr=sum(r.success for r in old)/len(old); new_sr=sum(r.success for r in new)/len(new)
        lp=max(0.0,new_sr-old_sr)
        return TaskStats(len(rows),sr,lp,
            sum(r.prediction_error for r in rows)/len(rows),
            sum(r.novelty for r in rows)/len(rows),
            sum(r.planner_dependency for r in rows)/len(rows))

    def priority(self, task:TaskSpec) -> float:
        s=self.stats(task.task_id)
        # Frontier preference: peak around 55% success; learning progress and novelty increase priority.
        frontier=math.exp(-((s.success_rate-.55)/.28)**2)
        cold=1.0/math.sqrt(1+s.attempts)
        return max(1e-6, .45*frontier + .25*s.learning_progress + .15*min(1,s.mean_novelty) + .10*min(1,s.mean_prediction_error) + .05*cold)

    def sample(self,tasks:list[TaskSpec],n:int=1):
        if not tasks: raise ValueError('tasks cannot be empty')
        weights=[self.priority(t) for t in tasks]
        return self.rng.choices(tasks,weights=weights,k=int(n))

    def stage_mix(self,current:list[TaskSpec],mastered:list[TaskSpec],harder:list[TaskSpec],novel:list[TaskSpec],n:int):
        """Approximate 50/25/15/10 mix while allowing empty pools."""
        pools=[(current,.50),(mastered,.25),(harder,.15),(novel,.10)]
        available=[(p,w) for p,w in pools if p]
        if not available: raise ValueError('at least one pool must be non-empty')
        out=[]
        for _ in range(int(n)):
            total=sum(w for _,w in available); x=self.rng.random()*total; acc=0
            for pool,w in available:
                acc+=w
                if x<=acc:
                    out.extend(self.sample(pool,1)); break
        return out
