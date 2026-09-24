from __future__ import annotations
from enum import Enum
import numpy as np
from awa.v2.transfer import OODCategory, TransferEpisode, TransferBenchmark, build_transfer_suite
from awa.v2.curriculum import ProceduralTaskFactory
from .procedural_arena import ProceduralArenaEnv


class AdaptationMode(str, Enum):
    ZERO_SHOT = "zero_shot"
    IN_CONTEXT = "in_context"
    LEARNING = "learning"


def evaluate_game_policy_transfer(
    policy,
    *,
    base_seed: int = 0,
    count_per_category: int = 2,
    exposures: int = 4,
    horizon: int = 120,
    observation_adapter=None,
    adaptation_mode: AdaptationMode | str = AdaptationMode.ZERO_SHOT,
    prediction_fn=None,
):
    """Evaluate true zero-shot, persistent-context, or learning adaptation separately."""
    mode=AdaptationMode(adaptation_mode)
    factory=ProceduralTaskFactory(base_seed)
    suite=build_transfer_suite(factory,count_per_category=count_per_category,difficulty=.65)
    rows=[]
    for category,tasks in suite.items():
        for task in tasks:
            if hasattr(policy,"begin_task"):
                policy.begin_task(task, preserve_context=mode is AdaptationMode.IN_CONTEXT)
            for exposure in range(int(exposures)):
                env=ProceduralArenaEnv(task,horizon=horizon)
                structured=env.reset(seed=task.seed+exposure*104729)
                obs = observation_adapter(env, structured) if observation_adapter is not None else structured
                done=False; total=0.0; steps=0; planners=0; pred_errors=[]; trajectory=[]
                # Zero-shot explicitly resets policy memory every episode. In-context keeps it.
                if mode is AdaptationMode.ZERO_SHOT and hasattr(policy,"reset_task"):
                    policy.reset_task(task)
                elif mode is AdaptationMode.LEARNING and hasattr(policy,"reset_episode"):
                    policy.reset_episode(task)
                while not done:
                    out=policy.act(env,obs) if hasattr(policy,"act") else policy(obs)
                    if isinstance(out,tuple): action,used_planner=out; planners+=int(bool(used_planner))
                    else: action=out
                    if prediction_fn is None and hasattr(policy,"predict_next"):
                        pred=policy.predict_next(env,obs,action)
                    elif prediction_fn is not None:
                        pred=prediction_fn(env,obs,action)
                    else:
                        pred=None
                    nxt,reward,done,info=env.step(action)
                    if pred is not None:
                        p=np.asarray(pred,dtype=np.float32).reshape(-1); n=np.asarray(nxt,dtype=np.float32).reshape(-1)
                        if p.shape==n.shape: pred_errors.append(float(np.sqrt(np.mean(np.square(p-n)))))
                    trajectory.append((np.asarray(structured).copy(),np.asarray(action).copy(),float(reward),np.asarray(nxt).copy(),bool(done)))
                    structured=nxt; obs = observation_adapter(env, structured) if observation_adapter is not None else structured
                    total+=float(reward); steps+=1
                rows.append(TransferEpisode(category,task.task_id,exposure,bool(info.get("success",False)),total,planners/max(1,steps),float(np.mean(pred_errors)) if pred_errors else 0.0))
                if mode is AdaptationMode.LEARNING and hasattr(policy,"adapt"):
                    policy.adapt(task,trajectory)
            if hasattr(policy,"end_task"):
                policy.end_task(task)
    return TransferBenchmark().evaluate(rows)
