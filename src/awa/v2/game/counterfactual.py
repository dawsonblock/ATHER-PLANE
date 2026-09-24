from __future__ import annotations
import numpy as np
from awa.v2.branching import capture_snapshot, restore_snapshot


def alternative_actions(action, count: int = 2):
    a=np.asarray(action,dtype=np.float32).reshape(-1); rows=[]
    if int(count)<=0: return rows
    b=a.copy(); b[:2]*=-1; rows.append(b)
    if len(rows)<count:
        c=a.copy(); c[0],c[1]=-a[1],a[0]; c[2:]=0; rows.append(c)
    if len(rows)<count:
        z=np.zeros_like(a); rows.append(z)
    return rows[:int(count)]


def add_counterfactual_branches(engine, env, task, obs, goal, actual_action, *, count: int = 2, episode_id: str = "episode", step: int = 0):
    """Branch alternative actions from the identical real simulator snapshot."""
    if int(count)<=0: return []
    snap=capture_snapshot(env); rows=[]
    try:
        for j,action in enumerate(alternative_actions(actual_action,count)):
            restore_snapshot(env,snap)
            nxt,reward,done,info=env.step(action)
            next_goal=env.goal_vector().copy(); labels=env.constraint_labels().copy()
            rec=engine.record_step(
                task,state=np.asarray(obs,dtype=np.float32),goal=np.asarray(goal,dtype=np.float32),
                action=action,predicted_next=np.asarray(obs,dtype=np.float32),actual_next=nxt,
                task_reward=float(reward),uncertainty=float(np.linalg.norm(np.asarray(nxt)-np.asarray(obs))),
                risk=float(labels.max()),task_importance=task.difficulty,
                information_value=float(np.linalg.norm(np.asarray(nxt)-np.asarray(obs))),success=bool(info.get('success',False)),
                metadata={
                    'counterfactual':True,'done':True,'episode_id':f'cf:{episode_id}:{step}:{j}','step':0,
                    'source_episode_id':episode_id,'source_step':int(step),'next_goal':next_goal.tolist(),'constraint_labels':labels.tolist(),
                },
            ).record
            rows.append(rec)
    finally:
        restore_snapshot(env,snap)
    return rows
