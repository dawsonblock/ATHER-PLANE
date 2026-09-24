from __future__ import annotations
import copy
import numpy as np
import torch
from awa.environments.factory import make_environment
from awa.actions import action_spec_from_config


def _env_action(spec, seed: int):
    g=torch.Generator(device='cpu'); g.manual_seed(int(seed))
    if spec.kind=='discrete':
        idx=int(torch.randint(0,spec.dim,(1,),generator=g).item()); return idx
    low=torch.as_tensor(spec.low,dtype=torch.float32); high=torch.as_tensor(spec.high,dtype=torch.float32)
    u=torch.rand(spec.dim,generator=g); return (low+(high-low)*u).numpy()


def qualify_environment(cfg: dict, steps: int=8) -> dict:
    env=make_environment(cfg,cfg.get('seed',0)+17000); spec=action_spec_from_config(cfg); obs=env.reset()
    finite=bool(np.isfinite(np.asarray(obs)).all()); rewards=[]; completed=False; obs_shape=list(np.asarray(obs).shape)
    snapshot_roundtrip=None
    if hasattr(env,'state_dict') and hasattr(env,'load_state_dict'):
        try:
            snap=copy.deepcopy(env.state_dict()); action=_env_action(spec,cfg.get('seed',0)+999)
            o1,r1,d1,_=env.step(action); env.load_state_dict(copy.deepcopy(snap)); o2,r2,d2,_=env.step(action)
            snapshot_roundtrip=bool(np.allclose(np.asarray(o1),np.asarray(o2),rtol=1e-6,atol=1e-7) and abs(float(r1)-float(r2))<1e-7 and bool(d1)==bool(d2))
            env.load_state_dict(copy.deepcopy(snap)); obs=env.reset() if False else obs
        except Exception:
            snapshot_roundtrip=False
    for i in range(int(steps)):
        action=_env_action(spec,cfg.get('seed',0)+i)
        obs,r,done,_=env.step(action); rewards.append(float(r)); finite=bool(finite and np.isfinite(np.asarray(obs)).all() and np.isfinite(r))
        if done: completed=True; obs=env.reset()
    close=getattr(env,'close',None)
    if close is not None: close()
    return {'kind':cfg['env'].get('kind','delayed_cue'),'action_type':spec.kind,'obs_dim':int(cfg['env']['obs_dim']),'action_dim':int(spec.dim),'observation_shape':obs_shape,'steps':int(steps),'finite':finite,'episode_completed':completed,'snapshot_roundtrip':snapshot_roundtrip,'mean_reward':float(np.mean(rewards)) if rewards else 0.0}


def qualify_environment_multiseed(cfg: dict, seeds=(1,2,3,4,5), steps: int=16) -> dict:
    """Run environment interface qualification over several independent seeds.

    Failures are captured per seed so optional benchmark adapters cannot be
    accidentally described as qualified merely because they import.
    """
    reports=[]
    for seed in seeds:
        local=copy.deepcopy(cfg); local['seed']=int(seed)
        try:
            report=qualify_environment(local,steps); report['seed']=int(seed); report['ok']=bool(report['finite'] and report['obs_dim']>0 and report['action_dim']>0)
        except Exception as exc:
            report={'seed':int(seed),'ok':False,'error_type':type(exc).__name__,'error':str(exc)}
        reports.append(report)
    ok=sum(int(r.get('ok',False)) for r in reports)
    snapshot=[r.get('snapshot_roundtrip') for r in reports if 'snapshot_roundtrip' in r and r.get('snapshot_roundtrip') is not None]
    return {
        'environment':cfg.get('env',{}).get('kind','delayed_cue'),
        'seeds':[int(s) for s in seeds],
        'passed':ok,
        'failed':len(reports)-ok,
        'pass_rate':ok/max(1,len(reports)),
        'snapshot_roundtrip_rate':(sum(bool(x) for x in snapshot)/len(snapshot)) if snapshot else None,
        'reports':reports,
    }
