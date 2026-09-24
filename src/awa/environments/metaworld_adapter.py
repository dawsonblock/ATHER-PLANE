from __future__ import annotations
import numpy as np


class MetaWorldContinuousAdapter:
    """Optional Meta-World MT1 adapter.

    Meta-World changes APIs across releases; this adapter uses MT1 when present
    and fails with an explicit message rather than silently guessing.
    """
    def __init__(self, task_name: str='reach-v3', seed: int=0):
        try:
            import metaworld
        except ImportError as e:
            raise ImportError("Install Meta-World separately to use this adapter") from e
        if not hasattr(metaworld,'MT1'):
            raise RuntimeError('Installed Meta-World does not expose MT1; update the adapter for that release')
        bench=metaworld.MT1(task_name,seed=seed)
        env_cls=bench.train_classes[task_name]; self.env=env_cls()
        tasks=list(bench.train_tasks)
        if not tasks: raise RuntimeError('Meta-World MT1 returned no train tasks')
        self.env.set_task(tasks[0]); self.seed=seed
        obs,_=self.env.reset(seed=seed) if hasattr(self.env,'reset') else (self.env.reset(),{})
        self._obs_dim=int(np.asarray(obs,dtype=np.float32).size)
        self._low=np.asarray(self.env.action_space.low,dtype=np.float32).reshape(-1); self._high=np.asarray(self.env.action_space.high,dtype=np.float32).reshape(-1)

    @property
    def obs_dim(self): return self._obs_dim
    @property
    def action_dim(self): return int(self._low.size)
    @property
    def action_low(self): return self._low.copy()
    @property
    def action_high(self): return self._high.copy()
    @property
    def action_type(self): return 'continuous'

    def reset(self):
        out=self.env.reset(seed=self.seed); self.seed+=1
        obs=out[0] if isinstance(out,tuple) else out
        return np.asarray(obs,dtype=np.float32).reshape(-1)

    def step(self, action):
        a=np.clip(np.asarray(action,dtype=np.float32).reshape(-1),self._low,self._high)
        out=self.env.step(a)
        if len(out)==5:
            obs,reward,terminated,truncated,info=out; done=bool(terminated or truncated)
        else:
            obs,reward,done,info=out
        return np.asarray(obs,dtype=np.float32).reshape(-1),float(reward),bool(done),dict(info)

    def close(self):
        fn=getattr(self.env,'close',None)
        if fn is not None: fn()
