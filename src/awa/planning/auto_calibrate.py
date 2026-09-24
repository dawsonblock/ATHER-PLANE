from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch

from awa.environments.factory import make_environment


@dataclass
class CalibrationReport:
    samples: int
    failure_threshold: float
    failure_rate: float
    fit_loss: float

    def as_dict(self): return self.__dict__.copy()


@torch.no_grad()
def collect_transition_uncertainty(cfg, components, device, episodes: int=5, max_samples: int=2000):
    """Collect action-conditioned disagreement and realized next-latent error."""
    us=[]; errors=[]
    for ep in range(int(episodes)):
        env=make_environment(cfg,cfg.get('seed',0)+9000+ep); obs=env.reset()
        belief=components.world_model.initial_belief(1,device); prev=components.action_spec.zero(1,device); done=False; t=0
        while not done and len(us)<max_samples:
            o=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
            post=components.world_model.observe(belief,prev,o,step_index=t); belief=post.belief
            if components.action_spec.kind=='discrete':
                action,idx=components.actor.deterministic_action(belief.vector); env_action=int(idx.item())
            else:
                action=components.actor.deterministic_action(belief.vector); env_action=action.squeeze(0).cpu().numpy()
            pred,u=components.uncertainty(belief.vector,action)
            nxt,_,done,_=env.step(env_action)
            target=components.world_model.encoder(torch.as_tensor(nxt,dtype=torch.float32,device=device).unsqueeze(0))
            err=((pred-target)**2).mean(-1)
            us.append(float(u.item())); errors.append(float(err.item()))
            obs=nxt; prev=action; t+=1
        close=getattr(env,'close',None)
        if close is not None: close()
        if len(us)>=max_samples: break
    return np.asarray(us,dtype=np.float32),np.asarray(errors,dtype=np.float32)


def fit_calibrator_from_errors(calibrator, uncertainty, errors, quantile: float=0.75, steps: int=300, lr: float=0.05):
    u=np.asarray(uncertainty,dtype=np.float32).reshape(-1); e=np.asarray(errors,dtype=np.float32).reshape(-1)
    if u.size==0 or u.size!=e.size: raise ValueError('equal non-empty uncertainty/error arrays required')
    q=float(np.clip(quantile,0.05,0.95)); threshold=float(np.quantile(e,q)); failures=(e>=threshold).astype(np.float32)
    loss=calibrator.fit_numpy(u,failures,steps=steps,lr=lr)
    return CalibrationReport(int(u.size),threshold,float(failures.mean()),float(loss))
