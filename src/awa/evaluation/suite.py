from __future__ import annotations
from dataclasses import dataclass, asdict
import copy
import statistics
import torch
from awa.environments.delayed_cue import DelayedCueEnv
from awa.environments.robust_cue import RobustDelayedCueEnv


@dataclass
class SuiteResult:
    name: str
    success_rate: float
    mean_reward: float
    episodes: int


def _rollout(cfg, components, device, env, deterministic=True):
    import torch.nn.functional as F
    obs = env.reset(); belief = components.world_model.initial_belief(1, device)
    prev = torch.zeros(1, cfg["model"]["action_dim"], device=device); total=0.0; done=False; t=0
    while not done:
        o=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
        out=components.world_model.observe(belief,prev,o,step_index=t); belief=out.belief
        logits=components.actor.logits(belief.vector)
        idx=logits.argmax(-1) if deterministic else components.actor.distribution(belief.vector).sample()
        action=int(idx.item()); prev=F.one_hot(idx,cfg["model"]["action_dim"]).float()
        obs,r,done,_=env.step(action); total+=r; t+=1
    return total


@torch.no_grad()
def run_generalization_suite(cfg, components, device, episodes=20):
    ec=cfg["env"]
    specs=[
        ("baseline", dict(cue_delay=ec.get("cue_delay",8), noise=0.0, dropout=0.0)),
        ("long_delay", dict(cue_delay=max(ec.get("cue_delay",8)+4, int(ec.get("cue_delay",8)*1.5)), noise=0.0, dropout=0.0)),
        ("noise_0_10", dict(cue_delay=ec.get("cue_delay",8), noise=0.10, dropout=0.0)),
        ("dropout_0_20", dict(cue_delay=ec.get("cue_delay",8), noise=0.0, dropout=0.20)),
        ("noise_dropout", dict(cue_delay=ec.get("cue_delay",8), noise=0.10, dropout=0.20)),
    ]
    results=[]
    for name,s in specs:
        rewards=[]
        for ep in range(episodes):
            kwargs=dict(cue_delay=s["cue_delay"], horizon=max(ec["horizon"],s["cue_delay"]+4), obs_dim=ec["obs_dim"], seed=cfg["seed"]+5000+ep)
            if s["noise"] or s["dropout"]:
                env=RobustDelayedCueEnv(**kwargs, observation_noise=s["noise"], dropout_prob=s["dropout"])
            else:
                env=DelayedCueEnv(**kwargs)
            rewards.append(_rollout(cfg,components,device,env))
        results.append(SuiteResult(name, sum(r>0.5 for r in rewards)/episodes, statistics.mean(rewards), episodes))
    return [asdict(x) for x in results]
