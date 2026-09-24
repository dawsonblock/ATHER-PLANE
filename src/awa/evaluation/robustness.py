from __future__ import annotations
import torch
import torch.nn.functional as F
from awa.environments.robust_cue import RobustDelayedCueEnv


@torch.no_grad()
def evaluate_robustness(cfg, components, device, episodes=20,
                        noise_levels=(0.0, 0.05, 0.1), dropout_levels=(0.0, 0.1, 0.25)):
    ec = cfg["env"]
    rows = []
    for noise in noise_levels:
        for dropout in dropout_levels:
            successes = 0; rewards = []
            for ep in range(episodes):
                env = RobustDelayedCueEnv(
                    cue_delay=ec.get("cue_delay", 8), horizon=ec["horizon"], obs_dim=ec["obs_dim"],
                    seed=cfg["seed"] + 5000 + ep, observation_noise=noise, dropout_prob=dropout,
                )
                obs = env.reset(); belief = components.world_model.initial_belief(1, device)
                prev = torch.zeros(1, cfg["model"]["action_dim"], device=device)
                total=0.0; done=False; t=0
                while not done:
                    o=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0)
                    out=components.world_model.observe(belief,prev,o,step_index=t); belief=out.belief
                    idx=components.actor.logits(belief.vector).argmax(-1)
                    action=int(idx.item()); prev=F.one_hot(idx,cfg["model"]["action_dim"]).float()
                    obs,reward,done,_=env.step(action); total+=reward; t+=1
                successes += int(total > 0.5); rewards.append(total)
            rows.append({"noise":noise,"dropout":dropout,"success_rate":successes/episodes,
                         "mean_reward":sum(rewards)/episodes})
    return rows
