from __future__ import annotations
from dataclasses import dataclass
import statistics
import torch
from awa.utils import seed_everything, resolve_device
from awa.training.trainer import train, evaluate


@dataclass
class BenchmarkResult:
    seeds: list[int]
    success_rates: list[float]
    mean_rewards: list[float]

    def summary(self):
        return {
            "seeds": self.seeds,
            "success_mean": statistics.mean(self.success_rates),
            "success_stdev": statistics.stdev(self.success_rates) if len(self.success_rates) > 1 else 0.0,
            "reward_mean": statistics.mean(self.mean_rewards),
            "reward_stdev": statistics.stdev(self.mean_rewards) if len(self.mean_rewards) > 1 else 0.0,
        }


def run_multiseed(cfg: dict, seeds: list[int], out_root: str, episodes: int = 25):
    sr, mr = [], []
    for seed in seeds:
        local = {**cfg, "seed": seed}
        seed_everything(seed)
        device = resolve_device(local.get("device", "auto"))
        c = train(local, device, f"{out_root}/seed_{seed}")
        r = evaluate(local, c, device, episodes=episodes)
        sr.append(r["success_rate"]); mr.append(r["mean_reward"])
    return BenchmarkResult(seeds, sr, mr)
