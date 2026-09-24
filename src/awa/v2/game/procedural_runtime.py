from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch

from awa.v2.actor_baseline import OfflineActorCriticBaseline
from awa.v2.world import MultimodalWorldModel, RiskConstraintModel
from .belief import GameBeliefEncoder
from .procedural_arena import ACTION_DIM, OBS_DIM, ProceduralArenaEnv
from .goal_program import GOAL_DIM


@dataclass(frozen=True)
class ProceduralActorEvaluation:
    episodes: int
    success_rate: float
    mean_return: float
    constraint_violations: float
    inference_latency_ms: float

    def to_dict(self):
        return asdict(self)


@dataclass
class LoadedProceduralGameStack:
    encoder: GameBeliefEncoder
    world: MultimodalWorldModel
    actor: OfflineActorCriticBaseline
    device: torch.device


def load_procedural_game_stack(
    world_checkpoint: str | Path,
    actor_checkpoint: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> LoadedProceduralGameStack:
    """Load the exact v2.10+ procedural game checkpoints for evaluation-only use.

    The loader exists so qualification can train once and evaluate many disjoint
    scenario sets without invoking the training routine again. It never mutates the
    checkpoint and places all modules in eval mode.
    """
    dev = torch.device(device)
    wc = torch.load(Path(world_checkpoint), map_location=dev, weights_only=False)
    ac = torch.load(Path(actor_checkpoint), map_location=dev, weights_only=False)
    if wc.get("format") != "awa-v2.10-game-world-v1":
        raise ValueError("unsupported procedural world checkpoint format")
    if ac.get("format") != "awa-v2.10-game-actor-v1":
        raise ValueError("unsupported procedural actor checkpoint format")
    for name, expected in (("obs_dim", OBS_DIM), ("goal_dim", GOAL_DIM), ("action_dim", ACTION_DIM)):
        if int(wc.get(name, -1)) != int(expected):
            raise ValueError(f"procedural checkpoint {name} mismatch")
    if int(ac.get("state_dim", -1)) != int(wc.get("belief_dim", -2)):
        raise ValueError("actor/world belief dimension mismatch")
    if int(ac.get("action_dim", -1)) != ACTION_DIM:
        raise ValueError("actor action dimension mismatch")

    cfg = dict(wc["encoder_config"])
    encoder = GameBeliefEncoder(**cfg).to(dev)
    encoder.load_state_dict(wc["encoder"], strict=True)

    world = MultimodalWorldModel(
        int(wc["belief_dim"]),
        ACTION_DIM,
        hidden=int(wc["hidden"]),
        components=3,
        horizons=tuple(int(x) for x in wc["horizons"]),
    ).to(dev)
    state = wc["world"]
    risk_hidden = int(wc.get("risk_hidden", state["risk.net.0.weight"].shape[0]))
    risk_constraints = int(wc.get("risk_constraints", state["risk.net.4.weight"].shape[0]))
    world.risk = RiskConstraintModel(
        int(wc["belief_dim"]), ACTION_DIM, hidden=risk_hidden, constraints=risk_constraints
    ).to(dev)
    world.load_state_dict(state, strict=True)

    actor = OfflineActorCriticBaseline(
        int(ac["state_dim"]), ACTION_DIM, [-1.0] * ACTION_DIM, [1.0] * ACTION_DIM,
        hidden=int(ac["hidden"]), device=dev,
    )
    astate = ac["state"]
    actor.actor.load_state_dict(astate["actor"], strict=True)
    actor.target_actor.load_state_dict(astate.get("target_actor", astate["actor"]), strict=True)
    actor.critic.load_state_dict(astate["critic"], strict=True)
    actor.target_critic.load_state_dict(astate["target_critic"], strict=True)
    actor.steps = int(astate.get("steps", 0))

    encoder.eval(); world.eval(); actor.actor.eval(); actor.target_actor.eval(); actor.critic.eval(); actor.target_critic.eval()
    for module in (encoder, world, actor.actor, actor.target_actor, actor.critic, actor.target_critic):
        for p in module.parameters():
            p.requires_grad = False
    return LoadedProceduralGameStack(encoder, world, actor, dev)


@torch.no_grad()
def evaluate_procedural_actor(
    world_checkpoint: str | Path,
    actor_checkpoint: str | Path,
    tasks,
    *,
    device: str | torch.device = "cpu",
    horizon: int = 120,
    seed_offset: int = 31,
) -> ProceduralActorEvaluation:
    """Evaluate actor-only behavior on a fixed task set without any training step."""
    runtime = load_procedural_game_stack(world_checkpoint, actor_checkpoint, device=device)
    rows = []
    for task in list(tasks):
        env = ProceduralArenaEnv(task, horizon=int(horizon))
        obs = env.reset(seed=int(task.seed) + int(seed_offset))
        temporal = runtime.encoder.initial(1, runtime.device)
        prev_action = torch.zeros(1, ACTION_DIM, device=runtime.device)
        step = 0
        done = False
        total = 0.0
        violations = 0
        decision_ms = []
        info = {}
        while not done:
            temporal, belief = runtime.encoder.observe(
                temporal, obs, prev_action, env.goal_vector(), step
            )
            t0 = time.perf_counter()
            action = runtime.actor.action(belief).squeeze(0).cpu().numpy().astype(np.float32)
            decision_ms.append((time.perf_counter() - t0) * 1000.0)
            obs, reward, done, info = env.step(action)
            total += float(reward)
            violations += int(np.any(env.constraint_labels() > 0.5))
            prev_action = torch.as_tensor(action, dtype=torch.float32, device=runtime.device).unsqueeze(0)
            step += 1
        rows.append((bool(info.get("success", False)), total, float(violations), float(np.mean(decision_ms) if decision_ms else 0.0)))
    if not rows:
        raise ValueError("evaluation task set is empty")
    return ProceduralActorEvaluation(
        episodes=len(rows),
        success_rate=float(np.mean([x[0] for x in rows])),
        mean_return=float(np.mean([x[1] for x in rows])),
        constraint_violations=float(np.mean([x[2] for x in rows])),
        inference_latency_ms=float(np.mean([x[3] for x in rows])),
    )
