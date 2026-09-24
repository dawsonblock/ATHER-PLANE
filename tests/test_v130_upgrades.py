import json
import numpy as np
import pytest
import torch

from awa.config import validate_config, ConfigError
from awa.memory.replay import PrioritizedEpisodeReplayBuffer, Transition
from awa.model import WorldModel
from awa.runtime.checkpoint import CheckpointManager
from awa.planning.actor import CategoricalActor
from awa.planning.uncertainty import UncertaintyEnsemble
from awa.environments.delayed_cue import DelayedCueEnv
from awa.environments.robust_cue import RobustDelayedCueEnv
from awa.evaluation.calibration import expected_calibration_error, brier_score
from awa.reasoning.adapters import CallableReasoner


def _cfg():
    return {
        "seed": 1, "device": "cpu",
        "env": {"horizon": 16, "cue_delay": 4, "obs_dim": 8, "action_dim": 3},
        "model": {"obs_dim": 8, "latent_dim": 8, "deterministic_dim": 16,
                  "stochastic_dim": 4, "hidden_dim": 16, "action_dim": 3, "dynamics": "gru",
                  "slow_enabled": False, "experts_enabled": False},
        "planner": {"enabled": True, "horizon": 2, "candidates": 4, "elites": 2,
                    "iterations": 1, "uncertainty_limit": 1.0},
        "training": {"steps": 10, "batch_size": 2, "sequence_length": 2,
                     "replay_capacity": 100, "discount": 0.99, "lambda": 0.95},
    }


def test_config_validation_detects_dim_mismatch():
    cfg = _cfg(); cfg["model"]["obs_dim"] = 9
    with pytest.raises(ConfigError):
        validate_config(cfg)


def test_prioritized_sequence_replay_and_priority_update():
    r = PrioritizedEpisodeReplayBuffer(100, alpha=0.6, beta=0.4)
    for ep in range(3):
        for i in range(5):
            r.add(Transition(np.array([ep, i], np.float32), i % 2, float(i),
                             np.array([ep, i + 1], np.float32), i == 4))
    batch = r.sample_sequences(3, 3, torch.device("cpu"))
    assert batch.importance_weights.shape == (3,)
    assert len(batch.sequence_ids) == 3
    r.update_priorities(batch.sequence_ids, np.array([1.0, 2.0, 3.0]))
    assert max(r.priorities[s] for s in batch.sequence_ids) >= 3.0
    state = r.state_dict()
    r2 = PrioritizedEpisodeReplayBuffer(10)
    r2.load_state_dict(state)
    assert r2.valid_sequences(3) == r.valid_sequences(3)
    assert r2.priorities == r.priorities


def test_expert_world_model_routes():
    wm = WorldModel(8, 8, 16, 4, 3, 16, "gru", experts_enabled=True,
                    experts_count=4, experts_top_k=2)
    b = wm.initial_belief(3, torch.device("cpu"))
    out = wm.observe(b, torch.zeros(3, 3), torch.randn(3, 8))
    assert out.router_weights.shape == (3, 4)
    assert torch.allclose(out.router_weights.sum(-1), torch.ones(3), atol=1e-5)


def test_bundle_checkpoint_restores_modules_and_environment(tmp_path):
    wm = WorldModel(8, 8, 16, 4, 3, 16, "gru")
    actor = CategoricalActor(wm.belief_dim, 3, 16)
    unc = UncertaintyEnsemble(wm.belief_dim, 8, 2, 16)
    opt = torch.optim.Adam(wm.parameters(), lr=1e-3)
    original = next(wm.parameters()).detach().clone()
    env = DelayedCueEnv(cue_delay=3, horizon=8, obs_dim=8, seed=2)
    env.reset(); env.step(2)
    path = tmp_path / "bundle.pt"
    CheckpointManager.save_bundle(path, {"wm": wm, "actor": actor, "unc": unc}, {"wm": opt},
                                  extra={"env": env.state_dict(), "step": 3})
    with torch.no_grad():
        next(wm.parameters()).add_(100)
    extra = CheckpointManager.load_bundle(path, {"wm": wm, "actor": actor, "unc": unc}, {"wm": opt})
    assert torch.allclose(next(wm.parameters()), original)
    assert extra["step"] == 3
    env2 = DelayedCueEnv(cue_delay=3, horizon=8, obs_dim=8, seed=99).load_state_dict(extra["env"])
    assert env2.t == env.t and env2.cue == env.cue


def test_robust_environment_and_calibration():
    env = RobustDelayedCueEnv(cue_delay=3, horizon=8, obs_dim=8, seed=1,
                              observation_noise=0.1, dropout_prob=0.5)
    obs = env.reset()
    assert obs.shape == (8,)
    assert np.isfinite(obs).all()
    ece = expected_calibration_error([0.9, 0.1], [1, 0], bins=2)
    assert 0 <= ece <= 1
    assert brier_score([1, 0], [1, 0]) == 0.0


def test_callable_reasoner_json_contract():
    fn = lambda prompt: json.dumps([{"name": "find", "arguments": {"object": "cup"}, "constraints": {}}])
    r = CallableReasoner(fn)
    goals = r.decompose("find cup", {})
    assert goals[0].name == "find"
    assert goals[0].arguments["object"] == "cup"
