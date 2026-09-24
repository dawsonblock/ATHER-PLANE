import torch
import torch.nn.functional as F
from awa.model import WorldModel
from awa.planning.actor import CategoricalActor
from awa.planning.mpc import CEMPlanner


def test_world_model_shapes():
    wm = WorldModel(16, 32, 64, 8, 3, 64, "gru")
    b = wm.initial_belief(4, torch.device("cpu"))
    obs = torch.randn(4, 16)
    action = F.one_hot(torch.tensor([0,1,2,0]), 3).float()
    out = wm.observe(b, action, obs)
    assert out.belief.deterministic.shape == (4, 64)
    assert out.belief.stochastic.shape == (4, 8)
    assert out.reward.shape == (4, 1)


def test_ssm_world_model_shapes():
    wm = WorldModel(16, 32, 64, 8, 3, 64, "ssm")
    b = wm.initial_belief(2, torch.device("cpu"))
    out = wm.observe(b, torch.zeros(2,3), torch.randn(2,16))
    assert out.belief.vector.shape == (2, 72)


def test_actor_shape():
    actor = CategoricalActor(72, 3, 32)
    logits = actor.logits(torch.randn(5,72))
    assert logits.shape == (5,3)


def test_planner_runs():
    wm = WorldModel(16, 16, 16, 4, 3, 32, "gru")
    b = wm.initial_belief(1, torch.device("cpu"))
    planner = CEMPlanner(wm, 3, horizon=2, candidates=4, elites=2, iterations=1)
    onehot, idx, probs = planner.plan(b)
    assert onehot.shape == (1,3)
    assert 0 <= idx < 3
    assert probs.shape == (2,3)
