import numpy as np
import torch
from awa.memory.replay import EpisodeReplayBuffer, Transition
from awa.training.returns import lambda_returns
from awa.model import WorldModel


def test_sequence_replay_contiguous():
    r = EpisodeReplayBuffer(100)
    for i in range(6):
        r.add(Transition(np.array([i], np.float32), i % 2, float(i), np.array([i+1], np.float32), i == 5))
    assert r.can_sample(2, 3)
    b = r.sample_sequences(2, 3, torch.device('cpu'))
    assert b.observations.shape == (2, 4, 1)
    for seq in b.observations.numpy():
        assert np.all(np.diff(seq[:,0]) == 1)


def test_lambda_return_shape():
    r = torch.ones(4, 3)
    v = torch.zeros(4, 3)
    d = torch.full((4,3), 0.99)
    out = lambda_returns(r, v, d, 0.95, torch.zeros(3))
    assert out.shape == (4,3)
    assert torch.all(out[0] >= out[-1])


def test_slow_state_updates():
    wm = WorldModel(4, 8, 16, 4, 3, 16, 'gru', slow_enabled=True, slow_dim=6, slow_stride=2)
    b = wm.initial_belief(2, torch.device('cpu'))
    out = wm.observe(b, torch.zeros(2,3), torch.randn(2,4), step_index=0)
    assert out.belief.slow.shape == (2,6)
    assert out.belief.vector.shape[-1] == 26
