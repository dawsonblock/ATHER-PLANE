from pathlib import Path
import json

import numpy as np
import torch
from torch import nn

from awa.v2.agent import AetherV2Agent
from awa.v2.datasets import OfflineTransitionDataset, write_synthetic_dataset
from awa.v2.representation import (
    FrozenBackboneProjector,
    CachedRepresentation,
    RepresentationCache,
    build_pretrained_representation,
    module_fingerprint,
    tensor_sha256,
)
from awa.v2.representation_train import GeometryPreservingProjectionTrainer


class CountingBackbone(nn.Module):
    def __init__(self, in_dim=6, out_dim=10):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        return self.linear(x.float())


def test_tensor_hash_is_shape_and_content_stable():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    assert tensor_sha256(x) == tensor_sha256(x.clone())
    assert tensor_sha256(x) != tensor_sha256(x.reshape(3, 2))


def test_frozen_backbone_projection_only_trains_projection():
    backbone = CountingBackbone()
    enc = FrozenBackboneProjector(backbone, 10, 4)
    assert all(not p.requires_grad for p in backbone.parameters())
    assert any(p.requires_grad for p in enc.projection.parameters())
    out = enc(torch.randn(3, 6))
    out.sum().backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert any(p.grad is not None for p in enc.projection.parameters())


def test_cache_hits_avoid_reencoding(tmp_path: Path):
    torch.manual_seed(1)
    backbone = CountingBackbone()
    enc = FrozenBackboneProjector(backbone, 10, 4)
    cached = CachedRepresentation(enc, RepresentationCache(tmp_path))
    x = torch.randn(2, 6)
    y1 = cached(x)
    calls = backbone.calls
    y2 = cached(x)
    assert backbone.calls == calls
    assert torch.allclose(y1, y2)
    assert cached.cache.stats()["entries"] == 2


def test_cache_key_changes_when_backbone_changes(tmp_path: Path):
    torch.manual_seed(2)
    x = torch.randn(1, 6)
    b1, b2 = CountingBackbone(), CountingBackbone()
    with torch.no_grad():
        b2.linear.weight.add_(1.0)
    c = RepresentationCache(tmp_path)
    fp1, fp2 = module_fingerprint(b1), module_fingerprint(b2)
    assert fp1 != fp2
    c.put(x, fp1, torch.ones(1, 10))
    assert c.get(x, fp1) is not None
    assert c.get(x, fp2) is None


def test_offline_dataset_manifest_and_shapes(tmp_path: Path):
    path = write_synthetic_dataset(tmp_path / "d.npz", n=11, obs_dim=7, action_dim=2)
    ds = OfflineTransitionDataset(path)
    assert len(ds) == 11
    assert ds.manifest.observation_shape == (7,)
    assert ds.manifest.action_shape == (2,)
    first = ds[0]
    assert first["observation"].shape == (7,)
    assert first["action"].shape == (2,)
    out = tmp_path / "manifest.json"
    ds.write_manifest(out)
    saved = json.loads(out.read_text())
    assert saved["sha256"] == ds.manifest.sha256


def test_offline_dataset_rejects_bad_contract(tmp_path: Path):
    p = tmp_path / "bad.npz"
    np.savez(p, observations=np.zeros((2, 3)), actions=np.zeros((2, 1)))
    try:
        OfflineTransitionDataset(p)
    except ValueError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("bad dataset was accepted")


def test_geometry_projection_training_updates_projection_only():
    torch.manual_seed(3)
    backbone = CountingBackbone(in_dim=6, out_dim=10)
    enc = FrozenBackboneProjector(backbone, 10, 5)
    features = enc.encode_features(torch.randn(12, 6))
    before = [p.detach().clone() for p in enc.projection.parameters()]
    trainer = GeometryPreservingProjectionTrainer(enc.projection, lr=1e-2)
    metrics = trainer.step(features)
    after = list(enc.projection.parameters())
    assert metrics.loss >= 0
    assert any(not torch.allclose(a, b) for a, b in zip(before, after))
    assert all(p.grad is None for p in backbone.parameters())


def test_agent_accepts_injected_representation():
    torch.manual_seed(4)
    backbone = CountingBackbone(in_dim=6, out_dim=10)
    enc = FrozenBackboneProjector(backbone, 10, 8)
    agent = AetherV2Agent(6, 2, latent_dim=8, local_dim=8, global_dim=8, hidden=16, low=[-1,-1], high=[1,1], encoder=enc)
    state = agent.initial_state(2, torch.device("cpu"))
    state, belief = agent.observe(state, torch.randn(2, 6), torch.zeros(2, 2), 0)
    assert belief.shape == (2, 16)


def test_module_provider_factory():
    backbone = CountingBackbone(in_dim=6, out_dim=10)
    enc = build_pretrained_representation("module", 7, backbone=backbone, feature_dim=10)
    assert enc.output_dim == 7
    assert enc(torch.randn(2, 6)).shape == (2, 7)


def test_precompute_dataset_cache_indexes_current_and_next(tmp_path: Path):
    from awa.v2.datasets import precompute_representation_cache
    path = write_synthetic_dataset(tmp_path / "cache_ds.npz", n=5, obs_dim=6, action_dim=2)
    ds = OfflineTransitionDataset(path)
    backbone = CountingBackbone(6, 10)
    enc = FrozenBackboneProjector(backbone, 10, 4)
    cache = RepresentationCache(tmp_path / "cache", namespace=ds.manifest.sha256[:16])
    index_path = tmp_path / "index.json"
    index = precompute_representation_cache(ds, enc, cache, index_path)
    assert index["transitions"] == 5
    assert len(index["observation_keys"]) == 5
    assert len(index["next_observation_keys"]) == 5
    assert index_path.exists()
    # Repeating the exact precompute should be pure cache hits.
    again = precompute_representation_cache(ds, enc, cache)
    assert again["writes"] == 0
    assert again["hits"] == 10
