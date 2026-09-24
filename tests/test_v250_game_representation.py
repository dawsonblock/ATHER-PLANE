from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from awa.v2.agent import AetherV2Agent
from awa.v2.datasets import OfflineTransitionDataset, write_synthetic_game_dataset
from awa.v2.representation import RepresentationCache
from awa.v2.video_representation import (
    EpisodeClipBuilder,
    StreamingGameFeatureEncoder,
    ToyVideoBackbone,
    VideoClipSpec,
    VJEPA2HFBackbone,
    VJEPA21TorchHubBackbone,
    build_vjepa_game_representation,
    precompute_game_video_cache,
)
from awa.v2.world_training import CachedFeatureSequenceDataset


class FakeProcessor:
    def __call__(self, clip, return_tensors="pt", **kwargs):
        x = torch.as_tensor(clip).float()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        return {"pixel_values_videos": x}


class FakeVJEPA(nn.Module):
    def __init__(self, hidden=8, frames=4):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.config = SimpleNamespace(hidden_size=hidden, frames_per_clip=frames, _commit_hash="fake-commit")
        self.hidden = hidden

    def forward(self, pixel_values_videos=None, **kwargs):
        x = pixel_values_videos.float()
        # B,T,C,H,W -> one token per frame, then widen to hidden size.
        per_frame = x.mean(dim=(2, 3, 4), keepdim=False).unsqueeze(-1)
        tokens = per_frame.repeat(1, 1, self.hidden)
        return SimpleNamespace(last_hidden_state=tokens)


class FakeHubModel(nn.Module):
    def __init__(self, hidden=7):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.embed_dim = hidden

    def forward(self, x):
        # B,T,C,H,W -> B,T,D
        base = x.float().mean(dim=(2, 3, 4)).unsqueeze(-1)
        return base.repeat(1, 1, self.embed_dim)


def test_episode_clip_builder_never_crosses_done_boundary(tmp_path: Path):
    p = write_synthetic_game_dataset(tmp_path / "g.npz", episodes=2, episode_length=3, height=8, width=8)
    ds = OfflineTransitionDataset(p)
    spec = VideoClipSpec(frames_per_clip=4)
    b = EpisodeClipBuilder(ds.arrays["observations"], ds.arrays["next_observations"], ds.arrays["dones"], spec)
    # First observation of second episode is transition 3; all left padding must repeat frame 3.
    clip = b.clip(3)
    first = torch.as_tensor(ds.arrays["observations"][3]).permute(2, 0, 1)
    assert clip.shape == (4, 3, 8, 8)
    assert all(torch.equal(clip[i], first) for i in range(4))


def test_hf_vjepa_adapter_freezes_model_and_pools_tokens():
    model = FakeVJEPA(hidden=8, frames=4)
    bb = VJEPA2HFBackbone("fake", model=model, processor=FakeProcessor())
    clip = torch.randint(0, 255, (2, 4, 3, 8, 8), dtype=torch.uint8)
    out = bb(clip)
    assert out.shape == (2, 8)
    assert bb.frames_per_clip == 4
    assert all(not p.requires_grad for p in model.parameters())
    assert len(bb.fingerprint()) == 64


def test_vjepa21_local_adapter_with_injected_model():
    model = FakeHubModel(hidden=7)
    bb = VJEPA21TorchHubBackbone(
        ".", model=model, preprocessor=lambda x: [x.float().permute(1,0,2,3)], feature_dim=7
    )
    out = bb(torch.ones(1, 4, 3, 6, 6))
    assert out.shape == (1, 7)
    assert len(bb.fingerprint()) == 64


def test_game_cache_fuses_telemetry_and_is_reusable(tmp_path: Path):
    p = write_synthetic_game_dataset(tmp_path / "g.npz", episodes=2, episode_length=5, height=8, width=8, telemetry_dim=4)
    ds = OfflineTransitionDataset(p)
    bb = ToyVideoBackbone(feature_dim=10)
    spec = VideoClipSpec(frames_per_clip=4)
    cache = RepresentationCache(tmp_path / "cache", namespace=ds.manifest.sha256[:16])
    index = precompute_game_video_cache(ds, bb, cache, clip_spec=spec, index_path=tmp_path / "index.json")
    assert index["visual_feature_dim"] == 10
    assert index["telemetry_dim"] == 4
    assert index["feature_dim"] == 14
    assert len(index["observation_keys"]) == len(ds)
    again = precompute_game_video_cache(ds, bb, cache, clip_spec=spec)
    assert again["writes"] == 0
    assert again["hits"] == len(ds) * 2


def test_game_cache_is_directly_compatible_with_world_sequence_loader(tmp_path: Path):
    p = write_synthetic_game_dataset(tmp_path / "g.npz", episodes=3, episode_length=6, height=8, width=8)
    ds = OfflineTransitionDataset(p)
    bb = ToyVideoBackbone(feature_dim=8)
    cache = RepresentationCache(tmp_path / "cache", namespace=ds.manifest.sha256[:16])
    index = precompute_game_video_cache(ds, bb, cache, clip_spec=VideoClipSpec(frames_per_clip=3))
    seq = CachedFeatureSequenceDataset(ds, cache, index, sequence_length=3)
    row = seq[0]
    assert row["features"].shape == (3, 12)  # 8 visual + 4 telemetry
    assert row["next_features"].shape == (3, 12)


def test_streaming_game_encoder_matches_offline_cache(tmp_path: Path):
    p = write_synthetic_game_dataset(tmp_path / "g.npz", episodes=1, episode_length=6, height=8, width=8)
    ds = OfflineTransitionDataset(p)
    bb = ToyVideoBackbone(feature_dim=9)
    spec = VideoClipSpec(frames_per_clip=4)
    cache = RepresentationCache(tmp_path / "cache", namespace=ds.manifest.sha256[:16])
    index = precompute_game_video_cache(ds, bb, cache, clip_spec=spec)
    stream = StreamingGameFeatureEncoder(bb, spec, telemetry_dim=4)
    for i in range(6):
        online = stream.step(ds.arrays["observations"][i], ds.arrays["telemetry"][i]).cpu()
    offline = cache.get_by_key(index["observation_keys"][5])
    assert offline is not None
    assert torch.allclose(online, offline, atol=1e-7)


def test_agent_can_advance_from_external_video_latent():
    agent = AetherV2Agent(5, 2, latent_dim=6, local_dim=8, global_dim=8, hidden=16, low=[-1,-1], high=[1,1])
    state = agent.initial_state(1, torch.device("cpu"))
    z = torch.randn(1, 6)
    state, belief = agent.observe_latent(state, z, torch.zeros(1, 2), 0)
    assert belief.shape == (1, 16)


def test_vjepa_game_factory_supports_injected_hf_backbone():
    model = FakeVJEPA(hidden=8, frames=4)
    enc = build_vjepa_game_representation(
        "vjepa2-hf", 5, model_name_or_path="fake", backbone=model, processor=FakeProcessor()
    )
    out = enc(torch.randint(0,255,(2,4,3,8,8),dtype=torch.uint8))
    assert out.shape == (2,5)
