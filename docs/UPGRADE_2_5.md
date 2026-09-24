# Aether v2.5 — pretrained game-video perception

v2.5 adds a first-class path from raw game frames to the existing v2 world-model and planner stack.

## What changed

- `VJEPA2HFBackbone` uses Hugging Face `AutoVideoProcessor` + `AutoModel` for frozen V-JEPA 2 features.
- `VJEPA21TorchHubBackbone` supports Meta's V-JEPA 2.1 PyTorch-Hub backbones, local-first by default.
- `EpisodeClipBuilder` builds causal clips and never crosses episode boundaries.
- `precompute_game_video_cache` caches frozen video features plus optional structured telemetry in the same content-addressed representation store used by v2.2 world-model training.
- `StreamingGameFeatureEncoder` mirrors the offline clip semantics for live game frames.
- `AetherV2Agent.observe_latent` lets expensive video inference run outside the recurrent agent while the exact projector from a qualified world checkpoint remains authoritative.
- game transition datasets may include `telemetry` and `next_telemetry` arrays.
- new commands: `awa-v2-game-cache` and `awa-v2-game-repr-smoke`.

## Supported dataset contract

An `.npz` game dataset contains:

- `observations`: RGB frames, `N x H x W x C` or `N x C x H x W`
- `actions`
- `rewards`
- `next_observations`: exact next RGB frame
- `dones`
- optional `telemetry` and `next_telemetry`

The cache index stores causal video feature vectors and is intentionally compatible with `awa-v2-world-qualify`.

## Recommended first real game stack

1. V-JEPA 2 ViT-L frozen encoder.
2. 64-frame causal clips initially, then benchmark shorter/sparser clips for latency.
3. concatenate low-dimensional game telemetry to the frozen visual feature before Aether's trainable projection.
4. train the v2.2 world model from cached features.
5. train the v2.4 actor in the exact saved latent space.
6. compare actor-only vs policy-seeded MPPI/iCEM at equal world-model calls.
7. train Value of Computation from real counterfactual planner benefit.

## V-JEPA 2.1

Meta's V-JEPA 2.1 is newer and targets higher-quality temporally consistent dense features. Aether supports it through a local PyTorch-Hub checkout. The base package does not download external repositories automatically.

## macOS

The Aether game path operates on in-memory frame tensors and therefore does not require Meta's `decord` video-file loader. This avoids the official repository's `decord`-on-macOS limitation for live game capture. You still need a compatible PyTorch/Transformers setup for the selected backbone.

## What this release does not claim

- No pretrained V-JEPA weights are bundled in the ZIP.
- No commercial game is automatically controlled.
- No game-playing performance result is claimed from the synthetic video smoke dataset.
- V-JEPA 2/2.1 remains a representation backbone; Aether's game-specific dynamics, actor, risk model and VOC still require training on game interaction data.
