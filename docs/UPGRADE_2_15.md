# Aether v2.15 — ViZDoom + V-JEPA 2 Game Integration

v2.15 adds the first direct game integration selected for Aether's visual-control qualification: ViZDoom. The purpose is not to make Doom the final target; it is to give Aether a fast, reproducible first-person environment with RGB frames, structured game variables, deterministic seeds, and save/load support.

## Architecture

```text
ViZDoom
  ├─ RGB screen_buffer ──> 64-frame causal clip ──> frozen V-JEPA 2 ──> cached visual features ─┐
  ├─ telemetry/game variables ─────────────────────────────────────────────────────────────────┤
  └─ scenario goal ────────────────────────────────────────────────────────────────────────────┤
                                                                                               v
                                                                                     Aether temporal belief
                                                                                               v
                                                                                  world / actor / skills / planner
                                                                                               v
                                                                                       4-D Doom action ABI
```

Aether's ViZDoom action ABI is stable and hybrid:

1. turn left/right, continuous `[-1, 1]`
2. move forward/backward, continuous `[-1, 1]`
3. attack, categorical
4. use/interact, categorical

The direct adapter maps the continuous controls to ViZDoom delta buttons when configured and maps attack/use to binary buttons.

## Recommended first experiment

Start with `my_way_home` because it tests visual navigation and memory without combat complexity. Use the structured track first, then the pixel/V-JEPA track with the same seeds.

Recommended progression:

1. `basic`
2. `my_way_home`
3. `health_gathering`
4. `defend_the_center`
5. `predict_position`
6. `take_cover`
7. `deadly_corridor`
8. `deathmatch`

## Installation

Base Aether remains dependency-light. ViZDoom and V-JEPA are optional:

```bash
pip install -e ".[dev,doom-vjepa]"
```

The released V-JEPA weights are not bundled in Aether. Download/cache them separately or pass `--allow-download` to the V-JEPA cache CLI.

## Real ViZDoom checks

```bash
awa-v2-vizdoom-check --scenario my_way_home --track structured
awa-v2-vizdoom-check --scenario my_way_home --track pixel
```

Collect RGB gameplay:

```bash
awa-v2-vizdoom-collect runs/doom-pixel.npz \
  --scenario my_way_home \
  --track pixel \
  --episodes 32
```

Precompute frozen V-JEPA 2 features:

```bash
awa-v2-vizdoom-vjepa-cache runs/doom-pixel.npz \
  --cache-dir runs/doom-vjepa-cache \
  --model facebook/vjepa2-vitl-fpc64-256 \
  --frames 64
```

The cache is content-addressed and binds the raw causal clip, telemetry, encoder fingerprint, dataset SHA-256, and clip contract.

## V-JEPA contract

v2.15 defaults to `facebook/vjepa2-vitl-fpc64-256`. The integration treats 64 frames as a checkpoint contract and fails closed when a caller supplies a different clip length for a backbone declaring 64 frames.

The Hugging Face adapter now prefers the model's `get_vision_features()` method when available and falls back to `forward()` for compatible injected/test models.

## Structured vs pixel qualification

Do not combine these benchmark scores:

```text
structured: ViZDoom game variables -> Aether
pixel:      RGB -> V-JEPA 2 -> Aether
```

Use identical scenario/seed sets where possible. The difference between the two tracks estimates the cost of perception rather than conflating it with control/world-model failures.

## Snapshot boundary

ViZDoom's native save/load restores world state, but its documented load behavior does not rewind all episode accounting such as tic/total-reward counters. The direct adapter therefore labels its snapshot semantics `world_state_only`. It is useful for controlled branches but v2.15 does not claim mathematically exact counterfactual episode-time replay.

## What v2.15 does not claim

- No V-JEPA weights are shipped inside the ZIP.
- No real ViZDoom training run is included.
- The built-in explorer policy is for coverage/data collection, not an expert demonstration policy.
- The real ViZDoom executable/package is optional and was not required for the dependency-free release regression tests.
