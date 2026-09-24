# Aether v2.9 — Procedural Game Lab

## Purpose

v2.9 makes reusable-learning claims testable inside the repository. Earlier releases supplied world models, planners, curriculum, memory, skills and transfer metrics; v2.9 adds a deterministic game-like environment and an end-to-end training path tying those systems together.

## Procedural arena

`awa.v2.game.ProceduralArenaEnv` exposes a fixed 32-D observation and four bounded actions: 2-D movement, attack and interact. The twelve curriculum stages progressively introduce obstacles, collection, hidden goals, moving enemies, combat, resource constraints, cover, keys/doors, dynamics shifts, procedural layouts and composition.

The environment is deterministic under `TaskSpec.seed`, supports exact `state_dict/load_state_dict` branching, emits explicit constraint labels, and can render RGB frames for the v2.5 V-JEPA pipeline. Texture and lighting seeds affect pixels but not structured state. Friction/movement parameters are hidden from structured observations by default so dynamics adaptation cannot cheat.

## Data generation

`collect_game_dataset` records chain-consistent transitions with actions, rewards, next observations, terminal flags, four constraint labels, episode IDs and stage IDs. A JSON sidecar records the exact TaskSpecs. `LogicalArenaTeacher` provides coherent but intentionally imperfect demonstrations; `RandomArenaPolicy` is available for exploratory collection.

## Reusable-learning loop

`ReusableGameLoop` feeds live gameplay into `ReusableLearningEngine`: surprise/novelty analysis, structural replay, concept coverage, skill discovery, hindsight relabeling for failures, and resumable checkpoints all operate on actual game episodes.

## End-to-end training

`train_game_stack` runs a compact qualification stack on structured game data:

1. identity frozen-feature cache
2. cached contiguous sequence construction
3. multimodal world-model training and overshooting
4. horizon rollout qualification and uncertainty calibration
5. explicit constraint-risk training
6. TD3+BC actor training
7. actor-only and policy-seeded MPPI closed-loop evaluation
8. versioned world/actor checkpoints and JSON report

The smoke configuration is intentionally tiny and is an execution qualification, not a gameplay-performance claim.

## Commands

```bash
awa-v2-game-lab-generate --out runs/game/game.npz
awa-v2-game-train runs/game/game.npz --out-dir runs/game/train
awa-v2-game-loop-smoke
awa-v2-game-lab-smoke
```

## Next empirical work

Run large procedural datasets with multiple random seeds and compare actor-only, always-on planning, adaptive Value-of-Computation planning, skill reuse, and long-context adaptation. The primary success criterion is held-out task performance improving while planner dependency falls.
