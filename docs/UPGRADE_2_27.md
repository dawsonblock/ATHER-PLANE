# Aether v2.27 — Executable Ablation + RNG Resume Closure

v2.27 converts the v2.26 ablation preregistration from labels into concrete runtimes. The release also captures process RNG state in neural checkpoints so resumed stochastic training can continue from the saved random stream rather than merely restoring weights and optimizer tensors.

## Why this release exists

v2.26 could preregister seven configurations but the canonical trainer still executed the full belief/world/actor path. That meant an `actor_only` label was not yet a real actor-only baseline. v2.27 fixes that methodological defect.

The executable ladder is now:

```text
actor_only
  raw observation + goal -> offline actor/critic

belief_actor
  temporal belief reconstruction -> offline actor/critic

world_actor
  temporal belief + learned world model -> actor

world_planner
  world_actor + fixed policy-seeded MPPI

adaptive_compute
  world_actor + real-branch VOC fitting + actor/MPPI/iCEM compute selection

reusable_replay
  adaptive_compute + grounded future-goal hindsight replay

full_curriculum
  reusable_replay + adaptive curriculum data allocation
```

The previous broad `reusable_learning` label was deliberately narrowed to `reusable_replay`. Skill discovery, Engram, sparse MoE, ensemble dynamics, and other experimental families are not credited to the stable ablation ladder until they receive their own executable and independently measurable path.

## New commands

Run a single grounded variant:

```bash
awa-v2-ablation-run \
  --variant actor_only \
  --out-dir runs/ablation_actor_only \
  --device cpu
```

Run the dependency-light miniature that executes every variant:

```bash
awa-v2-ablation-runtime-smoke
```

Generate the immutable 5-seed / 2-split / 25K+100K protocol:

```bash
awa-v2-ablation-protocol \
  --seeds 1701,1702,1703,1704,1705 \
  --milestones 25000,100000 \
  --output runs/v227_ablation_protocol.json
```

## Grounded reusable replay

The `reusable_replay` and `full_curriculum` variants add one-step hindsight rows by relabeling an actually observed future position as a reach-goal target. They never invent a next state. Relabeled rows are terminal one-step training examples with lower structural weight so they can help actor/reconstruction learning without pretending to be contiguous multi-step world-model trajectories.

This is intentionally narrower than Aether's historical reusable-learning umbrella.

## Adaptive compute

`adaptive_compute`, `reusable_replay`, and `full_curriculum` fit a Value-of-Computation model from exact real-environment branches at identical states. The runtime chooses only among planner/budget combinations represented in that fitted data:

```text
actor
policy_mppi / 16 candidates
policy_icem / 32 candidates
```

The evaluation report records planner calls, world-model calls, latency, success, return, and constraint violations.

## RNG continuation

`game_world.pt` now records the stochastic state immediately after world training/calibration and `game_actor.pt` records the state after actor training:

```text
Python random
NumPy RNG
PyTorch CPU RNG
all CUDA RNG states when CUDA is present
```

On resume, weights and optimizer state are loaded first, then the saved RNG stream is restored before the next training updates. Older checkpoints remain loadable; they simply do not provide bitwise stochastic continuation.

A cumulative 25K -> 100K run is still not expected to be bitwise-identical to a fresh 100K-from-scratch run because the data schedule is intentionally different. RNG closure means the resumed continuation itself is reproducible.

## Scientific boundary

v2.27 proves that the ablation systems are distinct executable programs. It does not prove that any one is better. The 25K/100K five-seed study still needs to be executed before deleting cognitive mechanisms.
