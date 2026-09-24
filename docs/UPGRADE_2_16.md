# Aether v2.16 — ViZDoom Training Closure

## Purpose

v2.15 proved the ViZDoom environment and V-JEPA cache adapters. v2.16 closes the gap between those adapters and Aether's newer temporal-belief/control stack.

## Unified training path

```text
ViZDoom structured telemetry
        OR
RGB -> frozen V-JEPA clip features + telemetry
                 |
                 v
      goal-conditioned temporal belief
      (local GRU + global context)
                 |
                 v
       stochastic world model
       reward / continuation / risk / value
                 |
                 v
         hybrid actor/critic
    turn + forward | attack + use
                 |
                 v
      risk-aware hybrid MPPI
                 |
                 v
        fail-closed action guard
                 |
                 v
              ViZDoom
```

## Pixel workflow

1. Collect pixel transitions with `awa-v2-vizdoom-collect`.
2. Build causal frozen V-JEPA features with `awa-v2-vizdoom-vjepa-cache`. Cache misses are processed in batches.
3. Materialize the cache into the temporal-training ABI with `awa-v2-vizdoom-materialize`.
4. Train world/belief/risk/hybrid actor with `awa-v2-vizdoom-train --track pixel`.
5. Run the same frozen video backbone online through `ViZDoomRuntimeController`.

## Structured workflow

Structured ViZDoom datasets can be passed directly to `awa-v2-vizdoom-train --track structured`. This creates a perception-free control baseline using the exact same downstream temporal/world/actor architecture as the pixel track.

## Hybrid actions

The public four-float ABI remains `[turn, forward, attack, use]`, but v2.16 treats the last two controls categorically. Offline actor training uses BCE targets, online execution emits only `-1/+1`, and the hybrid MPPI planner never evaluates fractional attack/use actions in the world model.

## Scenario semantics

Success and risk are scenario-aware. Navigation/survival/combat scenarios no longer share one generic `final reward > 0` rule, and ammunition depletion is only treated as a risk signal for combat-oriented scenarios.

## Campaign

`awa-v2-vizdoom-campaign` is plan-only by default. The shipped configuration describes cumulative 25k, 100k, 250k, 500k and 1M transition milestones. After a stable checkpoint exists, a configurable fraction of collection decisions can come from risk-aware planning; those actions are recorded and become actor training targets in the next round.

## Qualification boundary

The dependency-free test suite uses DoomGame-compatible fakes and toy video backbones. A real ViZDoom executable and real Meta V-JEPA weights remain optional external dependencies and are not embedded in the release archive.
