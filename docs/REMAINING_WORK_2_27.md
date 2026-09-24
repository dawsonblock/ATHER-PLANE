# Aether v2.27 — Remaining Work

The highest-value work is now empirical rather than architectural.

## 1. Execute the 25K/100K ablation matrix

Run all seven executable variants under the preregistered five seeds and heldout/transfer splits. Compare not only success rate but sample efficiency, wall-clock cost, planner calls, model calls, latency, and constraint violations.

Do not proceed to 1M for a mechanism that is already clearly dominated at 100K unless the learning curves justify the additional spend.

## 2. Matched collector qualification

Hold the downstream learner fixed and collect equal transition budgets with:

```text
teacher
random
coverage
Aether actor
Aether actor + planner
```

Use one frozen source checkpoint for Aether collectors and separate downstream learner seeds. Report downstream heldout/transfer performance per collected transition and per GPU-hour. This is the direct test of whether Aether gathers better data.

## 3. Dedicated planner/VOC utility study

For one frozen checkpoint, compare actor-only against fixed MPPI/iCEM budgets and adaptive VOC allocation. Measure realized return gain from identical snapshots, compute cost, latency, risk, and planner dependence as the agent becomes familiar with a task.

The adaptive-compute layer should remain only if gain after compute cost is positive.

## 4. Compositional transfer benchmark

Create factor-isolated training sets and never-seen combinations for memory, combat, resource scarcity, key/door dependency, moving hazards, and changed dynamics. Report zero-shot success and episodes-to-recovery.

This should decide whether higher-level reusable mechanisms are worth reintroducing into the stable ladder.

## 5. Remove losing experimental families

After the ablation results exist, move or delete stable-package families that do not improve capability or efficiency. Candidates remain:

```text
skills/
training/reusable_engine.py
engram.py
ensemble.py
modular_world.py
training/modes.py
structural replay machinery beyond grounded replay
```

Do not keep them merely because they are interesting.

## 6. Deprecate orphan prototypes

`memory.py`, `hierarchy.py`, `context_replay.py`, and `decision.py` are still outside the canonical runtime spine. Add deprecation warnings, migrate any genuinely reused primitive, and remove them in the next compatibility-breaking release.

## 7. Retire v1 package surface

Move the actor/critic primitives that v2 still imports into neutral core modules, preserve old v1 through release tags, then remove the obsolete v1 console/runtime surface from the current package.

## 8. Consolidate CLI

Introduce a single `awa` command with stable subcommands, keep old aliases for one release, then remove the historical script explosion after automation is migrated.

## 9. External baseline and external environment

Once the surviving procedural configuration is known, compare it against a strong modern world-model baseline under matched data/compute. Then carry the surviving Aether subset into structured ViZDoom and RGB+V-JEPA. Procedural success alone is not evidence of general visual control.
