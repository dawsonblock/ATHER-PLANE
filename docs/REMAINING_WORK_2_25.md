# Aether v2.25 — Remaining Work and How to Add It

This document separates work that is still required for a strong empirical result from speculative features that should not be added until evidence justifies them.

## Priority 1 — Run the ablation matrix

The largest remaining uncertainty is architectural, not software plumbing.

Run the same seeds, tasks, milestones and evaluation splits for:

1. actor only,
2. belief + actor,
3. belief + world model + actor,
4. belief + world model + planner,
5. planner + adaptive compute,
6. reusable-learning/skills enabled,
7. adaptive curriculum enabled.

Add this by extending the preregistered experiment protocol with an `ablation_id`/system configuration hash, not by changing metrics after results exist. Keep dataset budgets and evaluator fixed. Remove any component that does not improve transfer/sample efficiency enough to justify its compute and maintenance burden.

## Priority 2 — Replace teacher-only collection with autonomous collection comparisons

The native procedural collector currently uses `LogicalArenaTeacher`. That is useful for controlled offline learning, but it does not prove autonomous exploration.

Add three matched collection modes:

```text
teacher
random/coverage explorer
Aether actor + planner exploration
```

All three must write the same dataset ABI and consume the same transition budget. Qualification should then compare downstream learning and transfer. Only after Aether-controlled collection wins should the outer loop be described as learned environment exploration rather than adaptive curriculum allocation.

## Priority 3 — Make cumulative training optimizer-complete for the world model

The actor checkpoint already carries optimizer state and v2.25 restores it. The belief/world checkpoint currently warm-starts model weights but recreates the AdamW optimizer at each milestone.

Add `trainer_state` to `game_world.pt` containing:

```text
optimizer state
GradScaler state when FP16 is active
global update count
```

On resume, validate architecture and precision, restore those states, and add a regression test that uninterrupted 100K training and 25K→100K resumed training produce numerically equivalent updates under a deterministic miniature setup.

## Priority 4 — Independent planner/VOC qualification

Routine native training no longer pays for planner diagnostics. The planner and adaptive-compute controller still need their own qualification campaign.

Add a fixed-checkpoint benchmark that records:

```text
actor success
planner success
planner gain
planner calls
world-model calls
latency
risk violations
```

Then fit/evaluate VOC on held-out decision states. Require planner usage to decrease with familiarity while task performance stays stable. Do not promote adaptive compute based only on predicted gain.

## Priority 5 — Compositional transfer benchmark

Current heldout/transfer splits are useful but do not fully isolate compositional reuse.

Create a preregistered factor set such as:

```text
memory
combat
resource scarcity
key/door dependency
moving hazards
changed dynamics
```

Train on factors separately and evaluate unseen combinations. Report zero-shot performance and episodes-to-recovery. This is the key test for whether skill/reusable-learning machinery is actually learning transferable structure.

## Priority 6 — Move experimental families out of the stable core after ablation

The following remain in the source tree because v2.25 does not yet have empirical grounds to delete them:

```text
skills/
training/reusable_engine.py
engram.py
ensemble.py
modular_world.py
training/modes.py
structural replay machinery
```

After ablation, move losing mechanisms to `awa.experimental.*` or archive them with their release tag. Do not delete them before the comparison data exists.

## Priority 7 — Retire orphan prototypes and legacy v1 cleanly

Four older v2 prototype modules are not on the canonical runtime spine:

```text
memory.py
hierarchy.py
context_replay.py
decision.py
```

They still have regression tests and therefore remain compatibility APIs in v2.25. The clean removal path is:

1. mark them deprecated for one release,
2. migrate any still-useful primitive into `awa.core`,
3. remove their console/docs/test surface,
4. archive behavior in the old release tag.

Do the same for v1 after moving the actor/critic primitives still reused by v2 into a neutral core module.

## Priority 8 — Collapse the CLI only after script compatibility is inventoried

The project still exposes many historical console entry points. Replacing them immediately could break automation without improving training quality.

Build one `awa` command with subcommands, map old entry points to thin compatibility shims for one release, log deprecation, then remove the aliases after verifying no canonical docs/configs use them.

## Priority 9 — ViZDoom/V-JEPA external validation

The procedural world is not enough to establish generality. After the reduced procedural stack wins its ablations, run the winning subset on:

```text
ViZDoom structured telemetry
ViZDoom RGB + frozen V-JEPA features
```

Keep structured and pixel tracks separate. Compare sample efficiency, transfer, planner dependence and wall-clock cost. Do not assume procedural-world gains transfer to visual control.

## Priority 10 — Only then consider new architecture

Do not add another memory system, planner, MoE, recursive code-rewriter or multi-agent layer before the above results exist. New architecture should be driven by a measured failure mode from the ablation/transfer evidence, not by novelty.
