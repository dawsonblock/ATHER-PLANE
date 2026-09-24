# Aether v2.26 — Remaining Work and How to Add It

v2.26 closes autonomous native collection and optimizer-complete cumulative world-model resume. The remaining work is now primarily empirical isolation rather than additional architecture.

## Priority 1 — Make all seven ablation systems executable

The preregistered matrix now binds immutable configuration hashes, but the canonical trainer still implements the full belief/world/actor stack. Add explicit execution adapters for:

```text
actor_only
belief_actor
world_actor
world_planner
adaptive_compute
reusable_learning
full
```

Do this with one versioned `SystemVariant` contract passed into training/evaluation. Do not scatter boolean flags across unrelated modules. Every variant must share the same seed, task generator, transition budget, evaluator and metric definitions.

The hard part is `actor_only`: it needs a stable raw state contract, probably `[observation, goal]`, rather than silently using the learned temporal belief. `belief_actor` should train the temporal representation without action-conditioned world-model planning. Those two baselines need real separate runtimes, not labels on the full checkpoint.

## Priority 2 — Run matched collector comparisons

Now that collection modes exist, run the same downstream training pipeline from equal transition budgets collected by:

```text
teacher
random
coverage
Aether actor
Aether actor + planner
```

Use one frozen source checkpoint for the Aether collectors and disjoint downstream learner seeds. Report downstream heldout/transfer performance per collected transition and per GPU-hour. This distinguishes "Aether explores better" from "Aether merely trains well on teacher data."

## Priority 3 — Save deterministic RNG lineage for exact resume experiments

Optimizer state is now restored, but exact bitwise continuation through stochastic Transformer dropout also depends on RNG state. Add checkpoint fields for:

```text
torch CPU RNG
CUDA RNG states when present
NumPy RNG
DataLoader generator state / deterministic sample order
```

This is required only if the scientific claim needs numerical equivalence between uninterrupted and resumed training. Normal statistical equivalence does not require bitwise identity.

## Priority 4 — Independent planner/VOC qualification

The canonical campaign still qualifies the frozen actor. Add a separate fixed-checkpoint benchmark that compares:

```text
actor
fixed planner budgets
adaptive VOC-selected planner budget
```

Record success, return, constraint violations, planner calls, world-model calls, latency and actual realized planner gain. The adaptive compute system should survive only if it improves task utility after compute cost.

## Priority 5 — Compositional transfer benchmark

Create factor-isolated train sets and unseen combinations from:

```text
memory
combat
resource scarcity
key/door dependency
moving hazards
changed dynamics
```

Report zero-shot success and episodes-to-recovery. This is the main test for reusable skills and higher-level structure.

## Priority 6 — Remove mechanisms that lose ablations

After 25K/100K multi-seed ablations, move losing experimental families out of the stable runtime or delete them from the current branch. Candidates include:

```text
skills/
training/reusable_engine.py
engram.py
ensemble.py
modular_world.py
training/modes.py
structural replay machinery
```

A component should not remain in the stable core merely because it is interesting.

## Priority 7 — Deprecate orphan compatibility prototypes

The older `memory.py`, `hierarchy.py`, `context_replay.py`, and `decision.py` modules remain outside the canonical runtime spine. Add deprecation warnings now, move any still-used primitive to neutral core modules, then remove them after one compatibility release.

## Priority 8 — Retire v1 runtime surface

Move the actor/critic primitives still imported by v2 into `awa.core`, preserve v1 through release tags, then remove old console/runtime surface from the active package.

## Priority 9 — Consolidate CLI

The package still carries many historical entry points. Add one `awa` command with subcommands, keep compatibility shims for one release, then remove old aliases after canonical docs and automation stop using them.

## Priority 10 — External validation

Only after the reduced procedural stack wins should the surviving system be carried into:

```text
ViZDoom structured telemetry
ViZDoom RGB + frozen V-JEPA features
```

The procedural environment is an instrumented research world, not proof of general visual control.
