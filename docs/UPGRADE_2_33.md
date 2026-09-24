# Aether v2.33 — Recursive DREAM-RSI Closure

v2.33 closes the implementation gaps found in the v2.32 audit. The action-time
agent is unchanged. The release upgrades the learning-system / research-OS loop
so the Aether adaptation now follows the important operational structure of
DREAM-RSI rather than only borrowing its replay idea.

## What changed

### 1. Real exact-transition online execution

`execute_training_allocation()` now executes the same
`TrainingAllocationDecision` that historical replay consumes. Collection stops
at the exact requested transition budget, writes a grounded dataset, trains the
normal Aether belief/world/actor stack, hashes the resulting artifacts, and
writes `allocation_execution_receipt.json`.

This removes the previous mismatch between abstract meta-policy decisions and
real campaign execution.

### 2. Complete decision menus

A grounded outcome can now record the complete set of
`TrainingAllocationOption` objects visible before execution. Replay evaluates a
candidate against that historical menu. If the candidate selects a decision
that was offered but never executed, replay does not synthesize an outcome. The
trace becomes unsupported and requires a real online probe.

### 3. Correct worker accounting

A meta-policy may select several training allocations in one round. v2.33 adds
`worker_budget`, which constrains

`sum(decision.parallel_worlds) <= worker_budget`.

This prevents four 4-world allocations from being represented as a 4-worker
batch merely because four meta-decisions were selected.

### 4. Iterative policy development

`DreamRSIOptimizer.propose()` now supports multiple revision rounds. Each round:

1. evaluates the current incumbent on the fixed replay pool;
2. evaluates a bounded declarative neighborhood;
3. records every candidate score, support state, and unsupported-decision count;
4. selects the best supported improvement;
5. uses the selected policy as the starting point for the next revision.

The revision transcript is stored in `proposal.revision_history`. This is the
safe declarative analogue of DREAM-RSI's repeated policy-development feedback
loop. Aether still refuses arbitrary generated Python policy code.

### 5. Recursive replay-pool expansion

`DreamRSIMetaController.ingest_online_world()` appends a newly completed real
online world, rebuilds support statistics, and rebinds the optimizer to the
expanded history. `DreamRSIRecursiveExperiment` closes the mechanical loop:

`real online round -> append history -> dreaming -> paired qualification -> next round`.

Replay remains proposal-only. Promotion still requires paired grounded online
evidence under the unchanged contract.

### 6. Exact-paper objective mode

`ReplayObjective(mode="paper")` implements the paper-form objective structure:

`best_quality - beta1 * attempted_worlds + beta2 * attempted_worlds / rounds`.

Aether's richer embodied-learning objective remains available as
`mode="aether"`. Keeping both avoids silently conflating the source mechanism
with Aether's extension.

### 7. Controlled fixed-vs-DREAM comparison contract

`FixedVsDreamComparisonPlan` requires both arms to begin from the identical
meta-policy and binds the contract, seed set, number of rounds, and transition
budget per round. This supports the same style of controlled comparison used by
DREAM-RSI: fixed exploration versus recursively improved exploration under
matched underlying agent/evaluator/resource conditions.

## New commands

- `awa-v2-dream-rsi-smoke` — fast grounded replay/promotion smoke.
- `awa-v2-dream-rsi-loop-smoke` — tiny real exact-transition online round,
  replay-pool expansion, and dreaming smoke.

## What this release still does not claim

v2.33 proves software closure, not an embodied-RL advantage. The controlled
multi-seed fixed-vs-DREAM experiment has not been run at meaningful scale.
Real ViZDoom, real V-JEPA downstream control, the 25K/100K seven-system
ablation, and external world-model comparisons also remain empirical work.
