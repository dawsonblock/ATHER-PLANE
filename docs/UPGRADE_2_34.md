# Aether v2.34 — DREAM-RSI Empirical Qualification

v2.34 does not add a new action-time cognitive mechanism. It turns the v2.33
DREAM-RSI adaptation into an executable paired empirical study with hard resource
ceilings, grounded checkpoint lineage, disjoint final generalization evaluation,
and resumable evidence.

## What changed

### 1. Paired fixed-vs-DREAM campaign runner

`FixedVsDreamCampaignRunner` executes independent paired seed replicates. For
every seed and recursive round, the fixed and adaptive arms receive the same:

- initial meta-policy;
- offered allocation menu;
- environment seed;
- maximum real transition budget;
- worker budget;
- action-time Aether source fingerprint;
- evaluator/interface fingerprint.

The fixed arm never changes its allocation policy. The DREAM arm may trial-deploy
a replay-selected policy in the next real round. A replay selection is recorded as
a trial deployment, not as empirical promotion.

### 2. Transition-budget ceiling

`DeclarativeTrainingAllocationPolicy` now has `transition_budget_cap` in addition
to `worker_budget`. The online and replay decision surface therefore enforces both

`sum(parallel_worlds) <= worker_budget`

and

`sum(transition_budget) <= transition_budget_cap`.

This closes a resource-accounting gap where several valid meta-decisions could
otherwise exceed the preregistered real-environment budget.

### 3. Grounded checkpoint lineage

Every executed allocation now records the resulting world and actor checkpoint
paths and hashes. Each arm continues the next recursive round from the best
*grounded* checkpoint produced in its previous round. Alternative allocations in
a round branch from the same parent checkpoint.

This makes the experiment a cumulative learning comparison rather than a series
of unrelated from-scratch fits.

### 4. Development generalization evaluator

The online evaluator can now score a trained allocation on generated heldout and
transfer compositions. DREAM replay therefore need not optimize training-task
success. When enabled, replay quality is heldout success and the transfer term is
measured transfer success.

These development evaluations are part of policy development and are not used as
the final unbiased campaign endpoint.

### 5. Independent final meta-test

After all recursive rounds, each seed's final fixed and DREAM checkpoints are
evaluated on a separate canonical heldout/transfer task set generated from a
base seed never used during replay-policy development. The primary final score is

`0.5 * (heldout_success + transfer_success)`.

The campaign report uses this independent final score for its paired final-gain
statistics. Recursive-round replay scores remain a separate learning-curve
measurement.

### 6. Deterministic paired statistics

The final report contains:

- per-seed final heldout and transfer success;
- per-seed independent final generalization gain;
- mean recursive learning-curve gain;
- 95% paired bootstrap intervals;
- worst-seed regression;
- exact transition use by arm;
- wall-clock use by arm;
- number of DREAM policy changes.

Incomplete seed sets return `INSUFFICIENT_EVIDENCE`.

### 7. Cloud-safe resume

Committed online-round receipts bind the policy id, complete menu, parent
checkpoint hashes, grounded replay-world digest, and output checkpoint hashes.
A rerun verifies those bindings before reuse. Mismatches fail closed rather than
silently resuming a different experiment.

## Canonical command

Plan only:

```bash
awa-v2-dream-rsi-campaign \
  --config configs/v2_34_dream_rsi_empirical.yaml \
  --out-dir runs/v2_34_dream_rsi
```

Execute:

```bash
awa-v2-dream-rsi-campaign \
  --config configs/v2_34_dream_rsi_empirical.yaml \
  --out-dir runs/v2_34_dream_rsi \
  --execute
```

Fast release smoke:

```bash
awa-v2-dream-rsi-campaign-smoke
```

## Canonical experimental budget

The shipped protocol uses five paired seeds and five recursive rounds. Each
round has a maximum 25,000-transition budget and a four-worker ceiling. Menu
items are 6,250-transition allocations, so at most four can execute in a round.
Only training factor signatures from the v2.30 compositional benchmark are
available to the meta-policy.

## Empirical boundary

v2.34 validates the experiment runner and tiny real executions. It does **not**
claim that DREAM-RSI improves embodied RL until the canonical paired campaign is
actually executed. Real ViZDoom and real V-JEPA qualification also remain
separate empirical gates.
