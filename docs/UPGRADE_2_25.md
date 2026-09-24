# Aether v2.25 — Reduction and Cumulative Empirical Closure

v2.25 is a reduction release. It does not add another intelligence mechanism. It removes work that the v2.24 canonical campaign was performing without changing the empirical question.

## What changed

### 1. Cumulative milestone training

The canonical native learning curve remains:

```text
25K → 100K → 250K → 500K → 1M
```

v2.24 treated each milestone as a separate training artifact built from zero. Per seed that meant:

```text
25K + 100K + 250K + 500K + 1M = 1.875M transition-usages
```

v2.25 builds a grounded lineage:

```text
25K checkpoint
  ↓ +75K new transitions
100K checkpoint
  ↓ +150K
250K checkpoint
  ↓ +250K
500K checkpoint
  ↓ +500K
1M checkpoint
```

Only the newly collected delta is used for the next warm-start training step. The cumulative dataset is still materialized and hashed for provenance. For the five-seed canonical campaign this reduces newly collected/trained transitions from 9.375M to 5M, a 46.7% reduction.

Each artifact records:

- cumulative dataset SHA-256,
- delta dataset SHA-256,
- parent training identity,
- parent checkpoint SHA-256,
- incremental transitions,
- cumulative transitions,
- incremental wall-clock training cost,
- cumulative logical training cost.

A corrupt matching parent lineage fails closed.

### 2. Shared campaign-level training cache

Training artifacts are now shared across `bootstrap/` and later `iterations/` under one campaign-level `_training_cache`. This prevents an unchanged active baseline from being retrained simply because a new meta-iteration began.

### 3. Execution-aware meta-policy search

The native procedural runner converts the meta-policy into integer curriculum stage counts. v2.24 could mutate 17 policy variants even when most variants produced the same real stage allocation.

v2.25 adds an execution signature and deduplicates candidate policies before replay/online qualification. The native default mutator changes only the four axes that can alter real curriculum allocation:

```text
quality_weight
information_weight
novelty_weight
transfer_weight
```

Replay-only knobs remain fixed for the native curriculum experiment rather than being presented as different training policies.

The v2.25 canonical config also uses `tasks_per_batch: 24` and `mutation.step: 0.20` to make the integer allocation less quantized while still deduplicating exact execution equivalents.

### 4. Planner diagnostics removed from routine native training

Canonical heldout/transfer qualification evaluates the frozen actor. v2.24 nevertheless ran planner rollouts during every training artifact.

v2.25 adds `run_planner_diagnostics`. The native canonical campaign sets it to `false`. Planner qualification remains available as a separate experiment and is not confused with actor learning evidence.

### 5. Warm-startable training stack

`train_game_stack()` now accepts:

```text
resume_world_checkpoint
resume_actor_checkpoint
run_planner_diagnostics
```

The world/belief model and offline actor/critic are restored with architecture checks before incremental training. The actor path restores target networks and optimizer state.

### 6. Lazy package exports

`awa.v2` and `awa.v2.game` keep their public convenience symbols but resolve them only when used. A plain `import awa.v2` no longer eagerly imports the game bridge, ViZDoom/V-JEPA adapters, scaling stack, campaign stack, skill stack and other unrelated modules.

Measured in the release environment:

```text
v2.24 import awa.v2: ~2.97 s, ~261 MB max RSS
v2.25 import awa.v2: ~0.77 s, ~92 MB max RSS
```

This is import-surface reduction, not a claim that training itself uses 169 MB less memory once PyTorch/model code is loaded.

## Canonical command

Plan first:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_25_reduction_empirical_closure.yaml \
  --out-dir runs/v2_25_meta
```

Bootstrap and execute:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_25_reduction_empirical_closure.yaml \
  --out-dir runs/v2_25_meta \
  --bootstrap-active \
  --execute
```

Continue later using the same output directory:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_25_reduction_empirical_closure.yaml \
  --out-dir runs/v2_25_meta \
  --execute
```

## Research boundary

v2.25 does not claim that cumulative warm-start training is superior to independent-from-zero milestone training. It is the canonical cost-efficient learning-curve protocol. If the scientific question specifically requires independent models at every milestone, set:

```yaml
runner:
  config:
    cumulative_milestone_training: false
```

The five-seed 1M campaign is still not claimed as executed by release validation.
