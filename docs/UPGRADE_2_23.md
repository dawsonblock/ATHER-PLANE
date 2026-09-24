# Aether v2.23 — Train Once, Evaluate Many

v2.23 fixes a concrete empirical and compute problem in the v2.22 native evolving campaign: each preregistered held-out/transfer cell previously invoked the combined training+evaluation routine independently. For the same policy, seed and transition milestone, that meant the same deterministic training dataset could be recollected and the same model retrained once per evaluation split.

v2.23 separates **training identity** from **evaluation identity**.

```text
policy + seed + milestone + training config
                  |
                  v
       content-addressed training identity
                  |
          collect + train once
                  |
                  v
       immutable training artifact
       /          |            \
 dataset      world ckpt     actor ckpt
                  |
          +-------+-------+
          |               |
          v               v
      heldout eval     transfer eval
          |               |
          +-------+-------+
                  v
         separate evidence cells
```

## Main changes

### Content-addressed training artifact cache

`NativeProceduralCampaignRunner` now derives a training identity from:

- exact Aether source fingerprint
- runtime dependency fingerprint
- declarative exploration policy
- task family
- seed
- transition milestone
- training-only runner configuration

Qualification-only settings such as heldout/transfer split, evaluation task count and evaluation seed offset do not affect the training identity. A completed artifact is committed only after the merged dataset, collection receipt, world checkpoint, actor checkpoint and training report all exist and their hashes are recorded.

The cache fails closed on partial artifacts or hash mismatches.

### Split-isolated evaluation

Training now receives only a deterministic **training-diagnostic** task namespace. Heldout and transfer tasks are evaluated later from the frozen checkpoints through `load_procedural_game_stack()` and `evaluate_procedural_actor()`.

This means qualification scenarios are no longer passed through the training routine at all.

### Stable logical cost accounting

Cache reuse should not make empirical records depend on execution order. Each cell therefore records both:

- `wall_clock_seconds`: logical training + evaluation cost for the cell
- `physical_wall_clock_seconds`: work actually performed in that invocation
- `training_wall_clock_seconds`
- `evaluation_wall_clock_seconds`
- `training_cache_reused`

`normalized_compute_cost` continues to use the logical cell cost so heldout-first versus transfer-first ordering does not change replay-policy scoring.

### Public procedural checkpoint runtime

New module: `awa.v2.game.procedural_runtime`.

It provides:

- `load_procedural_game_stack()`
- `evaluate_procedural_actor()`
- `LoadedProceduralGameStack`
- `ProceduralActorEvaluation`

The loader validates checkpoint formats and dimensions and places all modules in evaluation mode with gradients disabled.

## Run

Plan first:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_23_train_once_evaluate_many.yaml \
  --out-dir runs/v2_23_meta
```

Bootstrap and execute explicitly:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_23_train_once_evaluate_many.yaml \
  --out-dir runs/v2_23_meta \
  --bootstrap-active \
  --execute
```

The native smoke now executes both heldout and transfer cells for the same policy/seed/milestone and verifies that they share the exact checkpoint and training dataset while retaining disjoint evaluation scenarios:

```bash
awa-v2-native-campaign-smoke
```

## Research boundary

v2.23 does not add a new intelligence mechanism. It makes the existing closed loop cheaper and empirically cleaner. Replay/model simulation still cannot promote itself, arbitrary source rewriting remains outside the trusted loop, and the native procedural runner still does not establish ViZDoom/V-JEPA capability.
