# Aether v2.26 — Autonomous Collection + Ablation Preregistration

v2.26 continues the v2.25 reduction direction. It does not add a new cognitive subsystem. It closes three empirical-method gaps that blocked a serious test of whether Aether actually learns better.

## 1. Native collection can now become Aether-controlled

v2.25 still used `LogicalArenaTeacher` for native procedural data collection. That meant a successful campaign could establish adaptive curriculum allocation, but not autonomous exploration.

v2.26 adds three non-teacher collection families using the same procedural dataset ABI:

```text
teacher
random
coverage
Aether actor
Aether actor + bounded MPPI planner
```

The canonical v2.26 campaign uses:

```text
first milestone: coverage bootstrap
later milestones: frozen Aether actor + planner
```

The bootstrap distinction is mandatory. Aether cannot claim autonomous collection before a trained checkpoint exists. Every collection batch records both the requested and actual collection mode.

The frozen Aether collector loads the exact grounded parent checkpoint, maintains temporal belief across the episode, and can invoke a bounded policy-seeded MPPI search at a fixed stride. The collector checkpoint is never updated during collection.

## 2. Cumulative world-model training now resumes optimizer state

v2.25 restored world/encoder weights at larger milestones but recreated AdamW. v2.26 stores and restores:

```text
world/encoder parameters
AdamW optimizer state
GradScaler state when CUDA FP16 is active
global world-model update count
```

The state lives in `game_world.pt` under `trainer_state` using the versioned format:

```text
awa-v2.26-game-world-trainer-state-v1
```

The actor/critic optimizer-state resume from v2.25 is retained. This makes the 25K→100K→250K→500K→1M lineage materially closer to true continuous training instead of repeated warm starts.

## 3. Ablation configurations can be preregistered by content hash

The existing `ExperimentProtocol` now optionally binds a SHA-256 for every preregistered system configuration. The new v2.26 empirical ablation matrix is:

```text
actor_only
belief_actor
world_actor
world_planner
adaptive_compute
reusable_learning
full
```

Each configuration has an immutable hash before results exist. This prevents a system label from silently changing implementation or toggles after evaluation.

Generate the protocol with:

```bash
awa-v2-ablation-protocol \
  --seeds 1701,1702,1703,1704,1705 \
  --milestones 25000,100000 \
  --output runs/ablation_protocol.json
```

The protocol infrastructure is closed in v2.26. Not every ablation execution adapter is yet implemented; see `REMAINING_WORK_2_26.md`.

## Canonical autonomous campaign

Plan first:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_26_autonomous_empirical_closure.yaml \
  --out-dir runs/v2_26_meta
```

Then explicitly execute bootstrap:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_26_autonomous_empirical_closure.yaml \
  --out-dir runs/v2_26_meta \
  --bootstrap-active \
  --execute
```

After the first grounded milestone, larger cumulative milestones collect their new transition delta with the frozen parent Aether policy rather than the rule-based teacher.

## New validation commands

```bash
awa-v2-autonomous-collection-smoke
awa-v2-ablation-protocol --output runs/ablation_protocol.json
pytest -q tests/test_v2260_autonomous_ablation.py
```

## Research boundary

v2.26 does **not** establish that autonomous Aether collection is superior to teacher or coverage collection. It makes that claim testable under a controlled dataset budget.

It also does not establish that every Aether subsystem contributes. The ablation protocol binds the comparison matrix, but the missing execution variants still need to be implemented and run before components are removed.
