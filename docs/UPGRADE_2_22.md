# Aether v2.22 — Native Closed-Loop Execution + Load-Bearing Adaptive Compute

v2.22 removes two remaining integration gaps in v2.21.

First, the evolving campaign no longer requires an external subprocess adapter for Aether's built-in procedural game lab. A qualification-grade native runner now performs real procedural-environment collection, trains the existing temporal-belief/world-model/actor stack, evaluates deterministic held-out or transfer tasks, and emits the same content-addressed v2.21/v2.19/v2.18 evidence chain used by the generic orchestrator.

Second, the v2.20 adaptive reasoning-effort controller can now be made load-bearing inside `AdaptiveGamePolicy`. When a trained `ValueOfComputation` model is supplied, its measured planner-benefit predictions are mapped onto exact planner/budget effort levels. Levels absent from the trained VOC are not guessed; they remain unavailable. Low model reliability masks model-based planning before execution, while the downstream safety/risk layer remains separate.

## Native closed loop

```text
no grounded history
      |
      v
explicit --bootstrap-active --execute
      |
      v
active-policy grounded cells
      |
      v
DiscoveryTreeBuilder
      |
      v
historical replay / bounded meta-policy proposal
      |
      v
exact active x candidate paired matrix
      |
      v
NativeProceduralCampaignRunner
      |
      +--> real procedural collection
      +--> temporal-belief/world-model training
      +--> heldout/transfer evaluation
      +--> checkpoint/config/data/environment/dependency hashes
      |
      v
v2.19 protocol + v2.18 integrity
      |
      v
paired promotion gate
```

Bootstrap never promotes a policy. It only creates grounded history for the current active policy. If the evidence store is empty, plan mode returns `NEEDS_BOOTSTRAP` rather than failing with an opaque replay-world exception.

## Concrete curriculum-policy seam

`ExplorationPolicyCurriculumAdapter` translates the bounded declarative exploration policy into an inspectable stage allocation. The mapping is intentionally small and deterministic:

- foundation: stages 1, 2, 3, 5, 6, 9
- information/memory: stages 4, 7, 8
- novelty/dynamics: stages 10, 11
- transfer/composition: stage 12

The adapter uses largest-remainder allocation and emits a content-addressed receipt. It does not alter reward functions, evaluators, success semantics, or evidence qualification.

## Exact evidence

Every native campaign cell binds:

- Aether source-tree fingerprint
- collected training dataset SHA-256
- heldout/transfer task-set fingerprint
- runtime dependency fingerprint
- actor + world checkpoint digest
- exact runner/cell/policy config digest
- deterministic scenario IDs

The runner reports actor-only online metrics for the qualification cell, so planner calls and world-model calls are zero for those reported evaluation trajectories. Planner/world-model training diagnostics remain in the training report but are not mixed into the actor evaluation metrics.

## Adaptive compute runtime

`AdaptiveGamePolicy` now accepts:

```python
AdaptiveGamePolicy(
    encoder,
    actor_baseline,
    world,
    planners=planners,
    voc=voc,
    effort_controller=AdaptiveReasoningEffortController(...),
    model_reliability=0.9,
)
```

At each state:

```text
belief + actor action
      |
      v
VOC features
      |
      v
trained gain predictions for exact planner/budget choices
      |
      v
uncertainty + risk + task difficulty + novelty metadata
      |
      v
AdaptiveReasoningEffortController
      |
      +--> actor
      +--> policy_mppi budget N
      +--> policy_icem budget N
      +--> other trained exact choice
```

The runtime exposes `last_effort_trace` with the selected level, predicted gain, utility, uncertainty, risk, measured planner latency, and world-model call count. It does **not** fabricate realized gain; realized gain still requires exact branch/counterfactual measurement through the existing VOC qualification pipeline.

## Running from scratch

Plan first:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_22_native_evolving_campaign.yaml \
  --out-dir runs/v2_22_meta
```

With no grounded history this returns `NEEDS_BOOTSTRAP` and shows the bootstrap matrix.

Execute the explicit bootstrap and continue into the first proposal/paired validation:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_22_native_evolving_campaign.yaml \
  --out-dir runs/v2_22_meta \
  --bootstrap-active \
  --execute
```

For quick local qualification, reduce `validation.milestones`, seeds, epochs, and model size in a separate test config. Do not use reduced smoke settings as capability evidence.

## Research boundary

v2.22 still does not allow replay/model simulation to promote itself, does not rewrite arbitrary source code, and does not claim that the native procedural runner proves ViZDoom or V-JEPA performance. The native runner closes the built-in environment loop; external environments still require their own fixed evaluator adapters.
