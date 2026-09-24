# Aether v2.3 — Planner Qualification

v2.3 is the planner-measurement milestone. It does not assume that online search is useful. It adds controls and data collection needed to measure when search improves real return enough to justify its compute cost.

## New research controls

- `CEMPlanner`: policy-free vanilla CEM baseline.
- `MPPIPlanner`: policy-free MPPI baseline.
- `PolicySeededMPPI`: actor trajectory is the MPPI nominal sequence.
- `PolicySeededICEM`: actor trajectory initializes iCEM.
- `GradientPlanner`: differentiable action-sequence optimization that re-enables action gradients even under inference/no-grad callers.
- `NoSearchPlanner`: actor-only reference.

The planner arena converts a requested world-model transition-call budget into backend-specific candidate/iteration budgets. This makes comparisons approximately equal-compute rather than equal-iteration.

## Actor-only baseline

`OfflineActorCriticBaseline` is a compact TD3+BC-style continuous baseline with twin Q critics, EMA target actor/critic updates, and bounded actions. `latent_dataset_from_cached_features` builds its training data from the same v2.1 frozen-feature cache and projection used by the world-model path.

The actor is the default control path. Planners must show real incremental value over this baseline.

## Exact counterfactual branching

`awa.v2.branching` requires `state_dict` / `load_state_dict` support and verifies deterministic snapshot replay before collecting planner benefits. Every actor and planner branch starts from the same environment state. Probe branches are restored afterward; the main collection trajectory follows the actor.

Each record includes:

- actor return
- planner return
- gain over actor
- planner identity and budget
- actual world-model transition calls
- latency
- normalized compute cost
- uncertainty/risk/action/belief diagnostics

## Value of Computation

v2.3 fits `ValueOfComputation` directly from measured branch gains. The model predicts expected gain for every planner/budget choice and subtracts an explicit learned compute penalty. Qualification reports include gain MAE, utility MAE, choice accuracy, and learned cost scale.

## Fail-closed action guard

`SafeActionGuard` is now available as the last learned-risk gate. A proposal above the configured learned-risk threshold is replaced with a configured neutral/recovery action rather than silently returning the unsafe proposal. This is still not a substitute for deterministic actuator limits or certified safety logic.

## Commands

```bash
awa-v2-actor-smoke --epochs 8
awa-v2-planner-qualify --world-calls 64 --states 16 --branch-horizon 3
awa-v2-safety-smoke
awa-v2-arena --world-calls 64
```

The built-in planner qualification uses `ContinuousPointEnv` and an exact analytical point-world model solely to validate the measurement machinery. It is not a learned-world-model capability result. Real promotion requires the same APIs to be run with the trained v2.2 world model and declared benchmark environments.

## Promotion rule

A planner should not become default merely because its model-predicted score is higher. Promotion requires improved **real branch return per unit of compute**, reproducible across seeds/tasks, while the actor-only path remains the default when the expected value of search is non-positive.
