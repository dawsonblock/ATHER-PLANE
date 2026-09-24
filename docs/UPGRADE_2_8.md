# Aether v2.8 — Hardening, Checkpoint Integrity, and Async Runtime Correctness

v2.8 is deliberately a correctness release. The v2.7 archive passed its existing test suite, but targeted review exposed several defects that would matter during real game training.

## Reproduced defects

1. Trained executable skill policies and learned skill contracts disappeared after reusable-learning checkpoint restore.
2. NumPy/tensor values inside metadata could make JSON checkpoint saves fail.
3. Repeated adaptive-task generation returned the same task ids and environments.
4. Novelty scoring crashed when the combined state/goal vector width changed between tasks.
5. Rollouts tagged with a future policy version received full staleness weight.
6. Exact snapshot qualification assumed legacy four-element Gym step returns and failed on Gymnasium five-element returns.
7. SafeActionGuard returned a recovery action without evaluating recovery risk.
8. Replay export assumed experience was already in episode order and mixed hindsight rows into world-model datasets.
9. Transfer episodes-to-target reported a zero-based episode label rather than number of exposures.
10. Prediction-error computation allowed NumPy broadcasting, which could silently score mismatched predictions.

## Corrected behavior

### Resumable skill execution
Reusable-learning schema v2 stores the architecture and exact state of each `DistilledContinuousSkill` and `SkillContractModel`. v1 checkpoints remain loadable.

### Curriculum freshness
`AdaptiveTaskGenerator` now owns a persistent generation counter. The engine checkpoints and restores that counter so practice-task synthesis does not repeat after restart.

### Mixed-domain memory
Novelty scoring compares only same-width prototypes. Engram buckets include vector width and query filters still guard against hash collisions.

### Async rollout integrity
Future-version samples receive zero weight. Local rollout workers normalize Gymnasium/Gym reset and step signatures and close environments in `finally`.

### Branching ABI
Exact branching normalizes four- and five-element step results and accepts an explicit `start_observation`, avoiding dependence on private environment `_obs()` helpers.

### Safety
Recovery actions are clipped to action bounds and independently risk-scored. `SelectiveDecisionController` raises rather than emitting an unsafe action when no safe actor/recovery path exists.

### Replay export
Hindsight rows are excluded by default from world-model export. If `episode_id` metadata is present, async/interleaved rows are regrouped by episode and optionally ordered by `step`; episode boundaries are marked terminal.

### Validation
`tests/test_v280_hardening.py` permanently covers the reproduced failures. The release also keeps the complete inherited regression suite.

## Remaining capability boundary
These fixes improve correctness and resumability; they do not establish that Aether transfers better on a real game. The next milestone remains multi-seed empirical training/ablation on held-out visual/layout/dynamics/compositional/task OOD splits.
