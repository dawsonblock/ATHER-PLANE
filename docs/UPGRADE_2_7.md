# Aether v2.7 — Closed-Loop Transfer & Continual Learning

v2.7 closes the largest remaining gap in v2.6: reusable-learning components now have an integrated, resumable path from live experience back into Aether's existing world-model/actor training data contract.

## What was missing in v2.6

v2.6 had curriculum, surprise/novelty analysis, structural replay, counterfactual branching, multi-teacher distillation, skill discovery, EngramLite and rollout staleness controls, but several capabilities were still separate research utilities:

- no named task-verifier registry for synthesized tasks;
- no hindsight/goal relabeling;
- no hard separation between exploration reward and qualification reward;
- no learned skill precondition/termination/failure contracts;
- no explicit six-way held-out/OOD transfer benchmark;
- no reusable-learning state checkpoint/restore path;
- no standard export from analyzed replay back into the existing offline training dataset format;
- no local rollout worker pool implementing the same versioned-envelope semantics as async deployments;
- no concept-level competence tracking to expose catastrophic forgetting.

v2.7 adds those pieces without replacing the qualified world model, actor, planners, V-JEPA representation path, safety guard, or benchmark stack.

## Closed-loop coordinator

`ReusableLearningEngine` owns the non-neural learning state:

```text
TaskSpec / verifier
       ↓
exploit / explore / qualify mode
       ↓
experience analyzer
       ↓
structural replay + novelty + surprise
       ↓
curriculum / concept coverage
       ↓
teacher examples / skill discovery / EngramLite
       ↓
checkpoint + replay export
       ↓
existing Aether world-model / actor trainers
```

It deliberately does not own the neural optimizers. This keeps model training independently benchmarkable while making data-generation and transfer state resumable.

## Verifiable synthesized tasks

`TaskVerifierRegistry` maps `TaskSpec.verifier_name` to deterministic verifier functions. Built-ins include engine-provided success flags, distance-to-goal checks and metric thresholds. Games can register domain-specific verifiers without changing Aether.

## Hindsight goal relabeling

`HindsightGoalRelabeler` turns outcomes that were reached later in an episode into additional goal-conditioned supervision. It is especially useful when a failed attempt at one objective nevertheless demonstrates how to reach another state.

## Strict training-mode separation

`TrainingModeController` defines exploit, explore and qualification modes. `IntrinsicRewardComposer` adds novelty/information/surprise bonuses only in exploration mode. Qualification reward is exactly the task reward so curiosity shaping can never contaminate benchmark results.

## Learned executable skills and contracts

Successful clustered trajectories now populate a per-skill policy buffer. `DistilledContinuousSkill` trains a bounded goal-conditioned continuous policy for each promoted skill, so discovery produces executable behavior rather than metadata alone.

`SkillContractModel` predicts three separate probabilities:

- skill precondition is satisfied;
- skill should terminate successfully;
- skill is in a failure state.

This upgrades skill execution beyond symbolic names alone while keeping the existing symbolic precondition/effect registry as a conservative fallback.

## Transfer benchmark

The new transfer benchmark reports adaptation separately for:

1. seen conditions;
2. visual OOD;
3. layout OOD;
4. dynamics OOD;
5. compositional OOD;
6. task OOD.

Per category it records first-episode success, final-window success, adaptation AUC, episodes-to-target, prediction error and planner-dependency change. Aether can now measure whether experience actually reduces future search dependence.

## Continual-learning coverage

`ConceptCoverageTracker` records competence by reusable concept rather than only task ID. It identifies weak concepts even when aggregate task scores hide forgetting. `AdaptiveTaskGenerator` turns those diagnostics into new practice tasks, lowering difficulty for concepts that are currently out of reach and raising it after mastery.

## Local rollout pool

`LocalRolloutWorkerPool` provides deterministic threaded simulator collection and emits the same `RolloutEnvelope` used by staleness-aware queues. Distributed systems can replace the local execution backend without changing experience semantics.

## Replay export and resume

`LatentReplayExporter` writes analyzed experience into Aether's standard NPZ transition contract. `ReusableLearningEngine.save/load` persists task/curriculum history, novelty prototypes, structural replay, discovered skills, EngramLite entries, distillation examples, reasoning-effort configuration and training-mode state.

## Qualification boundary

The v2.7 tests prove these mechanisms are internally consistent and resumable. They still do not establish improved transfer in a real game. A capability claim requires held-out multi-seed training on the six OOD categories and comparison against v2.6 plus simpler actor-only/world-model baselines.
