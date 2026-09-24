# Aether v2.6 — Reusable Learning Engine

v2.6 changes the training objective from memorizing individual tasks toward extracting reusable structure from experience. The qualified v2.5 perception/world-model/actor/planner stack remains intact.

## Runtime/training loop

```text
procedural task factory
        ↓
learning-frontier scheduler
        ↓
play / explore / qualification
        ↓
experience analyzer
  ┌─────┼────────┬──────────┐
  ↓     ↓        ↓          ↓
novelty failure surprise uncertainty/risk
  └─────┼────────┴──────────┘
        ↓
structural prioritized replay
        ↓
world model + actor
        ↓
value/reasoning effort controller
        ↓
actor or bounded planner budget
        ↓
successful teacher trajectory
        ↓
multi-teacher distillation
        ↓
skill discovery / registry / composition
        ↓
Engram-lite reusable pattern memory
```

## Source-derived design ideas

The DeepSeek-V4.1-Flash technical report motivated four *principles* used here: automated agent-task/environment synthesis, controllable reasoning effort, on-policy distillation, and managing stale samples in asynchronous rollout training. It also motivated a small sparse conditional-memory experiment and the bounded-replay storage/compute trade-off. Aether does not copy DeepSeek's 552B MoE/CED/CSA2 language architecture.

The Bonsai 27B whitepaper is used only as a deployment reference for a future local semantic-reasoner tier. Its low-bit/hybrid-attention/DSpark work is not inserted into Aether's control or world-model path. Existing provider-neutral `CallableReasoner` remains the integration boundary for local language models.

## Curriculum

`ProceduralTaskFactory` defines twelve progressive concept stages and randomizes both presentation and rules. `LearningFrontierScheduler` prioritizes tasks near the learning frontier using success, learning progress, novelty, prediction error, and cold-start exploration. The helper `stage_mix()` implements the initial 50/25/15/10 current/mastered/harder/novel mixture.

## Structural replay

Each `ExperienceRecord` stores distinct signals rather than collapsing them into reward:

- TD error
- world-model prediction error / surprise
- novelty
- success/failure
- uncertainty
- risk
- task importance
- information value

`StructuralPriority` combines bounded versions of the learning signals and gives a modest boost to repeated surprise. Single extreme errors are capped rather than allowed to dominate replay indefinitely.

## Counterfactual causal branching

`branch_actions()` restores the exact same environment snapshot before every candidate action, generating cleaner state/action/consequence contrasts. The environment is restored at the end of collection.

## Reasoning effort

`ReasoningEffortController` selects among explicit compute levels by predicted gain minus normalized compute cost. The default ladder is actor → shallow policy-MPPI → medium policy-MPPI → policy-iCEM → strategic beam search. These labels are configuration, not hard-coded claims that one planner is best.

## Multi-teacher distillation

`TeacherPool` selects successful teachers by value minus risk and compute cost. `PolicyDistiller` supports weighted continuous-action regression into the actor. The teacher pool can contain Aether planners, scripted experts, human demonstrations, older checkpoints, or external semantic planners.

## Skills

`SkillDiscovery` clusters repeated successful trajectory transformations by latent displacement and action signatures. `SkillRegistry` promotes only candidates that meet declared success/confidence gates. `SkillComposer` searches over promoted precondition/effect transformations.

This is intentionally a conservative first compiler. Rich learned precondition/termination models remain future work.

## EngramLite and bounded context replay

`EngramLite` is a bounded sparse pattern store for reusable environment regimes, tactics and failure patterns. It is not an LLM Engram implementation. `BoundedContextReplay` stores periodic temporal-state checkpoints and reconstructs intermediate state by replaying recent transitions, trading storage for bounded computation.

## Asynchronous rollout hygiene

`RolloutEnvelope` binds experience to a policy version. `StalenessPolicy` exponentially downweights recent lag and rejects samples beyond a configured maximum. `ExperienceQueue` centralizes these semantics while leaving actual process/thread/distributed orchestration to the deployment layer.

## Qualification boundary

The new v2.6 tests and smokes prove component semantics and integration. They do not claim that automatic skill discovery or the new curriculum improves gameplay until those components are trained and evaluated on held-out game tasks.
