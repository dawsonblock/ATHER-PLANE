# Aether v2.20 — Evolving Exploration + Adaptive Compute

v2.20 integrates two ideas without changing the empirical trust boundary established in v2.18–v2.19.

The first source is **Dream-RSI: Recursive Self-Improvement through Evolving Worlds** (arXiv:2609.14858v1). Its relevant mechanism is historical discovery-tree replay: completed exploration histories become replay simulators in which alternative branching, ordering, parallelism and stopping policies can be evaluated cheaply before another expensive online rollout. Aether adapts this at the *macro-learning* level rather than at the per-frame game-action level.

The second source is the **DeepSeek-V4.1-Flash technical report**. Aether does **not** transplant CED, CSA2, FP4 KV cache, Engram, or SWA Bounded Replay into its non-Transformer world-model core. The transferable idea used here is explicit control of reasoning effort and the existing planner-to-policy distillation direction. Aether now has a graded compute allocator that trades expected gain against compute, latency and risk while respecting world-model reliability. The downstream risk guard remains authoritative.

## New meta-learning plane

`awa.v2.meta_exploration` adds:

- `DiscoveryTree`: append-only macro experiment history with a single primary parent per node.
- `EvidenceClass`: `observed`, `model_simulated`, `counterfactual`, `validated`.
- `ReplayWorld`: exposes only grounded (`observed`/`validated`) historical outcomes.
- `DeclarativeExplorationPolicy`: bounded weights and thresholds; no arbitrary generated code is executed.
- `ReplaySimulatorPool`: evaluates one policy across multiple historical worlds.
- `DeclarativePolicyMutator`: deterministic local policy search with explicit novelty/adversarial exploration floors.
- `MetaPolicyOptimizer`: proposes a replay-improved policy.
- `ExplorationPolicyPromotionGate`: requires paired, grounded online evidence before promotion.
- `EvolvingExplorationController`: runs the outer dream → online validation → promote/retain loop.

The important invariant is:

```
MODEL_SIMULATED / COUNTERFACTUAL -> search hints only
OBSERVED / VALIDATED           -> empirical promotion evidence
```

Replay can propose; replay cannot promote.

## Adaptive compute

`awa.v2.reasoning.effort` retains the old `ReasoningEffortController` ABI and adds:

- `EffortSignals`: uncertainty, novelty, risk, actor confidence, historical planner benefit, goal difficulty, model reliability and resource pressure.
- `AdaptiveReasoningEffortController`: chooses among actor/shallow/medium/deep/strategic compute levels using expected gain minus compute/latency/risk penalties.
- reliability gating: model-based planning levels are masked when the world model is below the configured reliability floor.
- `EffortOutcomeLedger`: records predicted vs realized planner gain, latency, world-model calls and success so planner dependence can be measured and later learned from.

This makes "how hard should I think?" an explicit control variable instead of a binary actor-vs-planner switch.

## Deliberate non-goals

v2.20 does **not**:

- allow an LLM or replay optimizer to rewrite arbitrary repository source code;
- treat world-model imagination as empirical evidence;
- replace the v2.19 preregistered five-seed qualification protocol;
- import Transformer-native DeepSeek cache mechanisms into Aether's world model;
- claim that the new outer loop improves transfer before real paired experiments are run.

## Intended loop

```
real campaigns
    -> grounded discovery trees
    -> historical replay worlds
    -> bounded declarative meta-policy search
    -> candidate exploration policy
    -> paired online validation
    -> v2.19 evidence/protocol qualification
    -> promote or retain current policy
```

Inside each environment step, adaptive compute independently controls actor/planner effort. Planner behavior can still be distilled into the actor using the existing teacher/distillation stack.
