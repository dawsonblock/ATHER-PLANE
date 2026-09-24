# v2.20 Source Adoption Matrix

This document records what Aether v2.20 actually adopts from the two supplied research reports and what remains deliberately outside the trusted path.

## Dream-RSI: Recursive Self-Improvement through Evolving Worlds (arXiv:2609.14858v1)

| Source mechanism | v2.20 treatment | Aether implementation |
|---|---|---|
| Completed discovery history as replay simulator | Adopted | `DiscoveryTree`, `ReplayWorld`, `ReplaySimulatorPool` |
| Explicit programmable exploration policy | Adopted with narrower surface | `DeclarativeExplorationPolicy`; bounded data, no arbitrary policy code execution |
| Branch / continue / stop / parallel allocation | Adopted at macro-learning level | replay parent scoring + `max_parallel` + stopping threshold |
| Offline policy improvement over historical worlds | Adopted | `DeclarativePolicyMutator`, `MetaPolicyOptimizer` |
| Redeploy improved exploration policy online | Adopted with stronger gate | `ExplorationPolicyPromotionGate` requires paired grounded online evidence |
| Recursive loop as new histories accumulate | Adopted | `EvolvingExplorationController` |
| Historical replay as proof of unseen outcomes | Rejected | simulated/counterfactual outcomes are never promotable evidence |
| Unrestricted generated exploration code | Rejected for v2.20 | only bounded declarative policy parameters may evolve |

Aether extends the paper's replay idea with an explicit evidence-origin type system. Learned-world-model imagination may guide search, but it cannot cross into empirical qualification until validated online.

## DeepSeek-V4.1-Flash technical report

| Source mechanism | v2.20 treatment | Reason |
|---|---|---|
| Controllable reasoning effort | Adapted | maps naturally onto Aether's actor/planner compute ladder |
| Large-scale synthesized agent tasks | Already represented / retained | Aether already has procedural task/world generation and controlled composition |
| On-policy distillation | Already represented / strengthened by telemetry | planner/teacher behavior already distills into the actor; v2.20 records realized compute benefit |
| Asynchronous rollout/post-training | Already represented | Aether already has rollout pools, process workers and staleness handling |
| Evaluation across agent scaffolds | Recommended qualification extension | useful for checking adapter/interface overfitting; not treated as a solved capability claim |
| Causal Encoder-Decoder (CED) | Not transplanted | Transformer-native model architecture, not an Aether world-model primitive |
| Compressed Sparse Attention 2 (CSA2) | Not transplanted | depends on the DeepSeek attention/KV architecture |
| FP4 KV cache | Not transplanted into core Aether | serving optimization for KV-heavy Transformer inference |
| SWA Bounded Replay | Not confused with experience replay | reconstruction strategy for Transformer SWA KV; conceptually distinct from discovery-tree replay |
| Engram / DSpark / Mega-mHC | Not transplanted | model-specific components; require their own evidence and checkpoint/kernel contracts |

## v2.20 trust boundary

The release has three separate information classes:

1. **Grounded experience** — observed or separately validated in the real environment; may support promotion.
2. **Historical replay** — zero-execution-cost recombination of grounded outcomes already present in completed trees; may propose a policy but cannot prove new outcomes.
3. **Model imagination / counterfactuals** — useful search hypotheses; never treated as empirical evidence.

The v2.19 preregistered experiment protocol remains the outer empirical qualification authority.
