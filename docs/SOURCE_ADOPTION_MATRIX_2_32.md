# Source Adoption Matrix — v2.32

Source: **Dream-RSI: Recursive Self-Improvement through Evolving Worlds**, arXiv:2609.14858v1.

| Source mechanism | v2.32 decision | Aether implementation |
|---|---|---|
| Online exploration creates structured history | Adopt | `GroundedTrainingOutcome` histories |
| Completed history becomes replay simulator | Adopt | `GroundedReplayWorld` / `SupportAwareReplayPool` |
| Same decision interface online and offline | Adopt | `TrainingAllocationDecision` + `DeclarativeTrainingAllocationPolicy.choose_batch()` |
| Optimize branch choice, parallelism, stopping | Adapt | factor/collector/budget/parallel/stop allocation contract |
| Replay objective balances quality, cost, parallelism | Adopt + extend | quality, transfer, adaptation, failure, transitions, compute, latency, parallelism |
| Keep underlying agent/evaluator/interfaces fixed | Strengthen | cryptographic `DreamRSIContract` |
| Improve exploration policy from replay | Adopt | bounded `DreamRSIOptimizer` |
| Current policy competes with revisions | Adopt | incumbent protection |
| Redeploy replay winner online | Adopt with stronger gate | paired grounded multi-seed online qualification required |
| LLM rewrites executable exploration-policy code | Reject | declarative bounded parameter mutation only |
| Replay can evaluate outcomes never observed | Reject | grounded history only; exact support tracked |
| Paper results imply embodied-RL improvement | Reject | Aether-specific online evidence remains required |

## Placement in the v2.31/v2.32 system split

- `agent_runtime`: unchanged.
- `learning_system`: owns the shared allocation decision contract and real task/collector materialization.
- `research_os`: owns historical replay, support qualification, candidate comparison, and empirical promotion.

This preserves the rule that meta-learning may change how training budget is allocated but cannot silently mutate action-time inference during an evaluation.
