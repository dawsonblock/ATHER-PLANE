# Remaining work after v2.32

1. Bind the materialized task/collector plan into the exact-transition campaign runner so `transition_budget` is enforced by the same cumulative replay-prefix machinery used by canonical training.
2. Export actual completed v2.31/v2.32 campaign histories into `GroundedTrainingOutcome` trees instead of synthetic smoke histories.
3. Run a controlled fixed-meta-policy vs DREAM-RSI allocation comparison with the same agent, evaluator, initialization, and resource budget.
4. Measure whether DREAM-RSI improves heldout/transfer capability per environment transition and per accelerator-hour.
5. Add support-distance diagnostics for unseen factor compositions. Exact historical support remains the current conservative gate.
6. Keep the action-time `agent_runtime` frozen during these experiments.
7. Continue the higher-priority empirical program: real structured ViZDoom, real V-JEPA, and the 25K/100K ablation campaign.
8. Do not claim recursive self-improvement in embodied RL until paired online experiments demonstrate it.
