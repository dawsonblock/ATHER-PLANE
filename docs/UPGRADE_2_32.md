# v2.32 — Grounded DREAM-RSI Meta-Learning

v2.32 integrates the parts of DREAM-RSI that fit Aether's architecture without adding another action-time cognitive mechanism.

The paper's central loop is preserved: online exploration creates structured history; completed histories become replay simulators; candidate exploration policies are evaluated cheaply in replay; the selected policy is redeployed online. Aether adapts that loop to **training allocation** rather than joystick control.

## What changed

- `awa.v2.learning_system.meta_allocation` defines one declarative decision interface used both online and in replay, plus a real bridge that materializes factorized tasks and the requested collector.
- `awa.v2.research_os.dream_rsi` adds grounded replay worlds, support-aware replay scoring, bounded policy mutation, incumbent protection, and paired online promotion.
- The underlying agent/evaluator/interface are cryptographically frozen with `DreamRSIContract` during a comparison.
- Replay accepts only `OBSERVED`/`VALIDATED` history. Simulated or counterfactual outcomes cannot enter the empirical replay pool.
- A replay winner is never promoted directly. It must pass paired real-online qualification.
- Historical support is explicit. Weakly supported replay winners are marked `ONLINE_PROBE_REQUIRED`; unsupported policies cannot win the replay selection.
- Policy improvement is declarative parameter search, not arbitrary LLM-written Python.

## Shared decision interface

`TrainingAllocationDecision` can allocate a factor signature, collector, transition budget, planner budget, and parallel worlds, or explicitly stop. `DeclarativeTrainingAllocationPolicy.choose_batch()` is the exact method used in both historical replay and future real allocation execution.

## Scientific boundary

DREAM-RSI reports strong cost/quality results in algorithm engineering, mathematical optimization, and GPU-kernel engineering. v2.32 does **not** treat those results as evidence that the method improves embodied RL. That must be measured in Aether's own controlled campaigns.

## Smoke check

```bash
awa-v2-dream-rsi-smoke
```

The smoke check builds grounded replay worlds under a frozen contract, dreams a bounded policy revision, then requires paired grounded online evidence before promotion.
