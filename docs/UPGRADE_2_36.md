# Aether v2.36 — Empirical Evidence Closure

v2.36 does not alter the action-time agent. It strengthens the handoff from real
hardware execution to auditable scientific evidence and adds a matched external
baseline contract.

## Real-execution evidence bundle

`awa-v2-real-evidence-bundle` binds a preflight receipt, a real ViZDoom training
receipt, stable checkpoint digests, and optionally a planner hardware benchmark
into one content-addressed report. Required missing or invalid evidence fails
closed.

The training receipt is accepted only when it proves a real ViZDoom backend; a
pixel-track receipt additionally requires a non-injected V-JEPA backbone. Stable
world/actor checkpoint hashes are recomputed and compared with registry hashes.

## Stronger real ViZDoom receipts

Real campaign receipts now record:

- transitions present before a resumed invocation;
- transition delta collected in the invocation;
- checkpoint-registry integrity verification;
- stable world/actor SHA-256 digests and file sizes.

This makes cloud preemption/resume behavior visible in the final evidence.

## External baseline contract

`awa-v2-external-baseline-compare` consumes standardized Aether and external
baseline result records. Pairs are matched on seed, milestone, and track. The
protocol can require equal transition counts and bound the ratio of accelerator
seconds. It reports paired success/return deltas with deterministic bootstrap
95% intervals.

The comparator deliberately does not download, run, or modify an external
implementation. External execution remains separately owned and must produce
its own provenance-bound records.

## New commands

```bash
awa-v2-real-evidence-bundle \
  --preflight runs/v2_35_execution/execution_preflight.json \
  --training-receipt runs/v2_35_structured/real_vizdoom_training_receipt.json \
  --planner-benchmark runs/planner_schedule_gpu.json \
  --output runs/real_execution_evidence.json \
  --require-planner

awa-v2-external-baseline-compare \
  --aether-records runs/aether_records.json \
  --baseline-records runs/baseline_records.json \
  --output runs/external_baseline_comparison.json
```

## Empirical boundary

Release validation does not claim native ViZDoom performance, V-JEPA downstream
performance, DREAM-RSI benefit, seven-system ablation results, or superiority
over an external baseline. v2.36 only makes those real results harder to lose,
mis-match, or overstate once produced.
