# Aether v2.24 — Milestone Empirical Closure

v2.24 fixes a methodological weakness in the evolving-policy promotion path. Earlier releases validated the exact preregistered matrix, but after validation they collapsed every task, split, and milestone into one per-seed average before promotion. That allowed strong early-milestone results to compensate for a worse final model, and a strong heldout split to hide a regression on transfer.

v2.24 keeps the v2.23 train-once/evaluate-many artifact contract and adds a structured milestone scorecard.

```text
preregistered paired campaign
        |
        v
25K -> 100K -> 250K -> 500K -> 1M
        |
        +--> paired delta at every milestone
        +--> normalized learning-curve area per seed
        +--> final-milestone paired gain per seed
        +--> heldout final guardrail
        +--> transfer final guardrail
        +--> planner-dependence diagnostics
        |
        v
milestone promotion gate
        |
   PROMOTE / RETAIN
```

## What changed

### Final performance is first-class

Promotion now evaluates the preregistered **final milestone** separately. Earlier milestones cannot numerically hide a final regression.

### Sample efficiency is measured across the whole curve

For every seed/task/split pair, v2.24 computes a transition-normalized area under the primary-metric learning curve. The candidate must satisfy the configured curve-gain threshold in addition to final performance.

### Split-specific guardrails

Heldout and transfer are checked independently at the final milestone. A positive pooled average cannot hide a transfer regression larger than the configured tolerance.

### Per-seed guardrail remains fail-closed

The worst final paired seed is checked directly. This preserves the earlier protection against a candidate with a good mean but one severe seed failure.

### Content-addressed scorecard

Each executed evolving-campaign iteration writes `milestone_scorecard.json` with a deterministic `scorecard_sha256`. The closed-loop promotion receipt binds to that scorecard.

### Standalone scorecard CLI

```bash
awa-v2-milestone-scorecard \
  --records evidence/run_records.jsonl \
  --baseline active \
  --candidate candidate-policy \
  --primary-metric success_rate
```

This is analysis-only. It does not bypass the experiment protocol or provenance gate.

## Canonical campaign

`configs/v2_24_milestone_empirical_closure.yaml` preregisters:

- five paired seeds
- heldout and transfer splits
- 25K, 100K, 250K, 500K and 1M transition milestones
- positive final mean gain
- non-regressing learning-curve gain
- maximum 5% final per-seed regression
- maximum 2% final per-split regression

Plan first:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_24_milestone_empirical_closure.yaml \
  --out-dir runs/v2_24_meta
```

Execution still requires explicit `--execute`, and bootstrap still requires explicit `--bootstrap-active`.

## Research boundary

v2.24 does not claim that Aether is more capable. It makes the claim boundary harder to game accidentally. The 25K→1M campaign remains real work that must be run on actual hardware; source tests and miniature smoke runs validate orchestration, not learning superiority.
