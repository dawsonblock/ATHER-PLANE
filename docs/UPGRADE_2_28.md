# Aether v2.28 — Resumable Ablation Campaign + Decision Closure

v2.28 turns the executable v2.27 ablation ladder into a campaign that can actually be run across cloud GPUs without hand-managing 140 result cells.

## What changed

The canonical study remains seven executable systems, five seeds, two milestones, and two evaluation splits:

```text
7 systems × 5 seeds × 2 milestones = 70 training jobs
70 frozen checkpoints × 2 splits = 140 evaluation cells
```

A training job is now the unit of work. The system trains exactly once for a given `(system, seed, milestone)` and evaluates the same frozen checkpoint on `heldout` and `transfer`. The old failure mode of retraining once per split is therefore structurally impossible in this runner.

## Cloud-safe execution

The campaign freezes `campaign_plan.json` and `protocol.json` before execution. A different config, protocol, system hash, seed set, milestone set, or decision threshold cannot silently resume into the same output directory.

Each job is trained in a temporary attempt directory. Only after training, both split evaluations, checkpoint hashing, records, and provenance have succeeded is the directory atomically committed with `job_receipt.json`. Interrupted attempts never count as evidence.

Dataset construction and training jobs use O_EXCL locks with stale-lock recovery. Multiple workers can share the same network volume and use deterministic sharding:

```bash
awa-v2-ablation-campaign \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation \
  --execute \
  --worker-count 4 \
  --worker-index 0
```

Run indices `1`, `2`, and `3` on the other workers. Re-running any worker skips valid completed receipts.

## Dataset fairness

All non-curriculum variants with the same seed/milestone share the exact same fixed-curriculum dataset. `full_curriculum` uses a separately declared adaptive-curriculum dataset contract. Dataset series are collected once to the largest requested milestone and smaller milestones are exact prefixes, so 25K is literally a prefix of the corresponding 100K source dataset.

The collector remains the deterministic coverage policy for the component-ablation study. Collector quality is a separate experiment; changing collectors inside this matrix would confound architecture with data quality.

## Automated decision report

`ablation_report.json` and `ablation_report.md` compute adjacent marginal comparisons:

```text
actor_only       -> belief_actor      : temporal memory
belief_actor     -> world_actor       : learned world model
world_actor      -> world_planner     : fixed planning
world_planner    -> adaptive_compute  : value-of-compute allocation
adaptive_compute -> reusable_replay   : grounded hindsight replay
reusable_replay  -> full_curriculum   : adaptive curriculum
```

The report uses paired final-milestone success, normalized learning-curve area, latency, and planner-call efficiency. It outputs only `KEEP`, `REMOVE_CANDIDATE`, or `UNCERTAIN` after the entire preregistered matrix and minimum seed count are present. Before that it returns `INSUFFICIENT_EVIDENCE` and no deletion recommendation.

`REMOVE_CANDIDATE` is deliberately benchmark-scoped. It means the component failed the frozen value thresholds in this experiment, not that the concept is useless in all domains.

## Commands

Plan without spending compute:

```bash
awa-v2-ablation-campaign \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation
```

Execute/resume:

```bash
awa-v2-ablation-campaign \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation \
  --execute
```

Limit one invocation, useful for spot/preemptible nodes:

```bash
awa-v2-ablation-campaign \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation \
  --execute --max-jobs 1
```

Rebuild consolidated evidence and the decision report without retraining:

```bash
awa-v2-ablation-report \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation
```

Dependency-light campaign smoke:

```bash
awa-v2-ablation-campaign-smoke
```

## Scientific boundary

v2.28 still does not claim that full Aether is better. It makes the experiment resumable, auditable, paired, and difficult to accidentally invalidate. The actual 25K/100K campaign must still be run on real compute.
