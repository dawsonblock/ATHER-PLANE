# Aether v2.21 — Evolving Campaign Closure

v2.21 closes the operational gap left by v2.20. v2.20 could build grounded replay worlds, dream over bounded exploration-policy changes, and independently gate online promotion. It did not provide one durable service that converts completed empirical runs into replay trees, plans exact paired validation cells, launches those cells through a bounded runner seam, applies the v2.19 protocol/integrity gates, and updates the active policy only after those gates pass.

v2.21 adds that service.

## Canonical loop

```text
completed grounded campaign evidence
        |
        v
DiscoveryTreeBuilder
  one world per seed/split
  root branches = system/task experiments
  branch depth = transition milestones
        |
        v
ReplaySimulatorPool
        |
        v
MetaPolicyOptimizer
        |
        v
bounded DeclarativeExplorationPolicy proposal
        |
        v
PairedCampaignPlan
  active x candidate
  exact seeds x tasks x splits x milestones
        |
        v
CampaignRunner
  explicit argv, no shell
        |
        v
CampaignEvidenceStore
  atomic, append-only, conflict rejecting
        |
        v
v2.19 ExperimentProtocol + v2.18 provenance/integrity
        |
        v
paired grounded seed aggregation
        |
        v
ExplorationPolicyPromotionGate
        |
        +--> PASS   -> candidate becomes active
        |
        +--> FAIL   -> active policy retained
```

Replay still cannot promote. Model-simulated and counterfactual evidence still cannot enter the online qualification path. Source-code self-rewrite is not part of this loop.

## New module

`awa.v2.evolving_campaign` contains:

- `MetricProjection`
- `CampaignCell`
- `PairedCampaignPlan`
- `CampaignEvidence`
- `CampaignEvidenceStore`
- `SubprocessCampaignRunner`
- `DiscoveryTreeBuilder`
- `ValidationMatrix`
- `ClosedLoopReceipt`
- `EvolvingCampaignOrchestrator`

The orchestrator state is atomic and resumable. Completed validation cells are committed individually, so restarting a partially completed iteration does not rerun cells unless explicitly requested.

## Discovery history conversion

A flat `RunRecord` matrix is not treated as a set of isolated one-node worlds. v2.21 reconstructs one replay world per `(seed, split)` and uses each `(system, task)` pair as a root branch. Increasing transition milestones extend that branch. This gives the replay policy a real branch/continue decision surface while preserving the original empirical records.

Optional metrics such as novelty, information gain, and uncertainty reduction are never invented. Missing optional metrics project to zero. Wall-clock seconds are scaled explicitly by `MetricProjection.latency_scale`.

## Exact validation matrix

For every proposal, v2.21 generates an exact matrix:

```text
{active, candidate}
  x seeds
  x tasks
  x splits
  x milestones
```

The matrix becomes a temporary v2.19 `ExperimentProtocol`. Qualification fails closed on missing cells, unexpected cells, duplicate cells, malformed metrics, invalid provenance, or train/evaluation scenario leakage.

Unlike the original v2.19 helper, `build_protocol_receipt()` now accepts an explicit baseline. The default remains `full` for backward compatibility; the evolving loop passes the currently active policy as the baseline.

## Runner contract

Plan mode is the default and does not execute external processes.

```bash
awa-v2-evolving-campaign \
  --config configs/v2_21_evolving_campaign.yaml \
  --history-records evidence/history.jsonl \
  --history-provenance evidence/provenance.json \
  --out-dir runs/v2_21_meta
```

Actual execution requires `--execute` and a configured `runner.argv`.

The runner uses `subprocess.run(argv, shell=False)`. Supported placeholders are:

```text
{policy_json}
{policy_id}
{seed}
{task}
{split}
{transitions}
{output}
{work_dir}
```

The child process must write `awa-v2.21-campaign-evidence-v1` JSON to `{output}`. The evidence contains one `RunRecord`, one matching `RunProvenance`, and a grounded evidence class. A mismatched cell, non-grounded class, or conflicting previously committed cell is rejected.

## Evidence document

```json
{
  "format": "awa-v2.21-campaign-evidence-v1",
  "evidence": "validated",
  "record": {
    "system": "active",
    "seed": 1701,
    "task": "vizdoom",
    "split": "heldout",
    "transitions": 25000,
    "metrics": {
      "success_rate": 0.61,
      "episode_return": 12.4,
      "constraint_violations": 0.0,
      "planner_calls_per_episode": 2.1,
      "world_model_calls_per_episode": 84.0,
      "inference_latency_ms": 9.3,
      "wall_clock_seconds": 4100.0
    },
    "checkpoint_sha256": "...64 lowercase hex...",
    "config_sha256": "...64 lowercase hex..."
  },
  "provenance": {
    "run_id": "active|1701|vizdoom|heldout|25000",
    "source_sha256": "...",
    "dataset_sha256": "...",
    "environment_sha256": "...",
    "dependency_lock_sha256": "...",
    "scenario_ids": ["heldout-map-001"]
  }
}
```

## Promotion semantics

After the exact protocol passes, all preregistered cells for each `(policy, seed)` are averaged on the configured primary metric. The aggregate provenance digest binds the constituent cell provenance receipts. The normal paired promotion gate then checks the mean gain and worst per-seed regression.

No policy is promoted if protocol closure fails.

## Commands

```bash
awa-v2-evolving-campaign-smoke
pytest -q tests/test_v2210_evolving_campaign.py
```

The smoke test uses synthetic evidence only to test orchestration mechanics. It is not empirical evidence about Aether capability.

## What remains environment-specific

v2.21 deliberately does not pretend that one generic adapter can define what a valid ViZDoom, Unity, or other external-environment campaign means. The orchestration and evidence ABI are complete, but the real runner configured in `runner.argv` must own environment-specific collection/training/evaluation and produce the required grounded evidence document.

That boundary is intentional: the meta-controller may choose where to spend experiment budget, but it does not get to redefine the evaluator or evidence semantics while it is being evaluated.
