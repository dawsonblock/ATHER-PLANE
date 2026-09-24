# Aether World Agent v2.29 — Compute-Aware Planning and Physical Execution Accounting

v2.29 is a systems/measurement release. It does not add a new cognitive mechanism. It makes the physical execution of existing planners explicit so equal logical candidate budgets are not mistaken for equal hardware cost.

## Research input

The release is informed by Kashaniyan and Jannesari, *Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling*, arXiv:2609.19499v1 (16 Sep 2026): https://arxiv.org/abs/2609.19499

The source studies multi-candidate LLM inference, not model-based RL. Its directly supported systems result is that a fixed candidate count does not determine physical cost: candidates can be grouped into fewer large batched calls or more small sequential calls. At fixed N=8, the paper compares 1×8, 2×4, 4×2 and 8×1 schedules and reports materially different latency, GPU-hours/utilization and gross GPU-device energy. The authors recommend reporting candidate count together with the generation-call schedule and GPU-level systems metrics.

Aether does **not** treat the paper as evidence that more RL data is automatically better or that a particular planner is superior. v2.29 adopts only the scheduling/accounting lesson.

## Aether mapping

For an Aether planner, the analogous distinction is:

- **logical work** — candidate trajectories × rollout horizon × stochastic samples/iterations;
- **physical execution** — actual `world.imagine_step(...)` invocations and the tensor batch size in each invocation.

Prior Aether releases reported `world_model_calls` as a candidate-wise logical count. That is useful for algorithmic budgeting, but it is not the same thing as the number of physical forward invocations.

v2.29 therefore preserves the legacy count and adds:

- `logical_world_model_transitions`;
- `physical_world_model_forwards`;
- `batch_size_histogram`;
- `max_batch_size`;
- `mean_world_model_batch_size`;
- `logical_transitions_per_forward`;
- `planner_world_batch_size` as a frozen execution-schedule control.

## Planner batching

MPPI, policy-seeded MPPI, CEM and iCEM already operated on candidate tensors. v2.29 does not falsely claim that vectorization is new. It makes that vectorization measurable and configurable.

`planner_world_batch_size: 0` means maximally batched execution. A positive value performs deterministic microbatching. For example, eight independent candidates become:

- `0` or `8`: `(8)`;
- `4`: `(4, 4)`;
- `2`: `(2, 2, 2, 2)`;
- `1`: `(1, 1, 1, 1, 1, 1, 1, 1)`.

The logical candidate workload is unchanged. The physical number of world-model forwards changes.

Policy-seeded planners also account for the actor-prior rollout separately. Beam search and gradient planning report their serial execution instead of being mislabeled as batched.

Risk-aware and hybrid Doom MPPI paths use the same physical schedule accounting, including stochastic future samples.

## Campaign evidence

The v2.28 70-training-job / 140-evaluation-cell experiment contract is retained. v2.29 adds an execution fingerprint to every completed job and writes a consolidated `compute_schedule_report.json`.

Per evaluation record, the campaign now records:

- planner calls per episode;
- logical world-model transitions per episode;
- physical world-model forwards per episode;
- mean world-model batch size;
- logical transitions per physical forward;
- inference latency;
- accelerator wall-hours for training when running on CUDA;
- peak allocated CUDA memory;
- source transition count and effective training rows.

The campaign config hash includes `planner_world_batch_size`, so changing the physical execution schedule requires a new output directory/protocol binding rather than silently changing cost semantics mid-experiment.

## Energy boundary

v2.29 does **not** estimate GPU energy from wall-clock time or TDP. Those are not interchangeable with measured device energy. `accelerator_energy_joules` therefore remains `null` and `energy_measurement` is `not_measured` unless a future release integrates a real hardware energy meter.

This is deliberate fail-closed measurement behavior.

## Decision report

The adjacent ablation report can now credit a component when it is statistically non-inferior in success and produces a preregistered meaningful reduction in any of:

- inference latency;
- planner calls;
- physical world-model forwards;
- logical world-model work.

This allows `adaptive_compute` to justify itself through real compute reduction even when final task success is unchanged.

## Canonical run

```bash
awa-v2-ablation-campaign \
  --config configs/v2_29_compute_efficiency.yaml \
  --out-dir runs/v2_29_ablation

awa-v2-ablation-campaign \
  --config configs/v2_29_compute_efficiency.yaml \
  --out-dir runs/v2_29_ablation \
  --execute

awa-v2-ablation-report \
  --config configs/v2_29_compute_efficiency.yaml \
  --out-dir runs/v2_29_ablation
```

To run a memory-constrained schedule, copy the config, set for example `planner_world_batch_size: 8`, and use a new output directory. Do not compare it with the canonical run as if the execution schedule were identical.

## Scientific boundary

Release validation proves the accounting, deterministic microbatch semantics, resumable campaign integration and report logic. It does not establish that maximal batching is always faster on every GPU, does not measure energy, and does not execute the full 25K/100K five-seed empirical campaign.
