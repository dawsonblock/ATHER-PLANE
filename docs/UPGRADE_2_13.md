# Aether v2.13 — Durable Training Campaigns

v2.13 turns the v2.12 research/scaling stack into a resumable cumulative training workflow intended for the first serious 100k → 250k → 500k → 1M transition campaign.

## What changed

### Resource-aware profiles

`detect_resource_profile()` selects conservative limits for CPU, Apple Silicon, ~12 GB CUDA, ~24 GB CUDA and 48 GB+ CUDA systems. Profiles bound batch size and temporal sequence length and recommend ensemble/planner/replay settings. Explicit VRAM/RAM values can be injected when planning for another machine.

The profile is a safety/operability limit, not an automatic performance claim.

### Cumulative sharded replay

The campaign owns one append-only `ShardedReplayStore`. Collection resumes from the number of already committed transitions after an interruption. Shards are content-hashed; schema drift is rejected. `materialize()` produces a chain-preserving NPZ view for the existing goal-conditioned temporal trainer.

### Replay source mixing

`mix_replay_arrays()` deterministically combines named replay sources under declared ratios and records a source id for every row. Episode ids are remapped across sources so unrelated episodes cannot silently alias. This is intended for one-step/offline mixtures; sequence learners continue to verify exact next-state chains.

### Durable stage orchestration

`GameTrainingCampaign` owns:

1. collection to a cumulative transition target,
2. replay materialization,
3. temporal belief/world/risk/actor training,
4. stage qualification,
5. checkpoint promotion,
6. exact phase resume.

The existing `ExperimentLedger` binds the run to an immutable config hash. A failed or interrupted collection phase can resume from already committed replay shards without regenerating them.

### Promotion and rollback

Every promoted world/actor pair is copied into a versioned stable-checkpoint registry with SHA-256 hashes and metrics. A candidate can be rejected for non-finite metrics, absolute thresholds, or excessive regression against the current stable checkpoint. The previous stable entry remains available for rollback.

### Campaign compute planning

`campaign_plan()` and `estimate_campaign_compute()` report cumulative transition targets and approximate optimizer update counts before the run starts. If you provide a *measured* sample throughput from your own hardware, the plan also derives a wall-clock GPU-hour estimate. v2.13 does not invent a throughput number.

## Default campaign

`configs/v2_13_training_campaign.yaml` defines four cumulative stages:

- 100k transitions: movement / obstacles / collection
- 250k: add memory, moving entities and combat
- 500k: add resources, tactics and multi-step key/door tasks
- 1M: all 12 curriculum stages including changed dynamics and composition

The targets are cumulative, not independent datasets.

## Commands

Plan only (safe default):

```bash
awa-v2-training-campaign --config configs/v2_13_training_campaign.yaml
```

Execute:

```bash
awa-v2-training-campaign \
  --config configs/v2_13_training_campaign.yaml \
  --out-dir runs/v2_13_campaign \
  --execute
```

Stop after one stage for qualification:

```bash
awa-v2-training-campaign --execute --stop-after-stage bootstrap-100k
```

Tiny end-to-end qualification:

```bash
awa-v2-training-campaign-smoke
```

## Intentional boundaries

v2.13 does not claim that the 1M campaign produces a strong agent. It makes that experiment durable and attributable. It also does not automatically infer GPU-hours unless a measured throughput is supplied. Large-scale distributed learner/actor separation, pixel-heavy V-JEPA campaigns and multi-node training remain later work.
