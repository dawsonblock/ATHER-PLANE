# Aether v2.12 — Controlled Modular Scaling

## Purpose

v2.12 is a controlled scaling release. It does not assume that a larger or more modular model is better. It adds the instrumentation needed to answer that experimentally under matched active compute and fixed dataset provenance.

## Sparse modular dynamics

`SparseMoETrunk` routes each belief/action row to only the selected top-k experts. `ModularWorldModel` preserves the normal Aether world-model interface, including multimodal future distributions, reward, continuation, terminal value, risk and uncertainty heads.

`matched_expert_hidden()` chooses the expert width so the selected expert path plus gate has approximately the same trunk MAC count as the two-layer monolithic trunk. The report records both total parameters and approximate active parameters; this prevents a sparse model's larger inactive capacity from being confused with its per-token active compute.

## Controlled scaling axes

`ControlledScalingRunner` supports four declared axes:

1. **Architecture** — monolithic vs sparse MoE at matched active trunk MACs.
2. **Data** — fixed model trained on increasing fractions of the same dataset.
3. **Model** — hidden-width scaling for monolithic and MoE dynamics.
4. **Context** — sequence length, temporal event slots and global-update stride.

Every point records training time, final training loss, horizon rollout RMSE, reward RMSE, value RMSE, parameter counts and active compute estimates. Multi-seed runs are summarized with bootstrap confidence intervals.

## Resume and provenance

`ExperimentLedger` binds an output directory to a canonical config hash. Completed phases are reused exactly on restart. Changing the experiment configuration in place is rejected instead of silently mixing incompatible results. State writes are atomic.

## Precision and devices

The temporal world-model trainer now accepts `fp32`, `bf16`, or `fp16`. FP16 is restricted to CUDA. The scaling CLI supports `auto`, `cpu`, `cuda`, or `mps` device selection.

## Commands

Quick qualification:

```bash
awa-v2-controlled-scaling-smoke
```

A real sweep:

```bash
awa-v2-scaling-sweep runs/game.npz \
  --config configs/v2_12_controlled_scaling.yaml \
  --out-dir runs/v2_12_scaling
```

The default data fractions `[0.10, 0.25, 0.50, 1.00]` correspond approximately to 100k/250k/500k/1M experience when the source dataset contains one million representative transitions. For serious claims, keep the held-out evaluation split fixed and use the configured multi-seed qualification rather than comparing single runs.

## Boundary

v2.12 establishes the machinery for controlled scaling studies; it does not establish that MoE, larger hidden widths, longer context or more data improve Aether. Promotion should depend on measured held-out transfer and compute-normalized performance. The v3.0 roadmap gate remains unchanged.
