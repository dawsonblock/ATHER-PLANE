# Aether v2.35 — Real Execution Readiness

v2.35 does **not** add a new cognitive mechanism. It hardens the boundary between
software closure and the real GPU/ViZDoom/V-JEPA experiments that now matter.
The action-time agent architecture remains the same as v2.34.

## 1. Fail-closed execution preflight

`awa-v2-execution-preflight` inspects the machine that will run the empirical
campaign. It records Python/Torch/CUDA versions, GPU identity/capability/VRAM,
free disk, output write access, ViZDoom/Transformers package availability,
V-JEPA cache presence, and a deterministic environment fingerprint.

The preflight never installs packages, downloads models, or relaxes a requested
requirement. Missing required CUDA, VRAM, disk, ViZDoom, Transformers, or local
V-JEPA state makes the report non-ready.

```bash
awa-v2-execution-preflight \
  --config configs/v2_35_empirical_execution.yaml \
  --out-dir runs/v2_35_execution
```

A passing preflight is only a readiness statement. It is not benchmark evidence.

## 2. Real ViZDoom training campaign gate

The historical ViZDoom campaign could execute against an injected test double in
unit tests. v2.35 adds a production wrapper that probes the campaign backend
before collection and refuses anything except native `vizdoom.DoomGame`.

For the pixel track it also loads V-JEPA through the non-injected Transformers
path and refuses test/fake model components. A successful run writes
`real_vizdoom_training_receipt.json` only after real collection, training and
evaluation complete.

Plan only:

```bash
awa-v2-vizdoom-real-campaign \
  --config configs/v2_35_vizdoom_structured.yaml \
  --out-dir runs/v2_35_structured
```

Execute:

```bash
awa-v2-vizdoom-real-campaign \
  --config configs/v2_35_vizdoom_structured.yaml \
  --out-dir runs/v2_35_structured \
  --execute
```

Pixel + V-JEPA:

```bash
awa-v2-vizdoom-real-campaign \
  --config configs/v2_35_vizdoom_pixel.yaml \
  --out-dir runs/v2_35_pixel \
  --execute \
  --allow-vjepa-download
```

The structured bring-up is intentionally small (2K → 10K → 25K transitions).
The pixel bring-up is even smaller (1K → 5K) because its purpose is to expose
real visual-pipeline failures before expensive scaling.

## 3. Planner hardware benchmark strengthened

The matched logical-work planner benchmark now records not only latency, memory,
logical transitions and physical forward counts, but also:

- numerical equivalence of returned action sequences across batch schedules;
- maximum action deviation from the maximally batched reference;
- logical transitions per second;
- physical forwards per second;
- Torch/CUDA runtime and GPU compute-capability/VRAM provenance.

This prevents a faster microbatch schedule from being treated as equivalent when
it silently changes the planner result.

## 4. Empirical status consumes real training receipts

`build_empirical_status()` now recognizes successful real training receipts in
addition to the lighter collection/backbone qualification receipt. A real
single-run receipt can promote T2/T3/T4 to `HARDWARE_VALIDATED`, but not to a
multi-seed `EMPIRICALLY_QUALIFIED` claim.

## 5. Canonical execution configs

- `configs/v2_35_empirical_execution.yaml` — hardware/dependency requirements.
- `configs/v2_35_vizdoom_structured.yaml` — real structured-state bring-up.
- `configs/v2_35_vizdoom_pixel.yaml` — real RGB + V-JEPA bring-up.
- `configs/v2_34_dream_rsi_empirical.yaml` — unchanged five-seed fixed-vs-DREAM
  experiment retained as the canonical DREAM-RSI qualification.

## Empirical boundary

Release validation runs software tests and CPU-safe planning/preflight checks.
It does not claim that this packaging environment executed native ViZDoom, real
V-JEPA, the five-seed DREAM-RSI campaign, the 70-job ablation matrix, or a real
GPU microbatch sweep.
