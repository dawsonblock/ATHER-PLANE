# Aether v2.29 Remaining Work

v2.29 closes the logical-vs-physical compute-accounting gap for planner rollouts. The next work should generate empirical evidence rather than add cognitive mechanisms.

## 1. Run the canonical 25K/100K ablation matrix

Execute all seven systems over five seeds and both heldout/transfer splits using `configs/v2_29_compute_efficiency.yaml`. This is still the primary gate for deciding which mechanisms survive.

## 2. Run a matched planner-schedule study

Hold checkpoint, task states, candidate budget, horizon, random seeds and planner algorithm fixed. Compare `planner_world_batch_size` values such as `0`, `32`, `8`, `4`, `1` on the same GPU. Measure latency, throughput, peak VRAM and, when direct hardware measurement becomes available, device energy.

The purpose is to identify the best schedule for each GPU rather than assuming maximum batching is always optimal.

## 3. Add direct GPU energy measurement

Integrate a hardware-backed meter (for example NVML total-energy counters when the device exposes them, or a validated sampled-power integration path). Store device identity, sampling/counter method, measurement window and meter uncertainty. Never substitute TDP × wall time as measured energy.

## 4. Add training-step physical accounting

Planner inference is now explicit. Training should next record optimizer-step counts, forward/backward microbatch schedule, tokens/rows processed, gradient accumulation, mixed-precision mode and data-loader wait time. This will separate a logical data budget from the physical training schedule as rigorously as v2.29 does for planning.

## 5. Matched collector study

Compare teacher, random, coverage, frozen Aether actor and Aether+planner collection under identical logical transition budgets and report both downstream capability and physical collection cost.

## 6. Planner/VOC utility study

From identical saved states, compare actor-only, fixed MPPI/iCEM and adaptive compute. Record realized return/success gain, physical forwards, logical model work, latency and constraint violations. Test whether planner dependence decreases with familiarity.

## 7. Compositional transfer

Train isolated concepts and test novel combinations. This remains the strongest test of whether replay/curriculum mechanisms create reusable structure instead of benchmark specialization.

## 8. Delete losing mechanisms

After complete evidence exists, produce a reduction release. Components that fail capability, transfer, sample-efficiency and compute-efficiency gates should move to `awa.experimental` or be archived.

## 9. External world-model baseline

Compare the smallest winning Aether configuration with a strong external world-model agent under matched environment transitions, seeds, accelerator budget and evaluation tasks.

## 10. ViZDoom repeat

Repeat the surviving ablations first on structured-state ViZDoom and then RGB + frozen V-JEPA. Procedural-environment evidence is not sufficient to claim visual-control generality.
