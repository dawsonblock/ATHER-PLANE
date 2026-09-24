# Remaining Work After v2.30

The architecture is frozen until the following evidence is collected. Do not add new cognitive modules merely because an experiment is missing.

## 1. Execute the 25K/100K seven-system matrix

Run all 70 training jobs / 140 heldout+transfer cells from `configs/v2_30_empirical_pivot.yaml`. Preserve the frozen output directory and protocol binding. Use the keep/remove report only after the complete matrix is present.

## 2. Real GPU planner schedule sweep

Use one frozen trained checkpoint on at least one actual target GPU. Compare planner world batch limits `0,32,8,4,1` under the same logical candidate budget. Record latency, peak memory, physical forwards and hardware identity. If practical, repeat on one consumer GPU and one datacenter GPU.

Do not estimate energy. Add NVML/device-energy integration only if the hardware/API exposes a defensible direct measurement.

## 3. Real ViZDoom structured-state qualification

Install the actual `vizdoom` package and run T2 qualification. Then train/evaluate the surviving Aether variants on real ViZDoom structured state with multiple seeds. FakeDoomGame tests remain CI only.

## 4. Real RGB + V-JEPA qualification

Run T3/T4 qualification using actual ViZDoom RGB frames and a non-injected V-JEPA checkpoint. Then train the downstream Aether stack from cached frozen V-JEPA features. Report representation cache identity and real model revision/commit when available.

## 5. Compositional-transfer study

Use `COMPOSITIONAL_BENCHMARK_2_30.json` or a versioned successor. Report zero-shot heldout composition, transfer composition, episodes-to-adaptation, planner dependence and compute. Require at least five seeds before calling T6 empirically qualified.

## 6. External baseline

Compare the smallest surviving Aether architecture against a strong modern world-model agent under matched environment interactions, seeds, observation interface and accelerator budget. Do not compare against a deliberately weak baseline only.

## 7. Finish legacy retirement

After the v2.30 matrix proves which stable modules survive:

- remove compatibility shims for orphan v2 prototypes after one deprecation release;
- split the entire v1 runtime into a separate historical package/archive;
- remove the old connectome CLI from the default install;
- collapse historical console scripts behind one `awa` subcommand interface.

## 8. Code readability pass

Refactor semicolon-heavy multi-operation lines in the actively trained/evaluated path first: `game/`, `planners.py`, `world.py`, `native_campaign.py`, `ablation_campaign.py`, and `variant_runtime.py`. Preserve behavior and verify with the full suite. Avoid spending time reformatting historical compatibility code that is scheduled for removal.

## 9. Data/environment scaling

Expand the factorized environment factory only when new factors correspond to a concrete capability question. Add difficulty calibration from measured agent success rather than static hand-tuned labels, deduplicate generated tasks by semantic signature, mine failure cases, and maintain train/heldout/transfer provenance.
