# Remaining empirical work after v2.38

The code path is ready to orchestrate the staged campaign. The remaining work is execution, not another cognitive release.

1. Run CUDA/ViZDoom preflight on the target GPU host.
2. Execute structured ViZDoom at 2K, 10K and 25K cumulative transitions.
3. Run the focused temporal-memory comparison.
4. Produce a held-out world-model horizon curve and qualify the longest trustworthy contiguous horizon.
5. Run focused fixed-planner and adaptive-compute comparisons.
6. Execute genuine RGB/V-JEPA 5K bring-up.
7. Run the action-equivalent physical GPU planner schedule sweep.
8. Produce compositional generalization evidence.
9. Run the paired five-seed fixed-vs-DREAM-RSI campaign.
10. Produce unseen-map zero-shot/adaptation evidence.
11. Run a matched external world-model baseline.
12. Run the full seven-system 25K/100K preregistered matrix before permanent pruning.
13. Authorize 250K+ scaling only after the preceding evidence supports it.

Do not add new cognitive modules merely to create a new release. New architecture should be driven by a concrete failure exposed by these experiments.
