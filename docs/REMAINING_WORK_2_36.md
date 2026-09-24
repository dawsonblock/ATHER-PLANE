# Remaining work after v2.36

1. Run the real structured ViZDoom bring-up on a CUDA host and preserve its
   preflight + training receipts.
2. Run the real RGB/V-JEPA bring-up after structured control is stable.
3. Execute the planner microbatch sweep on the actual target GPU and bind it
   into a real-execution evidence bundle.
4. Execute the v2.34/v2.35 five-seed fixed-vs-DREAM-RSI campaign unchanged.
5. Execute the seven-system 25K/100K ablation independently.
6. Run a strong external baseline (for example DreamerV3 or TD-MPC2) through its
   own reproducible harness, export standardized result records, and compare
   them with `awa-v2-external-baseline-compare` under matched budgets.
7. Do not add new cognitive modules until these experiments identify a concrete
   capability deficit.
8. Keep energy reporting unmeasured until direct hardware counters are wired and
   validated; do not substitute TDP × wall time.
