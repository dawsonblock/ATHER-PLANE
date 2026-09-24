# Remaining work after v2.34

1. Execute `configs/v2_34_dream_rsi_empirical.yaml` on five paired seeds. Do not
   change the frozen menu, thresholds, resource ceilings, or final meta-test after
   seeing results.
2. Report both capability and cost. A DREAM gain bought with substantially more
   real transitions or accelerator time is not equivalent to a same-cost gain.
3. If the procedural paired result is positive, repeat the surviving fixed-vs-DREAM
   comparison on real structured ViZDoom before making a general embodied-RL claim.
4. Run the existing 25K/100K seven-system ablation independently of the DREAM
   comparison; the two experiments answer different questions.
5. Complete real RGB + V-JEPA qualification only after structured ViZDoom is stable.
6. Add a true multi-decision-round historical tree only if the one-step replay-world
   adaptation proves useful. v2.34 deliberately uses grounded one-round allocation
   worlds plus cumulative checkpoint lineage; it does not pretend to reproduce every
   source-paper tree traversal detail.
7. If DREAM-RSI does not improve independent final heldout/transfer performance or
   useful sample efficiency, remove it from the canonical Learning System and retain
   it only as an experimental research-OS module.
