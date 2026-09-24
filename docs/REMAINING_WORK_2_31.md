# Remaining Work After v2.31

The architecture is now split cleanly enough to run the experiments that decide what survives.
The next work is empirical, not cognitive expansion.

1. Run real ViZDoom structured-state bring-up with tiny 2K–5K transition jobs and repair only
   failures exposed by real execution.
2. Run real frozen V-JEPA feature extraction and a small downstream control run; keep cached
   embeddings initially so perception and RL failures remain separable.
3. Execute the 25K five-seed seven-system ablation, then continue viable systems to 100K.
4. Run the physical planner microbatch benchmark on the target GPU with matched logical work.
5. Run compositional heldout/transfer tasks from the factorized environment factory.
6. Use the resulting marginal evidence to remove losing experimental families rather than
   retaining permanent compatibility shims.
7. After the empirical winner is known, physically migrate surviving flat implementation
   modules into the three canonical packages and turn historical paths into thin one-release
   compatibility wrappers.
8. Retire the remaining legacy v1 runtime once its unique reusable primitives are moved into
   a neutral stable core.
9. Refactor dense semicolon-heavy hot paths after empirical freeze, not before, to avoid a huge
   pre-experiment diff.
10. Compare the reduced Aether winner against an external recurrent/model-based baseline under
    matched transition and approximate compute budgets before scaling to 250K–1M transitions.

Do not add a new memory mechanism, LLM planner, exploration heuristic, or paper-derived module
until one of these experiments exposes a specific capability gap that the new mechanism is
intended to solve.
