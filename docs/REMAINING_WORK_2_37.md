# Remaining work after v2.37

The next action is empirical execution, not architecture expansion.

1. Run CUDA/ViZDoom preflight on the intended GPU host.
2. Execute structured ViZDoom through 2K, 10K, and 25K cumulative transitions.
3. Run actor-only vs belief-actor under the preregistered ablation protocol.
4. Produce a held-out horizon curve and qualify the longest trustworthy world-model rollout horizon.
5. Prove realized planner benefit with grounded branch outcomes, then test VoC/adaptive compute.
6. Run the real RGB/V-JEPA path to at least 5K transitions and the physical planner schedule benchmark.
7. Execute compositional/transfer evaluation and the five-seed fixed-vs-DREAM-RSI campaign.
8. Compare the reduced surviving system against a matched external world-model baseline.
9. Only after those gates pass should 250K, 500K, and 1M cumulative scaling be authorized.

Do not add cognitive modules merely to produce another release number. A new mechanism should respond to a measured failure exposed by these experiments.
