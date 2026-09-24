# Aether World Agent v2.2.0

v2.2 is the world-model qualification milestone. It deliberately avoids adding a new planner or memory subsystem.

## Added

- Episode-safe, chain-verified contiguous offline sequences.
- Cached frozen-feature sequence datasets bound to the exact v2.1 representation index.
- Offline multimodal world-model training with mixture NLL, reward, continuation and multi-step overshooting losses.
- Horizon qualification reports for latent rollout RMSE, one-step NLL, reward RMSE and continuation Brier score.
- Training of the horizon-uncertainty head against realized rollout error.
- Explicit offline constraint labels and a separate risk-model trainer/evaluator.
- Candidate-vs-baseline promotion gates that reject excessive per-horizon or NLL regressions.
- Synthetic chain-consistent world/risk dataset generation for dependency-free CI.
- `awa-v2-world-smoke`, `awa-v2-risk-smoke`, and `awa-v2-promote` CLIs.

## Scientific boundary

The smoke datasets prove that the pipeline executes and that the metrics are wired correctly. They are not evidence that Aether has learned useful real-world dynamics. Promotion claims require real cached representations, held-out environments and multi-seed comparisons.
