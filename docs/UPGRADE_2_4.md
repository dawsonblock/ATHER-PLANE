# Aether v2.4 — Closed-loop benchmark qualification

v2.4 closes the gap between independently trained v2.2/v2.3 artifacts and real closed-loop evaluation.

## What changed

- Strict, versioned v2 world checkpoint loader reconstructs the exact multimodal world model and projection.
- Versioned actor checkpoint format reconstructs the TD3+BC no-search baseline.
- Identity state-vector backbones now have a stable fingerprint, so state benchmarks can use the same provenance rules as frozen vision backbones.
- Multi-seed closed-loop qualification compares the actor and search controllers on identical seed sets.
- Planner budgets are specified in approximate world-model transition calls and translated per backend.
- Scorecards report return, success, latency, model calls, search rate, compute-normalized return, paired return gain, and paired win rate.
- Paired bootstrap intervals quantify return improvement over the actor without pretending a single-seed difference is evidence.
- `awa-v2-actor-train` saves an actor checkpoint in the exact latent space of a selected world checkpoint.
- `awa-v2-benchmark-qualify` evaluates saved world/actor artifacts against any state-vector environment supported by the AWA environment factory.

## Important boundary

The command-line checkpoint benchmark currently requires an identity state-vector representation so that online observations are provably in the same feature space as the cached training data. Vision benchmarks are supported by the Python API when the exact frozen feature encoder is supplied; the CLI intentionally refuses to guess which visual backbone produced a checkpoint.

External DMControl and Meta-World packages remain optional. Their adapters are not treated as qualified simply because they import.
