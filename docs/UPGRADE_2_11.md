# Aether v2.11 — Empirical Scaling and Robust Planning

## Purpose

v2.11 does not replace the v2.10 objective/belief architecture. It strengthens how that architecture is trained and evaluated at larger experience counts.

## Robust sampled-future planning

`RiskAwarePolicySeededMPPI` evaluates candidate action sequences over multiple samples from Aether's multimodal world model. Candidate utility combines expected return with lower-tail CVaR and optional epistemic-disagreement penalties. This avoids planning only through an averaged future that may never occur.

## Epistemic uncertainty

`WorldModelEnsemble` separates model disagreement from within-model stochasticity. `train_bootstrap_world_ensemble` trains multiple dynamics models on bootstrap resamples while freezing the shared temporal-belief encoder, so disagreement stays in one coordinate system.

## Hybrid actions

`HybridGameActionCodec` treats movement as continuous and attack/interact as binary decisions without breaking the existing 4-D environment ABI.

## Scalable replay

`ShardedReplayStore` writes append-only compressed shards with an atomic JSON index and SHA-256 integrity checks. It is intended as the next step beyond monolithic NPZ files for millions of transitions.

## Iterative data aggregation

`IterativeDataAggregator` supports actor/planner mixtures over repeated rounds and accepts a retraining callback so planner-corrected on-policy experience can feed the next policy revision instead of remaining a one-shot offline dataset.

## Qualification statistics

Bootstrap and paired-bootstrap confidence intervals plus `summarize_seed_metrics` make multi-seed reporting first-class.

## Boundary

The new modules improve experimental validity and scale-readiness. The bundled smoke tests do not establish that Aether is already a strong game-playing agent; that requires long-run multi-seed training against external baselines under matched compute.
