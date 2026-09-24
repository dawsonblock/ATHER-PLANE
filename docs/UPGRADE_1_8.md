# v1.8 Upgrade Notes

## Distributional continuous control

Set `training.continuous_q.critic_type: quantile_twin_q` to use twin quantile critics. `quantiles` controls support resolution and `n_step` controls the real replay target horizon. Scalar `twin_q` remains supported.

## Planner-benefit qualification

`awa-gate-fit` can now measure actor and planner branches over multiple real environment transitions from identical snapshots. This is more expensive but captures delayed planning benefit missed by one-step labels.

## Sequence caches

`CachedSequenceAdapter` now prefers backend-native incremental cache hooks when available and otherwise falls back to bounded context recomputation. The public cache contract is unchanged.

## Environment qualification

`awa-env-multiseed` reports per-seed failures and snapshot round-trip determinism instead of treating importability as successful qualification.
