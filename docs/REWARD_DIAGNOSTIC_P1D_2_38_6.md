# P1D paired reward-scale diagnosis

P1D uses a single real structured `my_way_home` collection for both arms. It
trains the same small world/actor stack three times per arm with paired training
seeds, then evaluates actor-only policies on the same eight held-out episode
seeds. The only training-data change is identity reward versus clipping the
scalar reward to `[-1, 1]`. It does not continue the main campaign checkpoint.

Run it after P1 passes:

```bash
awa-v2-vizdoom-reward-diagnostic \
  --config configs/v2_38_6_reward_diagnostic.yaml \
  --out-dir runs/v2_38_6/reward_diagnostic \
  --execute
```

The collector requires native ViZDoom, writes the shared dataset and both
transformed datasets, records checkpoints and training reports, and logs action
distributions/entropy for every held-out episode. Dataset metadata includes the
scenario ID, scalar reward distribution, success/death/timeout labels, and
telemetry-derived health/displacement proxies. Those proxies are explicitly
labeled; they are not asserted to be the exact internal components of Doom's
reward function.

The gate passes only if clipped reward has higher aggregate held-out navigation
success and improves success in at least two of the three paired training
seeds. If it fails, P2 remains blocked. Lower training loss alone cannot pass
P1D.
