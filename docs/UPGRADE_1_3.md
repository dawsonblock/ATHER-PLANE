# v1.3 upgrade notes

v1.3 turns several v1.2 interfaces into active training/runtime features.

## Prioritized sequence replay

Sampling remains episode-aware and never crosses terminal boundaries. Sequence priorities are based on model error and use proportional prioritization with importance weights.

## Sparse dynamics experts

World-model heads can use top-k residual experts. Routing can later be conditioned on a task embedding. A small load-balancing loss prevents immediate expert collapse.

## Full-state checkpoints

Bundle checkpoints save world model, actor, uncertainty estimator, all optimizers, RNG state, current belief, environment state, current observation/action and optionally replay contents.

## Robustness testing

`awa-robustness` sweeps observation noise and sensor dropout. This is intended to become part of the acceptance gate rather than a one-off demo.

## External adapters

Large pretrained perception and semantic models remain optional. The core still runs offline with no model downloads.
