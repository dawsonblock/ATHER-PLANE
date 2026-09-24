# Aether v2.10 — Real Adaptation Qualification

v2.10 closes the highest-priority integration gaps identified in the v2.9 game-lab audit. The release does not claim that Aether has mastered unseen games. It makes the training and evaluation path faithful enough that those claims can now be tested.

## Objective semantics

Curriculum stage is metadata only. `TaskSpec.goal.objectives` is executed by `GoalProgram`, and the policy receives a separate 13-D goal vector. Compositional/task OOD therefore changes executable prerequisites rather than only benchmark labels.

The built-in stage mapping includes chains such as `collect_key -> open_door -> reach_goal` and the full stage-12 chain. Reward shaping follows the active prerequisite.

## Valid procedural worlds

Critical entities are sampled in collision-free space connected to the player's free-space component. This prevents impossible episodes caused by spawning goals, keys, enemies, or objects inside obstacles or disconnected regions.

## Temporal belief path

The actual game trainer now uses `GameBeliefEncoder`: observation encoding, local GRU state, global Transformer context, and a goal embedding. World dynamics, risk, actor, planner, and runtime controller operate in the resulting belief space. Observation and goal reconstruction losses keep the belief grounded.

## Planner terminal bootstrap

The old transition-trunk `value` output is no longer used by planners. v2.10 trains a dedicated `terminal_value` head against discounted return-to-go. Legacy checkpoints that predate this head load with a zero terminal bootstrap instead of a random one.

## Reusable experience reaches SGD

Structural replay priority is exported as normalized `sample_weights`; game-specific hindsight relabeling rewrites both observation goal coordinates and the explicit goal vector; exact snapshot counterfactuals are isolated as one-step episodes. The game trainer consumes these rows through weighted world/actor losses and a one-step auxiliary pass.

## Adaptation qualification

Transfer modes are explicit:

- zero-shot: policy context resets every episode;
- in-context: weights remain frozen and temporal context may persist across repeated exposures to the same task;
- learning: episode context resets and an optional adaptation hook may update the learner between exposures.

Task boundaries always reset temporal state. Prediction error is measured from model/policy predictions rather than inserted as zero.

## Runtime hierarchy and VOC

`AdaptiveGamePolicy` uses executable qualified skills first, then the actor, then a VOC-selected planner when expected benefit exceeds compute cost. `fit_game_voc` can train VOC targets from exact real-environment snapshot branches, measuring return gain, latency, and world-model calls from identical starting states.

## Qualification boundary

The release validates contracts and end-to-end execution. Strong transfer/adaptation still requires real multi-seed training runs with held-out tasks, layouts and dynamics. Do not treat the bundled smoke run as a capability benchmark.
