# Aether v2 Architecture

Aether v2 is a policy-first world-model agent. Search is optional compute, not the default execution path.

Core split:

1. Representation: pretrained/frozen encoder + small adapter.
2. World: local temporal core + slower global context + multimodal stochastic future model.
3. Decision: actor/executable skills first; planner selected only when expected value of computation is positive.
4. Safety: predictive uncertainty and explicit outcome risk are separate signals; high-risk actions go through a fail-closed guard.
5. Memory: bounded recent context, episodic retrieval and sparse EngramLite reusable patterns.
6. Reusable learning: verifiable procedural tasks, structural replay, counterfactual branching, hindsight goal relabeling and multi-teacher distillation.
7. Skills: discovered trajectory motifs become dedicated bounded policies; symbolic applicability and learned precondition/termination/failure contracts gate execution.
8. Continual learning: concept coverage drives adaptive practice tasks and six explicit OOD categories measure transfer.
9. Hierarchy: semantic subgoals -> validated skills -> local controller/search.
10. Biology: MaleCNS stays in a separate hypothesis-testing lane.

Planner policy:

- familiar state: actor/skill
- local continuous correction: policy-seeded MPPI or iCEM
- differentiable difficult trajectory: gradient planner
- abstract discrete/subgoal problem: beam/tree planner
- uncertain state with information value: observe/explore before committing

Training policy:

- exploit: task reward only, actor/skills preferred
- explore: bounded novelty/information/surprise bonuses may shape data collection
- qualify: task reward only; no intrinsic bonuses, no training-only reward shaping

Every planner comparison should be performed at matched wall-clock or world-model-call budgets. Every transfer claim should report held-out category, adaptation curve and planner dependency rather than only final reward.
