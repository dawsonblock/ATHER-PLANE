# Aether v2 Build Plan

The v2 program is evidence-gated. New components stay optional until they improve capability or efficiency on reproducible tasks.

## Completed milestones

### v2.0 — architecture foundation
Policy-first decision system, stochastic world model, separated risk/uncertainty, multiple planner backends, value of computation, memory and hierarchy.

### v2.1 — representation foundation
Frozen pretrained backbone injection, deterministic feature cache, offline transition contract and provenance.

### v2.2 — world-model qualification
Episode-safe cached sequences, multimodal dynamics training, horizon metrics, uncertainty/risk calibration and promotion receipts.

### v2.3 — planner qualification
Actor-only baseline, clean planner controls, exact-state counterfactual branching, measured planner benefit and VOC fitting.

### v2.4 — checkpoint closed-loop qualification
Strict saved-artifact reconstruction and paired multi-seed compute-normalized actor/planner evaluation.

### v2.5 — game-video perception
Causal episode-safe V-JEPA 2/2.1 clip encoding, content-addressed frozen features, telemetry fusion, streaming/offline parity and reuse of the existing world/actor/planner training stack.

### v2.6 — reusable-learning infrastructure
Procedural task synthesis, learning-frontier curriculum, structural replay, causal branching, reasoning-effort control, heterogeneous teacher distillation, skill discovery/composition, EngramLite and rollout staleness semantics.

### v2.7 — closed-loop transfer and continual learning
1. Give synthesized tasks deterministic verifier identities.
2. Separate exploit/explore/qualification reward semantics.
3. Relabel achieved outcomes into additional goal-conditioned supervision.
4. Track competence by reusable concept and synthesize new tasks around weak concepts.
5. Distill promoted trajectory clusters into executable bounded continuous skill policies.
6. Learn skill precondition/termination/failure contracts.
7. Coordinate curriculum, replay, skills, memory and distillation in a resumable learning engine.
8. Export analyzed latent replay back into the standard Aether offline dataset contract.
9. Collect local simulator rollouts through versioned/staleness-compatible envelopes.
10. Measure seen, visual, layout, dynamics, compositional and task OOD adaptation separately.

### v2.8 — hardening and archive correctness
1. Reproduce defects against the packaged v2.7 archive rather than only the working tree.
2. Persist executable skill policies/contracts exactly across checkpoint restore.
3. Make curriculum generation fresh and resumable.
4. Normalize Gymnasium/Gym branching and rollout ABIs.
5. Fail closed when neither proposal nor recovery is acceptably safe.
6. Make replay export robust to hindsight augmentation and interleaved async episodes.
7. Add permanent regression tests for every reproduced defect.

## Next: v2.9 real transfer qualification
1. Connect a real game/simulator that supports deterministic reset/snapshot or reproducible seeds.
2. Train a v2.7 agent on procedurally varied tasks across multiple seeds.
3. Freeze weights and measure in-context adaptation to changed physics/control/reward rules.
4. Compare v2.7 against v2.6, actor-only and always-plan controls.
5. Measure first-episode success, adaptation AUC, episodes-to-target and planner-dependency decay.
6. Require skill policies to pass learned contracts and independent success/risk qualification before runtime promotion.
7. Promote only mechanisms whose marginal contribution survives ablation.

## Later

### v2.10 — controlled modular scaling
- monolithic vs small MoE at matched active compute
- data/model scaling curves
- long-context scaling

### v2.11 — semantic planning
- typed VLM/LLM subgoals
- world-model feasibility verification
- no direct actuator authority

## Promotion gates
A component remains off by default unless it demonstrates one or more of:
- >=5% task-success improvement
- >=15% sample-efficiency improvement
- >=20% robustness improvement
- >=20% compute reduction
- and no >3% critical benchmark regression

Planner changes must additionally improve real return/success per world-model call or per millisecond.


### v2.9 — procedural game lab
- built-in deterministic game-like environment across all 12 reusable-learning curriculum stages
- dataset collection and reusable-game loop
- end-to-end structured game world/actor/planner training
- isolated held-out OOD game qualification

### v2.10 — adaptation and hierarchical control
- frozen-weight adaptation curves on changed rules
- learned reachability-aware subgoal execution
- planner dependency reduction through skill reuse
