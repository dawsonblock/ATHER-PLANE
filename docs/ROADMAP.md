# Aether v2 Roadmap

## v2.0 — architecture reset (complete)
- policy-first runtime
- local/global belief state
- multimodal stochastic world model
- horizon-aware uncertainty
- explicit risk model
- multiple planner backends + value-of-computation
- memory and hierarchy primitives

## v2.1 — representation/offline data (complete)
- injectable frozen pretrained backbones
- deterministic content-addressed feature cache
- offline transition datasets + SHA-256 provenance

## v2.2 — world-model qualification (complete infrastructure milestone)
- episode-safe sequence training
- multimodal future prediction + overshooting
- horizon metrics and uncertainty calibration
- explicit risk training + promotion gates

## v2.3 — planner qualification (complete infrastructure milestone)
- strong actor-only baseline
- vanilla and policy-seeded planner controls
- exact counterfactual branching
- measured planner benefit + VOC training
- fail-closed action guard

## v2.4 — saved-artifact closed-loop qualification (complete infrastructure milestone)
- strict world/actor checkpoint reconstruction
- paired multi-seed closed-loop benchmark reports
- compute-normalized planner scorecards
- checkpoint provenance and bootstrap intervals

## v2.5 — pretrained game-video perception (complete infrastructure milestone)
- Hugging Face V-JEPA 2 frozen video backbone
- local-first V-JEPA 2.1 PyTorch-Hub backbone
- causal episode-safe game clips
- visual + telemetry frozen feature fusion
- offline/streaming feature parity
- game cache directly consumable by the existing world-model trainer

## v2.6 — reusable learning engine (complete infrastructure milestone)
- automated procedural task/environment generation
- learning-frontier curriculum scheduling
- structural replay priority from prediction error, novelty, failure, TD error and information value
- exact counterfactual causal branching
- controllable reasoning effort / planner budget selection
- heterogeneous teacher-to-actor distillation
- automatic skill discovery, registry and composition
- sparse Engram-lite pattern memory
- bounded context-state replay and rollout staleness controls

## v2.7 — closed-loop transfer and continual learning (complete infrastructure milestone)
- verifiable synthesized tasks
- hindsight goal relabeling
- exploit/explore/qualification separation
- learned skill contracts
- explicit visual/layout/dynamics/compositional/task OOD metrics
- concept-level forgetting diagnostics
- resumable reusable-learning engine and replay export
- local versioned rollout worker pool

## v2.8 — hardening and archive correctness (complete infrastructure milestone)
- resumable executable-skill and contract checkpoints
- fresh adaptive curriculum generation across restarts
- Gymnasium-compatible exact branching and rollout collection
- fail-closed recovery semantics
- async replay episode regrouping and hindsight-safe world-model export
- permanent regression coverage for reproduced v2.7 defects

## v2.9 — procedural game lab and end-to-end qualification (complete infrastructure milestone)
- deterministic 12-stage game curriculum with structured and RGB observations
- chain-consistent gameplay dataset generation
- live integration with reusable-learning replay/hindsight/skills
- end-to-end world-model + risk + actor + MPPI training smoke
- isolated visual/layout/dynamics/compositional/task OOD game tasks
- dynamics parameters hidden from policy observations by default

## v2.10 — real hierarchy/adaptation qualification (complete infrastructure milestone)
- executable objective programs and validated procedural worlds
- goal-conditioned temporal-belief training
- weighted structural replay / hindsight / counterfactual SGD path
- trained terminal planner bootstrap and game-specific VOC qualification

## v2.11 — empirical scaling and robust planning (complete infrastructure milestone)
- sampled-future/CVaR planning
- bootstrap dynamics ensembles and epistemic uncertainty
- hybrid-action codec
- sharded replay storage
- iterative planner-assisted data aggregation
- multi-seed confidence intervals

## v2.12 — controlled modular scaling (complete infrastructure milestone)
- monolithic vs small MoE at matched active compute
- data/model scaling curves
- long-context scaling
- multi-seed confidence scorecards and resumable experiment ledgers

## semantic planning — deferred until empirical gate
- typed VLM/LLM subgoals
- world-model feasibility verification
- no direct actuator authority

## v3.0 gate
Do not aggressively scale parameter count until Aether beats or materially complements strong actor-only, Dreamer-style and TD-MPC-style references on declared game/control tasks under matched compute and reproducible seeds.

## v2.13 — durable training campaigns (implemented)

- cumulative 100k -> 250k -> 500k -> 1M replay targets
- hardware-aware training profiles and explicit resource caps
- append-only replay with schema/integrity checks and materialization
- atomic stage resume, promotion gates and stable-checkpoint rollback history
- campaign compute/update estimates with optional measured-throughput wall-clock projection
- plan-only default CLI plus explicit `--execute` for expensive runs

Next evidence gate: run the default campaign on real hardware and use the resulting learning curves to decide whether v2.12 MoE/context scaling is justified. Distributed learner/actor separation and pixel-heavy campaigns remain later milestones.

## v2.14 — empirical qualification and real-game bridge (complete infrastructure milestone)
- versioned external game bridge protocol
- process-isolated rollouts and telemetry
- structured/pixel benchmark isolation
- failure triage, ablations and milestone reports

## v2.15 — ViZDoom + V-JEPA 2 integration (complete infrastructure milestone)
- direct ViZDoom structured and RGB adapters
- stable hybrid action mapping and fixed telemetry/goal ABI
- ViZDoom dataset collection with provenance
- frozen V-JEPA 2 causal clip caching and representation contracts
- structured-vs-pixel parity path for the same scenario/seed families

Next evidence gate: collect and train My Way Home structured and pixel baselines, then progress through combat/tactics scenarios only if memory/navigation learning curves are credible.
