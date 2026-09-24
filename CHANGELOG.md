## 2.38.2 — Bounded V-JEPA memory and campaign storage guard

- Streams V-JEPA clips through bounded batches instead of retaining the full cache-miss set in RAM.
- Sets the shipped V-JEPA batch size to 1 for the 24 GB GPU target.
- Requires 100 GiB of free persistent workspace before bootstrap/P0, with a RunPod 300 GB volume sizing guide.
- Includes the v2.38.1 scenario-paired promotions, fail-closed P2 gate, and P3 checkpoint hash checks.

## 2.38.1 — Real execution correction

- Paired incumbent and candidate ViZDoom promotion by scenario and reset seed.
- Fail-closed stage promotion and minimum success evidence for P2/P3/P8; fresh results required after the v2.38.0 P2 failure.
- Executor checks the evidence gate after each command and refuses cached v2.38.0 cross-scenario training decisions.
- P9 resolves only verified P3 actor/world checkpoints with matching SHA-256 hashes.
- Frozen action-time runtime and original v2.38.0 artifact remain untouched.

## 2.38.0 — Progressive Execution Orchestrator

- Added `awa-v2-progressive-executor`, which plans or executes exactly the first blocked capability phase.
- Added shell-free command execution with an explicit command allowlist and atomic phase receipts.
- Added transition-target guards to prevent accidental escalation to larger training stages.
- Added focused five-seed 25K temporal-memory, fixed-planner, and adaptive-compute ablation configs.
- Added automatic stable-checkpoint resolution for the physical GPU planner schedule benchmark.
- Added a second explicit acknowledgement before the canonical expensive five-seed DREAM-RSI campaign.
- Added `deploy/runpod/bootstrap.sh` and `awa-v2-runpod-bootstrap` for CUDA-host bring-up without auto-starting training.
- Preserved manual evidence boundaries for world-model horizon measurements, compositional transfer, unseen-map transfer, external baselines, and 250K+ scale authorization.
- Kept `awa.v2.agent_runtime` Python source byte-identical to v2.37.
- No new empirical capability result is claimed by this software release.

## 2.37.0 — Progressive Capability Curriculum

- Added an ordered fail-closed capability ladder that prevents later experiments from bypassing missing prerequisites.
- Added held-out world-model horizon qualification with contiguous-prefix reliability semantics.
- Separated world-model reliability from realized planner benefit as independent promotion gates.
- Added real V-JEPA, physical planner schedule, compositional, DREAM-RSI, transfer, baseline, and scale gates.
- Kept the canonical Agent Runtime source-identical to v2.36.
- No new empirical capability result is claimed by this software release.

# Changelog

## 2.36.0

- Added fail-closed real-execution evidence bundling with content-addressed preflight, training, stable-checkpoint, and planner artifacts.
- Real ViZDoom receipts now expose resume transition deltas and verify checkpoint-registry integrity.
- Added a matched external-baseline comparison contract with seed/milestone/track pairing, transition-budget checks, accelerator-budget bounds, and paired bootstrap intervals.
- Kept the action-time agent architecture unchanged.

# v2.35.0 — Real Execution Readiness

- Adds fail-closed GPU/dependency/disk/output preflight with an environment fingerprint.
- Adds a native-only real ViZDoom training campaign wrapper; injected/fake Doom backends are rejected before evidence collection.
- Requires non-injected Hugging Face V-JEPA components for real pixel campaigns.
- Adds small real structured (2K→10K→25K) and pixel/V-JEPA (1K→5K) bring-up configs to expose hardware/runtime failures before expensive scaling.
- Extends planner hardware benchmarking with numerical action-equivalence checks, throughput metrics, and GPU/Torch provenance.
- Extends the empirical maturity report to consume real ViZDoom training receipts while keeping multi-seed empirical qualification separate.
- Adds `awa-v2-execution-preflight` and `awa-v2-vizdoom-real-campaign`.
- Keeps the v2.34 action-time agent and DREAM-RSI empirical protocol unchanged; no new cognitive mechanism is added.

# v2.34.0 — DREAM-RSI Empirical Qualification

- Adds a resumable paired fixed-vs-DREAM campaign runner over independent seeds and recursive rounds.
- Adds a hard `transition_budget_cap` alongside the worker budget so selected meta-decisions cannot exceed the frozen real-environment ceiling.
- Continues both arms from the best grounded checkpoint produced in the prior round.
- Adds optional development heldout/transfer evaluation to grounded allocation outcomes.
- Adds a disjoint final heldout/transfer meta-test that is not used for replay-policy improvement.
- Adds paired bootstrap confidence intervals, worst-seed regression checks, transition accounting, and wall-time accounting.
- Adds content/hash-verified round resume, policy/menu/lineage validation, and deterministic records.
- Adds `awa-v2-dream-rsi-campaign` and `awa-v2-dream-rsi-campaign-smoke`.
- Keeps the action-time `agent_runtime` unchanged and makes no new cognitive claim.

# v2.33.0 — Recursive DREAM-RSI Closure

- Closed the v2.32 DREAM-RSI wiring gaps with exact-transition real allocation execution.
- Added complete offered decision-menu logging and fail-closed unsupported replay choices.
- Added a true worker budget over `sum(parallel_worlds)`.
- Added multi-round replay-policy development with per-revision feedback receipts.
- Added recursive online-world ingestion/replay-pool expansion.
- Added exact-paper and Aether replay-objective modes.
- Added a controlled fixed-vs-DREAM comparison contract.
- Added `awa-v2-dream-rsi-loop-smoke`.
- Action-time `agent_runtime` remains unchanged.

# v2.32.0 — Grounded DREAM-RSI Meta-Learning

- Keeps the action-time agent runtime frozen; the new mechanism operates only on training allocation.
- Adds one declarative `TrainingAllocationDecision` interface for both real execution and historical replay.
- Adds grounded historical replay worlds that reject model-simulated/counterfactual outcomes.
- Adds support-aware replay scoring and `ONLINE_PROBE_REQUIRED` handling for weak historical coverage.
- Adds a frozen `DreamRSIContract` binding the underlying agent, evaluator, and decision-interface version.
- Adds bounded declarative meta-policy mutation with incumbent protection; no arbitrary source-code rewriting.
- Adds paired grounded online qualification; replay can propose a policy but can never promote it directly.
- Adds `awa-v2-dream-rsi-smoke`, `configs/v2_32_dream_rsi_meta_learning.yaml`, integration audit, docs, and regression coverage.
- Preserves the v2.31 Agent Runtime / Learning System / Research OS split and the unexecuted empirical boundaries.

# v2.31.0 — Proper Agent / Learning / Research Split

- Adds three canonical lazy public surfaces: `awa.v2.agent_runtime`, `awa.v2.learning_system`, and `awa.v2.research_os`.
- Enforces one-way dependency intent: agent runtime cannot depend on learning or research governance; learning may consume runtime; research OS may inspect both.
- Adds `awa-v2-split-check`, a fail-closed AST import-boundary validator for the canonical split packages.
- Adds `configs/v2_31_system_split.yaml` with explicit responsibilities and forbidden responsibilities for all three layers.
- Keeps historical flat `awa.v2.*` modules as compatibility implementation details while the empirical v2.30 baseline remains frozen.
- Keeps skill discovery, Engram, sparse-MoE/modular-world, connectome, reusable-learning-engine, and legacy prototypes outside the canonical agent.
- Adds `SYSTEM_SPLIT_2_31.json`, `docs/UPGRADE_2_31.md`, `docs/REMAINING_WORK_2_31.md`, and split-boundary regression tests.
- Adds no new cognitive mechanism and does not upgrade any unexecuted empirical claim.

# v2.29.0 — Compute-Aware Planning + Physical Execution Accounting

- Separates logical imagined world-model transitions from physical batched `world.imagine_step` forward invocations.
- Adds deterministic planner microbatch scheduling through `planner_world_batch_size`; `0` means maximally batched and positive values cap physical world-model batch size.
- Adds execution fingerprints with physical forward count, batch-size histogram, max/mean batch size, and logical-transitions-per-forward.
- Extends adaptive-compute runtime traces with physical forward and batch-size telemetry.
- Adds campaign metrics for physical world-model forwards, logical model work, mean world-model batch size, accelerator wall-hours, effective training rows, and peak CUDA memory.
- Adds `compute_schedule_report.json` and integrates physical-forward/logical-work reductions into the preregistered marginal-value decision contract.
- Keeps energy reporting fail-closed: no energy is inferred from elapsed time; `accelerator_energy_joules` remains null unless directly measured.
- Adds `configs/v2_29_compute_efficiency.yaml`, `docs/UPGRADE_2_29.md`, `docs/REMAINING_WORK_2_29.md`, `COMPUTE_SCHEDULE_AUDIT_2_29.json`, and v2.29 regression tests.
- Preserves the v2.28 70-training-job / 140-evaluation-cell ablation matrix and train-once/evaluate-many evidence contract.

# v2.28.0 — Resumable Ablation Campaign + Decision Closure

- Adds a frozen 70-training-job / 140-evaluation-cell campaign for the seven executable v2.27 component variants.
- Trains once per `(system, seed, milestone)` and evaluates the same frozen checkpoint on heldout and transfer, eliminating retrain-per-split confounding.
- Adds exact shared dataset prefixes across 25K/100K milestones and shared fixed-curriculum datasets across comparable variants.
- Adds atomic attempt directories, job receipts, content hashes, O_EXCL locks, stale-lock recovery, deterministic worker sharding, and safe resume after preemption.
- Adds consolidated `records.jsonl`, provenance, experiment-protocol receipts, and a machine-readable Markdown/JSON ablation decision report.
- Adds adjacent marginal analysis for memory, world modeling, fixed planning, adaptive compute, grounded hindsight replay, and adaptive curriculum.
- Fails closed with `INSUFFICIENT_EVIDENCE` until the full preregistered matrix and minimum seed count are present; no premature deletion recommendation is emitted.
- Adds `evaluate_trained_variant()` so secondary splits are evaluated from the already-trained checkpoint rather than retraining.
- Adds `awa-v2-ablation-campaign`, `awa-v2-ablation-report`, and `awa-v2-ablation-campaign-smoke`.
- Adds `configs/v2_28_ablation_campaign.yaml`, `docs/UPGRADE_2_28.md`, `docs/REMAINING_WORK_2_28.md`, and v2.28 regression coverage.

# v2.27.0 — Executable Ablation + RNG Resume Closure

- Converts the v2.26 preregistered ablation labels into concrete runtime variants.
- Adds real raw-state `actor_only` and temporal-reconstruction `belief_actor` baselines instead of relabeling the full stack.
- Adds fixed-planner and real-branch VOC adaptive-compute evaluation paths over frozen world-model checkpoints.
- Narrows the previous `reusable_learning` ablation to `reusable_replay`, using grounded future-goal hindsight rows with lower structural weight.
- Adds `full_curriculum`, which requires an adaptive-curriculum dataset contract rather than silently running on fixed data.
- Adds process RNG checkpoint/restore for Python, NumPy, PyTorch CPU and all CUDA devices when available.
- Adds `awa-v2-ablation-run` and `awa-v2-ablation-runtime-smoke`.
- Adds `configs/v2_27_executable_ablation.yaml`, `ABLATION_MATRIX_2_27.json`, and v2.27 upgrade/remaining-work documentation.
- Keeps dormant skill/Engram/MoE/ensemble families outside the stable ablation claim until they earn separate executable evidence.

# v2.26.0 — Autonomous Collection + Ablation Preregistration

- Adds `CoverageArenaPolicy` and `FrozenAetherArenaPolicy` so native procedural collection can be teacher, random, coverage, frozen actor, or frozen actor+planner under one dataset ABI.
- Canonical autonomous collection fails closed into an explicit coverage bootstrap when no grounded parent checkpoint exists; later cumulative milestones use the frozen parent Aether checkpoint.
- Records requested/actual collector identity plus planner/world-model-call telemetry in collection evidence.
- Makes cumulative world-model resume optimizer-complete by checkpointing AdamW state, CUDA GradScaler state when active, and global update count.
- Extends experiment preregistration with per-system configuration SHA-256 bindings.
- Adds the seven-system empirical ablation specification and `awa-v2-ablation-protocol`.
- Adds `awa-v2-autonomous-collection-smoke`, `configs/v2_26_autonomous_empirical_closure.yaml`, v2.26 docs, and regression tests.
- Preserves v2.25 cumulative milestone cost reduction, train-once/evaluate-many qualification, execution-equivalent meta-policy deduplication, and lazy package loading.

# v2.25.0 — Reduction + Cumulative Milestone Training

- Converts native 25K→100K→250K→500K→1M milestones into a grounded cumulative lineage: each milestone warm-starts from the prior checkpoint and collects/trains only on the delta.
- Shares the content-addressed training cache across bootstrap and later meta-iterations so unchanged active baselines are reusable.
- Records parent checkpoint/identity, cumulative and delta dataset hashes, incremental transitions, and incremental/cumulative training cost.
- Fails closed when a matching cumulative parent artifact is corrupt.
- Makes native meta-policy search execution-aware: only curriculum-relevant axes mutate by default and policies with identical integer stage allocations are deduplicated.
- Sets the canonical native search to 24 tasks/batch and a 0.20 mutation step for less-quantized real curriculum changes.
- Disables planner diagnostics inside routine native training; planner/VOC value remains a separate qualification problem.
- Adds warm-start support to `train_game_stack()` and complete actor/critic state restoration.
- Makes `awa.v2` and `awa.v2.game` lazy public surfaces, reducing plain-import startup cost and memory without removing public convenience names.
- Adds `configs/v2_25_reduction_empirical_closure.yaml`, `docs/UPGRADE_2_25.md`, `docs/REMAINING_WORK_2_25.md`, and v2.25 regression coverage.

# v2.24.0 — Milestone Empirical Closure

- Replaces flat all-cell promotion averaging with a paired milestone scorecard.
- Adds explicit final-milestone paired-gain qualification so early wins cannot hide a worse final model.
- Adds transition-normalized learning-curve area as a sample-efficiency signal.
- Adds heldout/transfer final split guardrails and retains worst-seed regression protection.
- Writes content-addressed `milestone_scorecard.json` receipts for executed evolving iterations.
- Adds `awa-v2-milestone-scorecard` for standalone paired empirical analysis.
- Adds the canonical five-seed 25K→100K→250K→500K→1M campaign config.
- Preserves v2.23 train-once/evaluate-many artifacts, v2.22 native grounded execution, and all prior regression surfaces.

# v2.23.0 — Train Once, Evaluate Many

- Separates native procedural training identity from evaluation-cell identity.
- Adds a content-addressed training-artifact cache keyed by source, dependencies, policy, seed, milestone, task family and training configuration.
- Trains each policy/seed/milestone once and reuses the exact dataset/world/actor artifacts across heldout and transfer qualification cells.
- Adds hash verification and fail-closed handling for incomplete or corrupted cached training artifacts.
- Moves qualification to an evaluation-only runtime so heldout/transfer tasks are never passed into the training routine.
- Adds `load_procedural_game_stack()` and `evaluate_procedural_actor()` for strict frozen-checkpoint evaluation.
- Adds stable logical-vs-physical cost accounting so cache execution order cannot alter replay-policy scoring.
- Updates the native smoke to prove same-checkpoint/same-dataset reuse with disjoint heldout/transfer scenarios.
- Adds `configs/v2_23_train_once_evaluate_many.yaml`, v2.23 documentation, and new regression coverage.

# v2.22.0 — Native Closed-Loop Execution + Load-Bearing Adaptive Compute

- Adds `NativeProceduralCampaignRunner` for real built-in procedural collection, training, held-out/transfer evaluation, and content-addressed evidence.
- Adds deterministic `ExplorationPolicyCurriculumAdapter` so declarative meta-policy changes have an explicit, auditable effect on training-task allocation.
- Adds empty-history `NEEDS_BOOTSTRAP` planning and explicit `bootstrap_active()` execution; bootstrap creates grounded history but never promotes.
- Extends the generic evolving-campaign CLI to support `runner.type=native_procedural` and `--bootstrap-active`.
- Makes `AdaptiveReasoningEffortController` load-bearing inside `AdaptiveGamePolicy` when trained VOC gains are available.
- Masks planner/budget effort levels absent from the trained VOC instead of inventing gains for them.
- Adds per-decision `EffortRuntimeTrace` with chosen compute level, predicted gain, risk/uncertainty, measured latency, and world-model calls without fabricating realized gain.
- Extends procedural game qualification reports with observed constraint violations, inference latency, planner calls, and world-model calls.
- Adds `configs/v2_22_native_evolving_campaign.yaml`, v2.22 documentation, and regression coverage.

# v2.21.0 — Evolving Campaign Closure

- Adds `EvolvingCampaignOrchestrator`, closing history -> replay -> proposal -> exact paired online validation -> protocol qualification -> promotion.
- Converts empirical run matrices into branching replay worlds per seed/split, with system/task root branches and transition milestones as branch depth.
- Adds exact `PairedCampaignPlan` cells across active/candidate × seeds × tasks × splits × milestones.
- Adds atomic `CampaignEvidenceStore` with idempotent resume and conflict rejection.
- Adds `SubprocessCampaignRunner` with explicit argv templating and `shell=False`; runner outputs must satisfy the v2.21 evidence ABI.
- Composes every online proposal with the v2.19 experiment protocol and v2.18 provenance/split-leakage checks before promotion is considered.
- Extends the v2.19 protocol receipt helper with an explicit baseline while preserving `full` as the default.
- Adds `awa-v2-evolving-campaign`, `awa-v2-evolving-campaign-smoke`, `configs/v2_21_evolving_campaign.yaml`, and v2.21 regression coverage.
- Preserves the no-arbitrary-source-rewrite boundary and the rule that replay/model-simulated evidence cannot promote a policy.

# v2.20.0 — Evolving Exploration + Adaptive Compute

- Adds grounded macro-level discovery trees and historical replay worlds derived from the DREAM-RSI exploration pattern.
- Adds bounded declarative meta-policy search for branch/continue/stop allocation without arbitrary source-code execution.
- Adds explicit observed/model-simulated/counterfactual/validated evidence classes; only grounded evidence may support promotion.
- Adds novelty and adversarial exploration reserve floors to reduce replay-support collapse.
- Adds paired online promotion gates requiring five grounded seeds by default; replay can propose but cannot promote.
- Adds adaptive reasoning-effort allocation across actor/shallow/medium/deep/strategic budgets using gain, compute, latency, risk, resource pressure, and model reliability.
- Adds an effort outcome ledger for planner-dependence and realized-gain training evidence.
- Preserves the DeepSeek-specific CED/CSA2/FP4/SWA mechanisms as serving/model concerns rather than transplanting them into Aether's non-Transformer world model.
- Fixes package-version drift so `pyproject.toml`, `awa.__version__`, and `awa.v2.__version__` agree on 2.20.0.
- Adds `awa-v2-evolving-exploration-smoke`, `configs/v2_20_evolving_exploration.yaml`, and v2.20 regression tests.

# v2.19.0 — Preregistered Experiment Protocol Closure

- Adds a content-addressed preregistered experiment protocol for exact systems × seeds × tasks × splits × milestones.
- Fails closed on missing, unexpected, or duplicate empirical cells to prevent selective reporting.
- Adds paired-baseline completeness checks and metric-domain validation.
- Composes protocol qualification with v2.18 provenance, artifact-hash, and split-leakage integrity receipts.
- Adds a canonical five-seed ViZDoom heldout/transfer protocol through the 25k -> 1M milestones.
- Preserves the complete v2.18 evidence-integrity and v2.17 empirical-closure stack.

# v2.18.0 — Evidence Integrity Closure

- Adds content-addressed run provenance for source, dataset, environment, dependency lock, checkpoint, and config.
- Adds deterministic canonical evidence receipts.
- Rejects duplicate empirical run keys.
- Adds explicit train/validation vs heldout/transfer scenario-leakage detection.
- Fails qualification when provenance or artifact hashes are absent/malformed.
- Preserves the complete v2.17 empirical-closure stack and all prior Aether functionality.

# Changelog

## 2.17.0

- Shifted the release objective from architecture expansion to empirical closure.
- Added a versioned run-record ABI for multi-seed, milestone, held-out and transfer evidence.
- Added fail-closed qualification gates requiring five seeds per compared system, complete operational metrics, and held-out/transfer evidence.
- Added paired-seed bootstrap deltas against the full system without converting statistical evidence into unsupported capability claims.
- Added planner-dependence curves to test whether expensive deliberation decreases as reusable policy competence grows.
- Added a single evidence-bundle builder and `awa-v2-empirical-closure` CLI for reproducible qualification artifacts.
- Added `configs/v2_17_empirical_closure.yaml` defining the 25k -> 1M milestone matrix and core baseline/ablation systems.
- Preserved all v2.16 ViZDoom/V-JEPA training-closure functionality.

## 2.16.0

- Unified ViZDoom structured telemetry and materialized V-JEPA+telemetry features with the goal-conditioned GRU/global-context belief trainer.
- Generalized `GameBeliefSequenceDataset` so non-procedural game observation/goal dimensions can use the same temporal training path.
- Added `HybridGameActor` and `HybridOfflineActorCriticBaseline`: continuous turn/forward plus categorical attack/use with explicit BCE supervision.
- Added `HybridRiskAwarePolicySeededMPPI`; every world-model rollout and returned action keeps attack/use exactly `-1/+1`.
- Added scenario-aware ViZDoom success and four-channel constraint semantics instead of one generic reward-based success rule.
- Added batched Hugging Face V-JEPA inference with compatibility fallback and batched content-addressed cache filling.
- Added `materialize_vizdoom_cached_features` to preserve goals, rewards, constraints and provenance while feeding cached V-JEPA features into temporal belief training.
- Added canonical online `ViZDoomRuntimeController` for structured or rolling-clip pixel input with the same trained belief/world/actor checkpoints.
- Added fail-closed `SafeActionGuard` to the Doom runtime with an explicit no-attack/no-use recovery action.
- Added exact risk-head geometry to the v2.16 Doom checkpoint ABI and strict reconstruction on load.
- Added durable `ViZDoomTrainingCampaign` with sharded replay, exact resume, checkpoint promotion, structured/pixel tracks, and planner-assisted data aggregation after a stable checkpoint exists.
- Added `planner_used` provenance to Doom replay and preserve it through V-JEPA feature materialization.
- Added `awa-v2-vizdoom-materialize`, `awa-v2-vizdoom-train`, `awa-v2-vizdoom-campaign`, and `awa-v2-vizdoom-training-closure-smoke`.
- Added `configs/v2_16_vizdoom_training.yaml` with plan-only 25k -> 100k -> 250k -> 500k -> 1M milestones.
- Added eleven v2.16 closure regressions; complete source suite now contains 231 tests.


## 2.15.0

- Added direct `ViZDoomAetherEnv` integration for structured-state and RGB pixel tracks.
- Added a stable 4-D hybrid Doom action ABI mapped to ViZDoom delta turn/forward controls plus categorical attack/use.
- Added fixed-dimension normalized ViZDoom telemetry and scenario-goal vectors for structured baselines and telemetry fusion.
- Added deterministic seed/reset handling and world-state save/load snapshot envelopes with explicit `world_state_only` fidelity.
- Added `collect_vizdoom_dataset` with uint8 RGB preservation, telemetry, goals, actions, rewards, constraints, episode IDs and SHA-256 provenance.
- Added a deterministic ViZDoom exploration policy for initial coverage collection without claiming expert behavior.
- Added V-JEPA 2 clip-contract validation and ViZDoom pixel representation manifests.
- Updated the Hugging Face V-JEPA 2 adapter to prefer `get_vision_features()` when the model exposes the official feature API.
- Added `awa-v2-vizdoom-check`, `awa-v2-vizdoom-collect`, `awa-v2-vizdoom-vjepa-cache`, and dependency-free `awa-v2-vizdoom-smoke`.
- Added optional `vizdoom` and `doom-vjepa` dependency groups and `configs/v2_15_vizdoom_vjepa.yaml`.
- Added `tests/test_v2150_vizdoom_vjepa.py` covering action mapping, state contracts, snapshot semantics, pixel collection, V-JEPA feature routing, caching, manifests and structured ABI.
- Preserved the complete v2.14 empirical-qualification, v2.13 campaign, v2.12 scaling, v2.11 robust-planning and v2.10 adaptation regression surfaces.

## 2.14.0

- Added the versioned `aether.game.v1` newline-delimited JSON protocol for external Unity/Godot/custom simulators.
- Added bridge capability negotiation, finite-array validation, Gym-like reset/step adaptation, and optional opaque snapshot/restore.
- Added a Python reference bridge server plus a Unity environment-interface reference and protocol documentation.
- Added true-process `ProcessArenaRolloutPool` for isolated procedural-game qualification workers.
- Added append-only fsync'd `TelemetryRecorder`; durable campaigns now emit collection/training/promotion telemetry.
- Added evidence-based failure triage across perception, world-model, memory, actor, planner, skill, goal, risk, environment, and unknown categories.
- Added strict structured-vs-pixel `RepresentationContract`/`BenchmarkManifest` provenance with encoder fingerprints.
- Added standardized paired-seed ablation runner with bootstrap confidence intervals and deltas against the full system.
- Added 100k/250k/500k/1M milestone qualification reports in JSON and Markdown with desired trend checks.
- Added `awa-v2-game-bridge-check`, `awa-v2-qualification-report`, `awa-v2-empirical-qualification-smoke`, and `configs/v2_14_empirical_qualification.yaml`.
- Added `tests/test_v2140_empirical_qualification.py` covering bridge snapshots, process rollouts, telemetry, triage, benchmark isolation, ablations and milestone reporting.
- Preserved the complete v2.13 campaign, v2.12 scaling, v2.11 robust-planning and v2.10 adaptation regression surfaces.

## 2.13.0

- Added a durable cumulative `GameTrainingCampaign` for 100k -> 250k -> 500k -> 1M+ transition workflows.
- Added hardware-aware CPU/MPS/~12 GB/~24 GB/48 GB+ resource profiles with conservative batch/context limits.
- Added stage-level immutable resume through the existing experiment ledger and replay-aware recovery after interrupted collection.
- Hardened sharded replay with schema-drift rejection and chain-preserving NPZ materialization.
- Added deterministic named-source replay mixing with capacity-aware ratios and cross-source episode-id remapping.
- Added stable checkpoint registry with SHA-256 integrity, promotion history and rollback support.
- Added candidate promotion gates for finite metrics, absolute limits and regression against the current stable checkpoint.
- Added campaign compute planning based on cumulative targets/update counts and optional user-measured throughput.
- Added `awa-v2-training-campaign`, `awa-v2-training-campaign-smoke`, and `configs/v2_13_training_campaign.yaml`.
- Added `tests/test_v2130_training_campaign.py` covering resource profiles, replay integrity/mixing, promotion/rollback, planning and exact resume.
- Preserved the complete v2.12 controlled-scaling, v2.11 robust-planning and v2.10 adaptation regression surfaces.

## 2.12.0

- Added `SparseMoETrunk` and `ModularWorldModel` with hard top-k expert execution rather than dense expert evaluation.
- Added active-MAC matching so monolithic and small-MoE dynamics can be compared at approximately equal active trunk compute.
- Added compute profiles reporting total parameters, approximate active parameters, active trunk MACs and total trunk MACs.
- Added controlled architecture, data-size, model-size and temporal-context scaling curves on one declared dataset.
- Added multi-seed bootstrap confidence summaries for scaling points.
- Added immutable, atomic `ExperimentLedger` state with exact completed-phase resume and config-hash mismatch protection.
- Added FP32/BF16/CUDA-FP16 support to the goal-conditioned temporal world-model trainer.
- Added `awa-v2-controlled-scaling-smoke`, `awa-v2-scaling-sweep`, and `configs/v2_12_controlled_scaling.yaml`.
- Added `tests/test_v2120_controlled_scaling.py` covering sparse routing, compute matching, resumability, precision validation and real scaling curves.
- Preserved the complete v2.11 robust-planning, v2.10 real-adaptation and earlier regression surfaces.

## 2.11.0

- Added sampled multimodal rollout scoring with lower-tail CVaR for risk-aware planning.
- Added `RiskAwarePolicySeededMPPI` with explicit stochastic sample accounting and robust-score metadata.
- Added bootstrap world-model ensemble support in one shared temporal-belief space for epistemic uncertainty.
- Added `WorldModelEnsemble` disagreement metrics separate from each member's aleatoric mixture uncertainty.
- Added canonical hybrid game-action codec: continuous movement plus categorical attack/interact while preserving the 4-float ABI.
- Added append-only integrity-checked sharded replay storage for larger continual-learning datasets.
- Added iterative planner-assisted data aggregation orchestration with retraining callbacks.
- Added multi-seed bootstrap and paired confidence-interval utilities for empirical qualification.
- Added `awa-v2-empirical-scaling-smoke` and `configs/v2_11_empirical_scaling.yaml`.
- Preserved the complete v2.10 real-adaptation, v2.9 game-lab, v2.8 hardening, and prior regression surfaces.

## 2.10.0

- Replaced stage-driven game semantics with executable goal/objective programs and a separate 13-D goal vector.
- Removed curriculum-stage identity from policy observations; observation slot 7 now carries objective progress only.
- Fixed stage-8 tactics so cover is an actual prerequisite and the teacher can execute the tactical behavior.
- Added collision-safe, reachability-validated procedural placement for player, goal, objects, key, door, enemy and cover.
- Made reward shaping follow the active prerequisite instead of always pulling toward the final exit.
- Wired `LocalGlobalTemporalCore` into the real game training/inference path through `GameBeliefEncoder`.
- Added observation/goal reconstruction losses to keep temporal beliefs grounded.
- Added a trained terminal state-value head and removed planner dependence on the legacy untrained transition value output.
- Added backward-compatible zero terminal bootstrap for checkpoints predating the v2.10 value head.
- Added goal-conditioned offline datasets and weighted transition support.
- Wired structural replay priorities into world-model and actor SGD sample weights.
- Added objective-aware game hindsight relabeling and exact one-step counterfactual training branches.
- Added explicit zero-shot, in-context and learning adaptation modes with task-boundary temporal resets and measured prediction error.
- Added `AdaptiveGamePolicy` runtime hierarchy: qualified skill -> actor -> VOC-selected planner.
- Added real-environment snapshot-branch VOC fitting with return gain, latency and model-call cost.
- Made compositional/task OOD categories modify executable objective sequences instead of metadata-only labels.
- Added `awa-v2-game-adaptation-smoke`, `configs/v2_10_real_adaptation.yaml`, and `tests/test_v2100_real_adaptation.py`.
- Preserved the full v2.9 procedural-game and earlier regression surface.

## 2.9.0

- Added a dependency-free procedural continuous game lab with a stable 32-D observation and 4-D action contract.
- Added 12 curriculum stages spanning movement through compositional multi-step tasks.
- Added exact snapshot/restore, structured risk labels, deterministic RGB rendering, and visual-OOD appearance variation.
- Hid simulator dynamics parameters from default observations so physics shifts must be inferred from experience.
- Added a logical demonstration teacher and random exploration policy.
- Added chain-consistent procedural-game NPZ dataset generation with episode/stage metadata.
- Added `ReusableGameLoop` integrating live gameplay with reusable-learning replay, hindsight, concept coverage, skill discovery, and checkpointing.
- Added end-to-end structured-game training: identity representation cache, multimodal world model, uncertainty calibration, risk training, TD3+BC actor, and policy-seeded MPPI evaluation.
- Added held-out game transfer evaluation across seen, visual, layout, dynamics, compositional, and task OOD categories.
- Added `awa-v2-game-lab-generate`, `awa-v2-game-train`, `awa-v2-game-loop-smoke`, and `awa-v2-game-lab-smoke`.
- Added `configs/v2_9_game_lab.yaml` and `tests/test_v290_game_lab.py`.
- Preserved the complete v2.8 hardening and prior v2/v1 regression surface.

## 2.8.0

- Audited the packaged v2.7 release and reproduced integration defects with targeted probes before patching.
- Persist executable skill-policy weights and learned skill-contract models in reusable-learning checkpoints.
- Added backward-compatible reusable-learning checkpoint schema v2 while retaining v1 loading.
- Added JSON-safe serialization for NumPy/tensor-rich metadata and restored Engram use counts.
- Fixed repeated adaptive curriculum batches with a resumable generation counter.
- Fixed mixed-dimension novelty-memory crashes and dimension-incompatible Engram bucket collisions.
- Reject rollout envelopes stamped with future policy versions.
- Added Gymnasium 5-tuple/reset compatibility to exact snapshot branching and planner-benefit collection.
- Added explicit start-observation branching support and action-space probe inference.
- Closed local rollout environments and hardened rollout argument validation.
- Hardened SafeActionGuard recovery bounds/risk accounting and fail-closed controller behavior.
- Treat non-finite risk/uncertainty as unsafe in selective decision logic.
- Reject prediction-error shape broadcasting and non-finite transition values.
- Exclude hindsight rows from world-model replay export by default and regroup interleaved episode rows when episode metadata exists.
- Corrected transfer episodes-to-target to a one-based exposure count.
- Hardened teacher selection, skill/contract/distillation trainers, structural replay validation, task verifiers, and task/outcome validation.
- Added `awa-v2-hardening-smoke` and `tests/test_v280_hardening.py`.
- Preserved the full v2.7/v2.6/v2.5 and legacy regression surfaces.

## 2.7.0

- Added deterministic task-verifier registry for synthesized curriculum tasks.
- Added hindsight future-goal relabeling for goal-conditioned experience reuse.
- Added exploit/explore/qualification mode controller with intrinsic rewards disabled during qualification.
- Added learned skill precondition, termination, and failure contract models.
- Added bounded goal-conditioned continuous skill policies and per-skill trajectory distillation.
- Added explicit six-category held-out/OOD transfer benchmark and adaptation curves.
- Added concept-level competence tracking and adaptive weak-concept task generation.
- Added local deterministic rollout worker pool using policy-versioned rollout envelopes.
- Added integrated `ReusableLearningEngine` coordinating curriculum, replay, novelty, distillation, skills, EngramLite, reasoning effort, and task verification.
- Added atomic reusable-learning checkpoint/restore and replay export into the standard Aether NPZ training contract.
- Added `awa-v2-closed-loop-smoke`, `awa-v2-transfer-smoke`, `awa-v2-skill-contract-smoke`, and `awa-v2-rollout-pool-smoke`.
- Preserved the complete v2.6 reusable-learning, v2.5 video, v2.4 benchmark, and prior regression surfaces.

## 2.6.0

- Added procedural 12-stage reusable-learning task factory with rule/appearance randomization.
- Added learning-frontier curriculum scheduling with learning-progress, novelty, surprise, and cold-start weighting.
- Added structured experience records separating novelty, failure, prediction error/surprise, uncertainty, risk, TD error, task importance, and information value.
- Added multi-signal prioritized replay with clipping, importance weights, and repeated-surprise boost.
- Added exact-state counterfactual action branch generation for causal training data.
- Added continuous reasoning-effort controller over actor and progressively larger planner budgets.
- Added heterogeneous multi-teacher selection and weighted planner-to-actor distillation.
- Added automatic successful-trajectory signature clustering, skill qualification registry, and skill composition.
- Added bounded `EngramLite` sparse pattern memory.
- Added periodic context checkpoints with bounded replay reconstruction.
- Added rollout policy-version envelopes with stale-sample downweighting/rejection for async simulation.
- Added `awa-v2-curriculum-smoke` and `awa-v2-reusable-smoke`.
- Preserved the full v2.5 pretrained-game-video, v2.4 closed-loop benchmark, v2.3 planner/VOC, v2.2 world-model, and prior regression surfaces.


## 2.5.0

- Added causal game-video clip construction with strict episode-boundary isolation.
- Added first-class Hugging Face V-JEPA 2 frozen feature extraction via `AutoVideoProcessor` + `AutoModel`.
- Added local-first Meta V-JEPA 2.1 PyTorch-Hub adapter.
- Added frozen video-feature + structured telemetry fusion before the trainable Aether projection.
- Added content-addressed structured-input cache support using precomputed SHA-256 digests.
- Added game-video cache indexes compatible with the existing v2.2 world-model sequence trainer.
- Added online `StreamingGameFeatureEncoder` with offline/streaming feature parity.
- Added `AetherV2Agent.observe_latent` for external pretrained-perception runtimes.
- Added optional `telemetry` / `next_telemetry` NPZ dataset fields.
- Added `awa-v2-game-cache` and `awa-v2-game-repr-smoke`.
- Added dependency-free synthetic RGB game dataset for CI qualification.
- Preserved the v2.4 checkpoint benchmark path and complete prior regression surface.


## 2.4.0

- Added strict world checkpoint reconstruction for v2.2/v2.4 world artifacts.
- Added backward-compatible inference of v2.2 risk-head geometry when legacy checkpoints omitted that metadata.
- Added versioned actor checkpoint save/load for the TD3+BC no-search baseline.
- Added deterministic identity state-vector representation with stable encoder fingerprinting.
- Added multi-seed closed-loop benchmark qualification on identical seed sets.
- Added paired actor-vs-planner return deltas and paired bootstrap confidence intervals.
- Added compute-normalized scorecards with latency, model calls, search rate, and return per 1k model calls.
- Added `awa-v2-actor-train`, `awa-v2-benchmark-qualify`, and `awa-v2-benchmark-smoke`.
- Preserved v2.3 exact-state branching/VOC fitting and the complete v2.2/v2.1/v1.x regression surface.

## 2.3.0

- Added TD3+BC-style actor-only continuous baseline in the v2 latent space.
- Added cached-feature-to-actor-baseline bridge so actor/planner comparisons share perception.
- Added policy-free vanilla CEM and MPPI controls.
- Kept policy-seeded MPPI and iCEM as distinct planner identities.
- Added exact snapshot replay verification and counterfactual real branch collection.
- Added planner-benefit matrices with actual model-call, latency, gain, and cost measurements.
- Added supervised Value-of-Computation fitting from measured planner/budget gains.
- Added equal-world-model-call budget translation and efficiency reporting.
- Hardened gradient planning so it works correctly under inference/no-grad callers without accumulating world-model gradients.
- Added fail-closed `SafeActionGuard` recovery behavior.
- Added `awa-v2-actor-smoke`, `awa-v2-planner-qualify`, and `awa-v2-safety-smoke`.
- Preserved the complete v2.2 world qualification, v2.1 representation stack, and v1.x regression surface.

## 2.2.0

- Added episode-safe, chain-verified cached feature sequences.
- Added offline multimodal world-model training with mixture NLL, reward, continuation, and multi-step overshooting losses.
- Added horizon-specific latent rollout RMSE and trajectory-ranking correlation.
- Added uncertainty-head calibration against realized multi-step rollout error.
- Added optional explicit `constraints` labels to offline NPZ datasets.
- Added separate risk-model training and Brier/accuracy/calibration reporting.
- Added candidate-vs-baseline world-model promotion gates.
- Added chain-consistent synthetic dataset generator for dependency-free qualification.
- Added `awa-v2-world-smoke`, `awa-v2-risk-smoke`, `awa-v2-world-qualify`, and `awa-v2-promote`.
- Preserved the complete v2.1 representation/cache stack and v2.0/v1.x regression surface.

## 2.1.0

- Added injectable frozen pretrained representation backbones.
- Added local Hugging Face/AutoModel vision backbone adapter for DINO-family and similar models.
- Added content-addressed deterministic frozen-feature caching with atomic writes.
- Added validated NPZ offline transition ingestion and dataset SHA-256 manifests.
- Added geometry-preserving projection warm-up training.
- Added `awa-v2-cache` and `awa-v2-repr-check`.
- Preserved the v2.0 planner/world-model/VoC architecture and full v1.x regression surface.


## 2.0.0

- Architectural reset: policy-first execution, search only when useful.
- Added clean `awa.v2` stack beside the v1.8 baseline.
- Added local/global temporal belief core.
- Added multimodal mixture future model.
- Added monotone horizon-aware uncertainty.
- Added explicit risk/constraint model separated from uncertainty.
- Added policy-seeded MPPI and iCEM planner backends.
- Added differentiable gradient planner backend.
- Added value-of-computation model for planner/budget selection.
- Added long-context and episodic memory primitives.
- Added reachability/risk-aware subgoal planner.
- MaleCNS remains outside the critical path.

# Changelog

## 1.8.0

### Added
- Optional quantile twin-Q continuous critic.
- Quantile Huber regression and conservative distributional target bootstrapping.
- Configurable n-step continuous-Q replay targets.
- Multi-step real-environment planner-benefit branch collector.
- Compute-aware planner-benefit labels.
- Native-cache probing hooks for sequence backends.
- Multi-seed environment qualification with per-seed error capture.
- Snapshot round-trip qualification for stateful environments.
- `v1_8_full.yaml` and `v1_8_continuous.yaml`.
- `awa-env-multiseed` CLI.

### Changed
- `awa-gate-fit` now supports `--branch-horizon` and `--planning-cost-per-1k`.
- Continuous critic checkpoints remain raw-value compatible while supporting distributional heads.
- Mamba cache integration now attempts native incremental execution before bounded recomputation.

### Compatibility
- Scalar twin-Q remains the default for older configs.
- One-step planner-benefit collection remains available as a compatibility wrapper.
- All v1.7 regression tests remain included.

## 1.7.0

### Added
- Action-conditioned transition uncertainty ensemble.
- Automatic uncertainty calibration from realized next-latent errors.
- Twin continuous Q critics plus EMA target critic.
- Running target-scale normalization without changing critic output units.
- Optional Q-regularized continuous actor objective.
- One-step real-environment planner-benefit dataset collector.
- Learned actor/planner arbitrator fitting from measured planner benefit.
- Backend-neutral cached sequence interface.
- Optional Meta-World MT1 adapter.
- Environment-interface qualification utility and CLI.
- `v1_7_full.yaml`, `v1_7_continuous.yaml`, and `v1_7_gate_fit.yaml`.
- `awa-calibrate`, `awa-gate-fit`, and `awa-env-check` CLIs.

### Changed
- Runtime uncertainty now evaluates the proposed action, not state alone.
- World-model uncertainty heads train on state-action → next-latent transitions.
- Complete checkpoints include optional Q critic, target Q critic, and target-scale state.
- Continuous training can combine imagination returns with twin-Q regularization.

### Compatibility
- All v1.6 discrete and continuous tests remain in the regression suite.
- State-only `UncertaintyEnsemble` construction remains supported with `action_dim=0`.

## 1.6.0
- Added end-to-end bounded continuous control, continuous CEM, vector-action replay, Gymnasium Box/dm_control adapters, and continuous checkpoint/resume.
# v2.38.6 — Planner and Reward Diagnosis

- Added a small paired raw-vs-clipped reward diagnostic on one shared `my_way_home` dataset with matched initialization and evaluation seeds. Promotion uses realized navigation success; reports include action entropy and clearly labeled reward/telemetry component proxies.
- Inserted P1D reward diagnosis and P1P planner oracle ladder before P2. P1P validates all five information ablations, horizons 1/2/4/8/16/32, actor-seeded/mixed candidate pools, predicted-versus-realized ranking, and proposal diversity. It never fabricates missing oracle reward measurements.
- Changed P0 storage qualification to use the declared provisioned RunPod volume quota and measured directory usage instead of backing-filesystem free space.
- Added per-transition scenario IDs and reward/success/health/timeout/displacement diagnostics to dataset metadata and training campaign evidence.
- Marked first-checkpoint registration as `baseline_registration`; normal promotion remains reserved for a comparison against an incumbent.
- Kept action-time architecture frozen.
- Gatefix 3 recomputes P1P reports from retained raw rows, binds them to verified P1 checkpoint files and a complete real-backend branch replay report, and rejects conflicting returns for identical candidate branches.
- Added a real ViZDoom reset-replay probe. P0 now checks local IPC socket capability before the native Doom engine can crash on restricted hosts.
- Gatefix 4 replaces the infeasible P1P-B imagined-state oracle with paired real ViZDoom branch returns as the reference. B evaluates learned dynamics and reward jointly, C adds terminal value, and D adds risk. The raw/report protocol advances to v2 and rejects v1 oracle claims; no empirical P1P result is claimed. This is an execution-blocking diagnostic protocol repair.
