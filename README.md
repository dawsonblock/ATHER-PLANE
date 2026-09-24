# Aether World Agent v2.38.6

**v2.38.6 Diagnostic Gate:** adds a paired raw-vs-clipped `my_way_home` reward test, then a fixed-seed planner real-branch comparison before P2. The `awa-v2-vizdoom-planner-collect` command collects repeatable native branches and closed-loop episodes with frozen P1 checkpoints. It also records scenario/reward telemetry in collection and training evidence, marks the first stable checkpoint as baseline registration, and checks the provisioned persistent-volume quota instead of host `df` free space. See `docs/PLANNER_DIAGNOSTIC_P1P_2_38_6.md`.

The v2.38.5 code and evidence remain frozen. v2.38.6 changes execution/evidence gates only; it does not add an action-time cognitive module.

**v2.38.4 Cumulative Training Repair:** The previous structured and pixel campaigns accumulated replay but reset the belief, world, actor, and optimizer weights at every stage. This release restores promoted checkpoints, optimizer/scaler states, update counts, and RNG; all stages in a track use the same model width. Later stage evidence requires verified resume lineage. The frozen v2.38.3 run and its failure remain archived. Fresh empirical qualification is required. See `docs/CUMULATIVE_TRAINING_FIX_2_38_4.md`.

**v2.38.3 Replay Budget Repair:** Selects recent experience while holding the 10K budget and retaining the complete first collection. See `docs/REPLAY_BUDGET_FIX_2_38_3.md`.

**v2.38.2 Execution Repair:** The P2 evidence gate, cross-scenario promotion, P9 checkpoint selection, and P8 V-JEPA memory usage are corrected without changing the action-time agent. See `docs/UPGRADE_2_38_2.md` and `docs/RUNPOD_CAPACITY_2_38_2.md`.

**v2.38 Progressive Execution Orchestrator:** the action-time agent remains frozen while the v2.37 capability ladder becomes a resumable, one-phase-at-a-time execution surface for CUDA hosts. `awa-v2-progressive-executor` plans or executes only the first blocked empirical phase, uses shell-free argv execution, applies transition-budget guards, and requires a second acknowledgement for the expensive five-seed DREAM-RSI campaign. Focused five-seed 25K pairwise gates now qualify temporal memory, fixed planning, and adaptive compute without forcing the complete seven-system 25K/100K matrix during bring-up.

> Current release: **v2.38.6** — paired reward and planner diagnostics; no new cognition.

Quick RunPod path:

```bash
bash deploy/runpod/bootstrap.sh
awa-v2-progressive-executor --config configs/v2_38_progressive_executor.yaml --evidence-root runs/v2_38_6
```

See `docs/UPGRADE_2_38.md` and `docs/REMAINING_WORK_2_38.md`.

# Aether World Agent v2.35.0

**Policy-first hierarchical world-model research platform.**

v2.35 keeps the v2.34 agent and DREAM-RSI experiment frozen while hardening real execution: fail-closed GPU/dependency preflight, native-only ViZDoom training receipts, real V-JEPA enforcement for pixel campaigns, and stronger matched-work planner hardware telemetry.

v2 preserves the validated v1.8 stack as a compatibility/research baseline and adds a clean `awa.v2` architecture built around multimodal latent futures, horizon-aware uncertainty, explicit risk, selective planning, and value-of-computation.

Quick checks:

```bash
pip install -e .[dev]
awa-v2-smoke
awa-v2-arena --world-calls 64
pytest -q
```

See `docs/V2_ARCHITECTURE.md` and `docs/V2_BUILD_PLAN.md`.



## v2.35 real execution readiness

v2.35 is an execution release, not a cognitive release. It adds a fail-closed machine preflight, native-only real ViZDoom training campaign wrapper, non-injected V-JEPA enforcement for the pixel track, and stronger hardware planner equivalence/throughput reporting.

```bash
# inspect the actual GPU host before spending compute
awa-v2-execution-preflight \
  --config configs/v2_35_empirical_execution.yaml \
  --out-dir runs/v2_35_execution

# plan then execute the real structured ViZDoom bring-up
awa-v2-vizdoom-real-campaign \
  --config configs/v2_35_vizdoom_structured.yaml \
  --out-dir runs/v2_35_structured

awa-v2-vizdoom-real-campaign \
  --config configs/v2_35_vizdoom_structured.yaml \
  --out-dir runs/v2_35_structured --execute
```

The release still does not claim that native ViZDoom, real V-JEPA, the canonical five-seed DREAM-RSI comparison, the 70-job ablation, or a real GPU planner sweep have been executed in the shipped artifact. See `docs/UPGRADE_2_35.md` and `docs/REMAINING_WORK_2_35.md`.


## v2.34 DREAM-RSI empirical qualification

v2.34 operationalizes the experiment that v2.33 only specified. `FixedVsDreamCampaignRunner` runs a fixed meta-policy and a replay-improved DREAM-RSI meta-policy under the same initial policy, offered menus, paired seeds, action-time agent/evaluator contract, worker ceiling, and per-round transition ceiling. Each arm continues from its best grounded checkpoint, so recursive rounds compare cumulative learning trajectories rather than unrelated from-scratch fits.

Development replay can score trained allocations on heldout and transfer compositions. The final paired endpoint is evaluated again on a disjoint factorized meta-test seed that was never used to improve the DREAM policy. Incomplete seed sets fail closed.

```bash
# freeze/inspect the five-seed, five-round plan
awa-v2-dream-rsi-campaign \
  --config configs/v2_34_dream_rsi_empirical.yaml \
  --out-dir runs/v2_34_dream_rsi

# execute/resume
awa-v2-dream-rsi-campaign \
  --config configs/v2_34_dream_rsi_empirical.yaml \
  --out-dir runs/v2_34_dream_rsi --execute

# tiny real-execution + resume regression
awa-v2-dream-rsi-campaign-smoke
```

See `docs/UPGRADE_2_34.md`, `docs/REMAINING_WORK_2_34.md`, `docs/SOURCE_ADOPTION_MATRIX_2_34.md`, and `DREAM_RSI_EMPIRICAL_PROTOCOL_2_34.json`. The canonical five-seed result is still unexecuted in the release artifact.



## v2.32 grounded DREAM-RSI meta-learning

v2.32 keeps the action-time `agent_runtime` frozen and adds a bounded DREAM-RSI-style outer loop over **training allocation**. Real campaign history becomes a grounded replay simulator; declarative allocation policies are compared cheaply in replay; replay winners must still pass paired real-online qualification before promotion.

The same `TrainingAllocationDecision` interface is used in replay and online execution. Decisions control task-factor allocation, collector choice, transition/planner budgets, parallel worlds, and stopping. Historical replay accepts only observed/validated outcomes, binds a fixed agent/evaluator/interface fingerprint, tracks exact historical support, and marks weakly supported winners for online probing. Arbitrary policy-code rewriting is not allowed.

```bash
awa-v2-dream-rsi-smoke

# canonical v2.32 integration contract
cat configs/v2_32_dream_rsi_meta_learning.yaml
```

See `docs/UPGRADE_2_32.md`, `docs/REMAINING_WORK_2_32.md`, and `DREAM_RSI_INTEGRATION_2_32.json`. The paper's reported discovery-domain gains are not treated as embodied-RL evidence for Aether.


## v2.31 proper system split

v2.31 makes the architectural split explicit without adding a new cognitive mechanism. The canonical public surfaces are now:

```text
awa.v2.agent_runtime   # inference-time cognition/action selection only
awa.v2.learning_system # collection, datasets, replay, environment generation, curriculum, training
awa.v2.research_os     # protocols, provenance, ablations, compute accounting, qualification
```

Dependency direction is enforced on the new split surface: `agent_runtime` cannot depend on the learning system or research OS; the learning system may consume the agent runtime; the research OS may observe both but cannot become part of action-time inference. Historical flat imports remain as compatibility implementation details for this release.

```bash
# verify the boundary contract
awa-v2-split-check

# canonical empirical campaign under the split
awa-v2-ablation-campaign \
  --config configs/v2_31_system_split.yaml \
  --out-dir runs/v2_31_ablation --execute
```

The canonical agent runtime excludes skill discovery, Engram, sparse-MoE/modular-world experiments, connectome experiments, and legacy prototype families. Those may remain available for isolated research, but they are not credited as part of the production agent until they win an explicit ablation.

See `docs/UPGRADE_2_31.md`, `docs/REMAINING_WORK_2_31.md`, and `SYSTEM_SPLIT_2_31.json`.

## v2.30 empirical pivot

v2.30 freezes cognitive expansion and makes empirical maturity explicit. `EMPIRICAL_STATUS.md` separates software validation from real ViZDoom/V-JEPA execution and benchmark qualification. The release adds a qualification path that refuses injected/fake ViZDoom and V-JEPA backends, a factorized compositional environment factory with split-leakage checks, and a real-device planner schedule benchmark for matched logical workloads.

Historical orphan v2 prototypes and the connectome experiment are moved behind `awa.experimental.*` compatibility boundaries. The v1 runtime remains for reproduction/compatibility only.

```bash
# show the evidence boundary
awa-v2-empirical-status

# generate factorized train/heldout/transfer task manifest
awa-v2-compositional-benchmark --output runs/compositional_benchmark.json

# requires the real optional dependency, not FakeDoomGame
pip install -e ".[dev,vizdoom]"
awa-v2-vizdoom-real-qualify --out-dir runs/real-doom --track structured

# execute the frozen 25K/100K ablation
awa-v2-ablation-campaign \
  --config configs/v2_30_empirical_pivot.yaml \
  --out-dir runs/v2_30_ablation --execute
```

See `docs/UPGRADE_2_30.md`, `docs/REMAINING_WORK_2_30.md`, `EMPIRICAL_STATUS.md`, and `EMPIRICAL_PIVOT_AUDIT_2_30.json`.




## v2.29 compute-aware planning + physical execution accounting

v2.29 applies the execution-schedule lesson from *Sample Count Is Not Enough: Candidate-Generation Strategy Shapes the Energy and Performance of LLM Test-Time Scaling* (arXiv:2609.19499) to Aether's model-based planning. A logical planner budget no longer stands in for physical execution cost. Planner evidence now separates imagined world-model transitions from actual batched `world.imagine_step` forwards and records the batch schedule used to execute them.

The existing MPPI/iCEM implementation already vectorized independent candidates. v2.29 makes that fact explicit, testable, configurable, and part of the empirical protocol rather than claiming a new reasoning algorithm. `planner_world_batch_size: 0` means maximally batched execution; a positive value enforces deterministic microbatching for memory-limited hardware. The same logical candidate workload can therefore be compared under different physical schedules without changing task semantics.

```bash
# canonical maximally batched campaign
awa-v2-ablation-campaign \
  --config configs/v2_29_compute_efficiency.yaml \
  --out-dir runs/v2_29_ablation

# execute/resume
awa-v2-ablation-campaign \
  --config configs/v2_29_compute_efficiency.yaml \
  --out-dir runs/v2_29_ablation \
  --execute

# rebuild capability + compute reports
awa-v2-ablation-report \
  --config configs/v2_29_compute_efficiency.yaml \
  --out-dir runs/v2_29_ablation
```

Every completed job now writes an execution fingerprint and consolidated `compute_schedule_report.json` containing logical world-model transitions, physical forward invocations, mean/max world-model batch sizes, latency, accelerator wall-hours and peak CUDA memory. Energy is deliberately **not estimated** from wall time; `accelerator_energy_joules` remains null unless a future hardware meter measures it. The keep/remove report can now credit non-inferior mechanisms that materially reduce physical forwards or logical model work, in addition to latency and planner-call reductions.

See `docs/UPGRADE_2_29.md`, `docs/REMAINING_WORK_2_29.md`, `configs/v2_29_compute_efficiency.yaml`, and `COMPUTE_SCHEDULE_AUDIT_2_29.json`.

## v2.28 resumable ablation campaign + decision closure

v2.28 operationalizes the v2.27 seven-system ladder as a preregistered, resumable 70-training-job / 140-evaluation-cell campaign. Each `(system, seed, milestone)` trains once and the same frozen checkpoint is evaluated on heldout and transfer splits. Campaign plans, thresholds, system hashes, datasets, checkpoints, and job receipts are bound before they can enter consolidated evidence.

The runner is safe to resume after cloud preemption and supports deterministic multi-worker sharding over a shared volume. Dataset construction is shared across comparable systems and smaller milestones are exact prefixes of the largest milestone dataset.

```bash
# freeze/inspect the plan
awa-v2-ablation-campaign \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation

# execute or resume
awa-v2-ablation-campaign \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation \
  --execute

# rebuild consolidated evidence without retraining
awa-v2-ablation-report \
  --config configs/v2_28_ablation_campaign.yaml \
  --out-dir runs/v2_28_ablation
```

`ablation_report.json` computes adjacent marginal contribution for temporal memory, the world model, fixed planning, adaptive compute, grounded hindsight replay, and adaptive curriculum. It refuses to issue `KEEP` / `REMOVE_CANDIDATE` decisions until the entire preregistered matrix and minimum seed count are present.

See `docs/UPGRADE_2_28.md`, `docs/REMAINING_WORK_2_28.md`, and `configs/v2_28_ablation_campaign.yaml`.

## v2.27 executable ablation + RNG resume closure

v2.27 turns the preregistered component ladder into seven concrete runtimes: `actor_only`, `belief_actor`, `world_actor`, `world_planner`, `adaptive_compute`, `reusable_replay`, and `full_curriculum`. The old broad `reusable_learning` label is intentionally narrowed because skill discovery, Engram, MoE and ensemble families are not credited until they have isolated executable paths.

The release also checkpoints Python, NumPy, PyTorch CPU and (when present) CUDA RNG state alongside optimizer-complete world/actor checkpoints so stochastic continuation can resume from the recorded random stream.

Useful checks:

```bash
awa-v2-ablation-runtime-smoke
awa-v2-ablation-run --variant actor_only --out-dir runs/actor_only
awa-v2-ablation-protocol --output runs/v227_ablation_protocol.json
pytest -q tests/test_v2270_executable_ablation.py
```

The current grounded autonomous campaign is also versioned for this release:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_27_autonomous_empirical_closure.yaml \
  --out-dir runs/v2_27_meta \
  --bootstrap-active --execute
```

See `docs/UPGRADE_2_27.md`, `docs/REMAINING_WORK_2_27.md`, `configs/v2_27_executable_ablation.yaml`, `configs/v2_27_autonomous_empirical_closure.yaml`, and `ABLATION_MATRIX_2_27.json`.

## v2.26 autonomous collection + ablation preregistration

v2.26 closes the largest remaining methodological gap in the native procedural campaign: data collection no longer has to remain teacher-only. The new canonical campaign bootstraps its first milestone with a non-semantic coverage explorer, then larger cumulative milestones collect only their transition delta with the frozen grounded parent Aether actor plus bounded policy-seeded planning. Every batch records both requested and actual collector identity.

World-model cumulative resume is also optimizer-complete: `game_world.pt` now carries AdamW state, CUDA GradScaler state when active, and the global world update count. The empirical ablation protocol can bind an immutable SHA-256 to every system configuration before results exist.

Canonical run:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_26_autonomous_empirical_closure.yaml \
  --out-dir runs/v2_26_meta

awa-v2-evolving-campaign \
  --config configs/v2_26_autonomous_empirical_closure.yaml \
  --out-dir runs/v2_26_meta \
  --bootstrap-active --execute
```

Useful checks:

```bash
awa-v2-autonomous-collection-smoke
awa-v2-ablation-protocol --output runs/ablation_protocol.json
pytest -q tests/test_v2260_autonomous_ablation.py
```

See `docs/UPGRADE_2_26.md` and `docs/REMAINING_WORK_2_26.md`.

## v2.25 reduction + cumulative milestone training

v2.25 is deliberately a reduction release. It removes wasted campaign work without adding a new cognitive mechanism. The native procedural campaign now shares its content-addressed training cache across bootstrap and later meta-iterations, warm-starts each larger milestone from the previous grounded checkpoint, collects only the transition delta, and trains only on that delta. The canonical 25K→100K→250K→500K→1M curve therefore consumes 1M newly collected/trained transitions per seed instead of 1.875M when every milestone is rebuilt independently.

Native meta-policy search is also execution-aware: only curriculum axes that can change the real task allocation are mutated, and policies that round to the same stage allocation are deduplicated before replay/online qualification. Planner rollouts are disabled inside routine native training because canonical qualification is actor-only; planner value remains a separate experiment. `awa.v2` and `awa.v2.game` now expose the same convenience API lazily, avoiding eager initialization of unrelated experimental and optional subsystems.

Canonical run:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_25_reduction_empirical_closure.yaml \
  --out-dir runs/v2_25_meta

awa-v2-evolving-campaign \
  --config configs/v2_25_reduction_empirical_closure.yaml \
  --out-dir runs/v2_25_meta \
  --bootstrap-active --execute
```

See `docs/UPGRADE_2_25.md` and `docs/REMAINING_WORK_2_25.md`.




## v2.24 milestone empirical closure

v2.24 preserves v2.23 train-once/evaluate-many artifacts and hardens policy promotion across full learning curves. Promotion now treats final-milestone performance, transition-normalized curve area, per-seed regressions, and heldout/transfer split regressions as separate gates instead of collapsing all cells into one average. Executed iterations emit a content-addressed `milestone_scorecard.json`. See `docs/UPGRADE_2_24.md`.

## v2.23 train-once / evaluate-many qualification

v2.23 separates native procedural **training identity** from **evaluation identity**. The same policy/seed/milestone is collected and trained once, committed as a content-addressed artifact, then evaluated independently on frozen heldout and transfer scenario sets. This removes duplicate retraining across evaluation splits and prevents qualification tasks from being passed into the training routine.

```bash
awa-v2-evolving-campaign \
  --config configs/v2_23_train_once_evaluate_many.yaml \
  --out-dir runs/v2_23_meta

awa-v2-evolving-campaign \
  --config configs/v2_23_train_once_evaluate_many.yaml \
  --out-dir runs/v2_23_meta \
  --bootstrap-active --execute
```

`awa-v2-native-campaign-smoke` now proves heldout/transfer reuse of the exact checkpoint and dataset while keeping the evaluation scenario sets disjoint. See `docs/UPGRADE_2_23.md`.


## v2.22 native closed-loop execution

v2.22 makes the evolving campaign runnable end-to-end on Aether's built-in procedural environment without an external subprocess adapter. Empty-history runs now fail into an explicit `NEEDS_BOOTSTRAP` plan; `--bootstrap-active --execute` collects grounded active-policy evidence, then the normal DREAM-RSI-style replay proposal, exact paired validation, provenance/protocol checks, and promotion gate continue.

```bash
awa-v2-evolving-campaign \
  --config configs/v2_22_native_evolving_campaign.yaml \
  --out-dir runs/v2_22_meta

awa-v2-evolving-campaign \
  --config configs/v2_22_native_evolving_campaign.yaml \
  --out-dir runs/v2_22_meta \
  --bootstrap-active --execute
```

The v2.20 adaptive reasoning-effort controller is also load-bearing in `AdaptiveGamePolicy` when supplied with trained VOC gain predictions. Exact planner/budget choices are used; untrained choices are not guessed, and low world-model reliability masks model-based planning. See `docs/UPGRADE_2_22.md`.


## v2.21 evolving campaign closure

v2.21 wires the v2.20 meta-exploration layer into one durable empirical loop. Completed grounded run records are reconstructed into branching replay worlds, a bounded declarative exploration-policy change is proposed offline, an exact active-vs-candidate validation matrix is preregistered, and a fixed subprocess runner seam can launch each paired real campaign cell. The resulting evidence is checked by the v2.19 protocol and v2.18 provenance/leakage gates before the normal paired promotion gate may update the active policy.

Plan mode is still the default:

```bash
awa-v2-evolving-campaign \
  --config configs/v2_21_evolving_campaign.yaml \
  --history-records evidence/history.jsonl \
  --history-provenance evidence/provenance.json \
  --out-dir runs/v2_21_meta
```

External execution requires both `--execute` and an explicit `runner.argv` in the config. The runner uses `shell=False`, must emit the versioned v2.21 evidence ABI, and cannot promote replay/model-simulated results. Validation cells are committed atomically and reused on resume.

```bash
awa-v2-evolving-campaign-smoke
pytest -q tests/test_v2210_evolving_campaign.py
```

See `docs/UPGRADE_2_21.md` and `configs/v2_21_evolving_campaign.yaml`.

## v2.20 evolving exploration + adaptive compute

v2.20 adds a bounded DREAM-RSI-style meta-learning plane over Aether's existing campaigns. Completed macro-level discovery histories become grounded replay worlds; a declarative exploration policy can be improved offline, but replay alone cannot promote it. Promotion requires paired online `observed`/`validated` evidence, preserving the v2.18/v2.19 provenance and preregistration boundary.

It also upgrades reasoning effort from a simple gain-minus-cost ladder to an adaptive compute controller that accounts for predicted gain, latency, risk, resource pressure, and world-model reliability. Model-based planning levels fail closed when reliability is below the configured floor; the normal downstream safety guard remains authoritative.

```bash
awa-v2-evolving-exploration-smoke
pytest -q tests/test_v2200_evolving_exploration.py
```

See `docs/UPGRADE_2_20.md` and `configs/v2_20_evolving_exploration.yaml`.



## v2.16 ViZDoom training closure

v2.16 closes the main integration gaps left by v2.15. ViZDoom structured telemetry and cached V-JEPA+telemetry features now train through the same goal-conditioned `GameBeliefEncoder` (local GRU + global context), stochastic world model, explicit risk head, terminal value, and hybrid actor/critic. Attack and use are categorical throughout the new actor/planner path rather than fractional continuous controls.

New commands:

```bash
awa-v2-vizdoom-materialize runs/doom-pixel.npz runs/doom-vjepa-cache/vizdoom_index.json runs/doom-features.npz --cache-dir runs/doom-vjepa-cache
awa-v2-vizdoom-train runs/doom-features.npz --track pixel --out-dir runs/doom-trained
awa-v2-vizdoom-campaign --config configs/v2_16_vizdoom_training.yaml
awa-v2-vizdoom-training-closure-smoke
```

The campaign command is plan-only unless `--execute` is supplied. Later stages can collect planner-assisted trajectories from the current promoted checkpoint; those actions re-enter replay and train the hybrid actor, providing an explicit planner-to-policy distillation path. V-JEPA cache misses are batched while preserving the established content-addressed cache ABI.

See `docs/UPGRADE_2_16.md`.


## v2.15 ViZDoom + V-JEPA 2 integration

v2.15 adds the first direct external game adapter selected for Aether's visual-control training: ViZDoom. It supports a structured telemetry track and a separate RGB track that feeds causal video clips into the frozen V-JEPA 2 representation/cache pipeline. The default visual checkpoint contract is `facebook/vjepa2-vitl-fpc64-256` with 64-frame clips.

New commands:

```bash
awa-v2-vizdoom-check --scenario my_way_home --track structured
awa-v2-vizdoom-collect runs/doom-pixel.npz --scenario my_way_home --track pixel --episodes 32
awa-v2-vizdoom-vjepa-cache runs/doom-pixel.npz --cache-dir runs/doom-vjepa-cache --model facebook/vjepa2-vitl-fpc64-256 --frames 64
awa-v2-vizdoom-smoke
```

Install the optional game/perception dependencies with:

```bash
pip install -e ".[dev,doom-vjepa]"
```

The structured and pixel benchmarks remain separate by design. See `docs/UPGRADE_2_15.md`.


## v2.14 empirical qualification and real-game bridge

v2.14 adds the missing external-game and empirical-qualification layer around the v2.13 durable training campaign. It introduces the versioned `aether.game.v1` JSONL protocol with optional exact snapshot/restore, a Gym-like bridge adapter, true-process procedural rollouts, append-only training telemetry, evidence-based failure triage, strict structured-vs-pixel benchmark contracts, paired-seed ablations, and standardized 100k/250k/500k/1M milestone reports.

New commands:

```bash
awa-v2-game-bridge-check --host 127.0.0.1 --port 8765
awa-v2-qualification-report metrics.jsonl --out-dir runs/v2_14_qualification
awa-v2-empirical-qualification-smoke
```

See `docs/UPGRADE_2_14.md` and `docs/GAME_BRIDGE_PROTOCOL.md`.

## v2.13 durable training campaigns

v2.13 adds cumulative 100k → 250k → 500k → 1M+ training campaigns with append-only replay, exact resume, hardware-aware resource profiles, checkpoint promotion/rollback, replay/checkpoint integrity checks, and plan-only-by-default execution. v2.14 retains this machinery and adds telemetry hooks around each campaign stage.

## v2.12 controlled modular scaling

v2.12 turns the empirical-scaling stack into a controlled scaling laboratory. It adds a genuinely sparse top-k MoE dynamics trunk, matches its active trunk MACs against the monolithic world model, and runs resumable model/data/context sweeps with immutable config hashes and multi-seed bootstrap confidence intervals. The temporal world-model trainer also supports FP32, BF16, and CUDA FP16 execution so scaling studies can use the same training code as the normal game stack.

New commands:

```bash
awa-v2-controlled-scaling-smoke
awa-v2-scaling-sweep runs/game.npz --config configs/v2_12_controlled_scaling.yaml --out-dir runs/v2_12_scaling
```

The default scaling configuration compares monolithic and MoE dynamics at matched active compute, traces data fractions, hidden-width scaling, and increasingly long temporal contexts, and records resumable phase state plus a provenance-bound scorecard. See `docs/UPGRADE_2_12.md`.


## v2.11 empirical scaling and robust planning

v2.11 keeps the v2.10 goal-conditioned temporal-belief architecture and strengthens the experiment around it. It adds sampled-future/CVaR planning, bootstrap world-model ensembles for epistemic uncertainty, a canonical hybrid action codec, append-only sharded replay with integrity hashes, iterative planner-assisted data aggregation, and multi-seed bootstrap confidence intervals.

New qualification command:

```bash
awa-v2-empirical-scaling-smoke
```

See `docs/UPGRADE_2_11.md`.


## v2.10 real adaptation qualification

v2.10 closes the highest-priority integration/correctness gaps found by auditing the v2.9 game lab. Curriculum stages are no longer task semantics or policy inputs; explicit objective programs, a separate goal vector, validated procedural worlds, and subgoal-aware shaping define what the agent must do.

The actual game training path now uses Aether's local/global temporal belief encoder. The world model, risk head, actor and planner operate in belief space; a dedicated terminal value head is trained for planner bootstrap; structural replay weights, objective-aware hindsight rows and exact counterfactual branches can feed SGD. Transfer evaluation distinguishes zero-shot, frozen-weight in-context, and learning adaptation, and records real prediction error.

New qualification command:

```bash
awa-v2-game-adaptation-smoke
```

See `docs/UPGRADE_2_10.md`.


## v2.9 procedural game lab and end-to-end learning qualification

v2.9 closes the largest remaining empirical gap in the v2.x line: Aether now ships with a deterministic, dependency-free procedural game laboratory and a complete data-to-training path rather than only abstract control benchmarks.

Core additions:

- fixed-contract 32-D structured game observation and 4-D continuous action space
- 12-stage procedural game curriculum covering movement, obstacles, collection, memory, moving hazards, combat, resources, tactics, key/door sequences, changed dynamics, procedural worlds, and compositions
- exact environment snapshot/restore for counterfactual branches
- deterministic RGB rendering with visual-OOD texture/lighting variation
- hidden dynamics parameters so changed physics must be inferred from transition history
- logical demonstration teacher and random exploration policy
- procedural game dataset generation with rewards, terminal flags, constraint labels, episode IDs and metadata
- direct integration with `ReusableLearningEngine` for surprise, novelty, replay, hindsight, skill discovery and resumable learning state
- end-to-end structured-game trainer: dataset -> identity feature cache -> multimodal world model -> horizon calibration -> explicit risk model -> TD3+BC actor -> policy-seeded MPPI evaluation
- held-out game transfer evaluation across seen, visual OOD, layout OOD, dynamics OOD, compositional OOD and task OOD
- new CLIs: `awa-v2-game-lab-generate`, `awa-v2-game-train`, `awa-v2-game-loop-smoke`, and `awa-v2-game-lab-smoke`

Quick game-lab smoke:

```bash
awa-v2-game-loop-smoke
awa-v2-game-lab-smoke
```

See `docs/UPGRADE_2_9.md`.


## v2.8 hardening, checkpoint integrity, and async/game-runtime fixes

v2.8 is a review/debug release. It does not add another cognitive subsystem; it fixes issues found by auditing the actual v2.7 archive and adds regression tests for every reproduced defect.

Core fixes:

- executable skill-policy weights and learned skill contracts now survive reusable-learning checkpoint/restore
- reusable-learning checkpoints are JSON-safe even when metadata contains NumPy arrays/scalars or tensors
- adaptive weak-concept task generation advances a resumable counter instead of repeating the same task batch
- novelty memory tolerates different latent/goal widths across tasks
- EngramLite filters dimension-incompatible bucket collisions
- future-policy rollout envelopes are rejected instead of being treated as fresh
- exact branching and planner-benefit collection support Gymnasium 5-tuple step/reset semantics
- branching can receive an explicit start observation rather than relying on private environment methods
- local rollout workers close environments reliably
- recovery actions are bounded and risk-evaluated; the decision controller fails closed when no safe recovery exists
- non-finite uncertainty/risk values are treated as unsafe
- prediction-error code rejects accidental broadcasting and non-finite transitions
- world-model replay export excludes hindsight rows by default and regroups interleaved async episodes when metadata permits
- transfer `episodes_to_target` is now an exposure count rather than a zero-based episode label
- teacher selection rejects non-finite candidate utilities/actions
- skill/contract/distillation trainers validate dimensions, batch sizes, weights, and epoch/batch settings
- structural replay rejects invalid priorities and malformed sampling requests

New check:

```bash
awa-v2-hardening-smoke
pytest -q tests/test_v280_hardening.py
```

See `docs/UPGRADE_2_8.md`.


## v2.7 closed-loop transfer and continual learning

v2.7 closes the largest practical gap in v2.6: the reusable-learning pieces are now wired into a resumable closed-loop coordinator and can feed analyzed experience back into the existing Aether offline training contract.

Core additions:

- named deterministic task-verifier registry for synthesized tasks
- hindsight/future-goal relabeling for goal-conditioned experience
- strict exploit / explore / qualification modes; intrinsic curiosity reward is disabled in qualification
- learned skill precondition / termination / failure contracts
- dedicated bounded continuous skill policies distilled from successful clustered trajectories
- six-way transfer benchmark: seen, visual OOD, layout OOD, dynamics OOD, compositional OOD, and task OOD
- concept-level competence tracking plus adaptive task generation around weak concepts
- deterministic local rollout worker pool emitting policy-versioned rollout envelopes
- `ReusableLearningEngine` coordinating curriculum, structural replay, novelty, skills, distillation, sparse memory, and reasoning effort
- atomic reusable-learning checkpoint/restore without unsafe pickle deserialization
- latent replay export back into Aether's standard NPZ world-model/actor training format

New checks:

```bash
awa-v2-closed-loop-smoke
awa-v2-transfer-smoke
awa-v2-skill-contract-smoke
awa-v2-rollout-pool-smoke
pytest -q tests/test_v270_closed_loop_learning.py
```

See `docs/UPGRADE_2_7.md`.



## v2.6 reusable learning engine

v2.6 moves the project from "learn this task" toward "extract reusable structure from experience." It does not replace the v2.5 video/world/planner stack; it adds a training layer around it.

Core additions:

- procedural `TaskSpec` factory that randomizes both appearance and environment rules
- adaptive learning-frontier curriculum scheduler with a 50/25/15/10 current/mastered/harder/novel mix helper
- experience records that keep novelty, failure, surprise/prediction error, uncertainty, risk, TD error, task importance, and information value separate
- structural replay priority combining TD error + world-model prediction error + novelty + failure + task/information value
- exact-state counterfactual action branching for cleaner causal data
- continuous reasoning-effort selection over actor/shallow/medium/deep/strategic compute budgets
- heterogeneous teacher pool for MPPI/iCEM/beam/human/scripted/older-checkpoint distillation
- automatic trajectory-signature skill discovery, qualification registry, and skill composition
- `EngramLite`, a bounded sparse pattern store for reusable regimes/tactics/failure patterns
- bounded temporal-state reconstruction via periodic context checkpoints + replay
- asynchronous rollout envelopes with explicit policy-version staleness downweighting/rejection

New checks:

```bash
awa-v2-curriculum-smoke
awa-v2-reusable-smoke
pytest -q tests/test_v260_reusable_learning.py
```

The DeepSeek-V4.1-Flash report influenced the automated task/environment pipeline, controllable reasoning effort, on-policy distillation, sparse conditional-memory direction, and stale asynchronous rollout handling. These are adapted as small research components; Aether does not copy DeepSeek's LLM architecture. The Bonsai 27B whitepaper is treated as a deployment reference for a future/local semantic reasoner only. Aether's world model, control policy, and planners remain separate from the language model.

See `docs/UPGRADE_2_6.md`.


## v2.5 pretrained game-video perception

v2.5 adds a first-class path from raw game frames into the same qualified Aether world-model/actor/planner stack. The expensive video backbone is frozen and cached; Aether trains only its own projection, dynamics, actor, risk, and planning layers.

Supported perception paths:

- Hugging Face V-JEPA 2 through `AutoVideoProcessor` + `AutoModel`
- Meta V-JEPA 2.1 through a local-first PyTorch-Hub checkout
- dependency-free `toy-video` backbone for CI/smoke qualification
- optional structured telemetry concatenated with frozen visual features before the Aether projection

New commands:

```bash
# Dependency-free clip/cache/streaming qualification
awa-v2-game-repr-smoke

# Real V-JEPA 2, local weights/cache by default
pip install -e '.[vjepa]'
awa-v2-game-cache game_transitions.npz \
  --provider vjepa2-hf \
  --model-name-or-path facebook/vjepa2-vitl-fpc64-256 \
  --cache-dir .awa-game-cache

# Then reuse the normal world-model pipeline
awa-v2-world-qualify game_transitions.npz \
  --cache-dir .awa-game-cache \
  --index .awa-game-cache/<dataset-hash-prefix>/game-index.json \
  --out-dir runs/game_world
```

The live path uses `StreamingGameFeatureEncoder` with the same causal clip semantics as offline caching. `AetherV2Agent.observe_latent` lets runtime video inference remain outside the recurrent agent so deployment can use the exact projector stored in the qualified world checkpoint.

No V-JEPA weights are bundled in this repository, and the synthetic game smoke is only a plumbing/ABI test. See `docs/UPGRADE_2_5.md`.

## v2.4 closed-loop benchmark qualification

v2.4 connects saved world-model and actor artifacts to a reproducible multi-seed closed-loop benchmark protocol. The actor and every planner are evaluated on identical seed sets; search budgets are normalized in approximate world-model transition calls; reports include paired return gains, win rates, latency, calls per decision, search rate, and compute-normalized return.

State-vector benchmarks can use the new fingerprinted `identity` representation end to end. The CLI refuses to treat a vision checkpoint as identity features; vision evaluation must provide the exact frozen feature encoder through the Python API.

New commands:

```bash
awa-v2-actor-train dataset.npz \
  --cache-dir .awa-representation-cache \
  --index .awa-representation-cache/<hash>/index.json \
  --world-checkpoint runs/v2_2_world/world_model.pt \
  --out runs/v2_4_actor/actor.pt

awa-v2-benchmark-qualify \
  --world-checkpoint runs/v2_2_world/world_model.pt \
  --actor-checkpoint runs/v2_4_actor/actor.pt \
  --env-config configs/v2_4_smoke.yaml \
  --seeds 0,1,2,3,4 \
  --world-calls 128

awa-v2-benchmark-smoke
```

See `docs/UPGRADE_2_4.md`.

## v2.3 planner-qualification milestone

v2.3 makes online planning an experimentally accountable subsystem. It adds a strong actor-only TD3+BC-style baseline, clean vanilla CEM/MPPI controls, separate policy-seeded MPPI/iCEM backends, exact environment snapshot branching, real counterfactual planner-benefit datasets, supervised Value-of-Computation fitting, and a fail-closed learned-risk action guard.

New commands:

```bash
awa-v2-actor-smoke --epochs 8
awa-v2-planner-qualify --world-calls 64 --states 16 --branch-horizon 3
awa-v2-safety-smoke
awa-v2-arena --world-calls 64
```

The bundled planner qualification uses an exact analytical model of the dependency-free `ContinuousPointEnv` to validate the measurement machinery. Those smoke numbers are **not** learned-agent capability claims. Real planner promotion is based on real branch return per world-model call / latency on declared benchmark tasks.

See `docs/UPGRADE_2_3.md`.

## v2.2 world-model qualification milestone

v2.2 turns the v2 world model from an architectural component into a measurable offline training/qualification pipeline. It consumes the exact frozen-feature cache/index produced by v2.1, constructs episode-safe contiguous sequences, trains multimodal action-conditioned dynamics, measures prediction quality by horizon, calibrates uncertainty against realized rollout error, and trains a separate explicit constraint-risk model when labels are present.

New commands:

```bash
awa-v2-world-smoke
awa-v2-risk-smoke
awa-v2-world-qualify dataset.npz \
  --cache-dir .awa-representation-cache \
  --index .awa-representation-cache/<dataset-hash-prefix>/index.json \
  --out-dir runs/v2_2_world
awa-v2-promote baseline.json candidate.json
```

Qualification reports include one-step mixture NLL, reward RMSE, continuation Brier score, latent rollout RMSE at declared horizons, trajectory-ranking correlation, uncertainty calibration error by horizon, and—when explicit labels exist—risk Brier/accuracy/calibration metrics. Synthetic smoke results prove execution only; they are not capability claims.

See `docs/UPGRADE_2_2.md`.


---

# Aether World Agent v1.8.0

A runnable research framework for hierarchical latent world-model agents. The project is built around one rule: **components are promoted by controlled benchmarks, not by architectural fashion**.

## v1.8 highlights

- optional distributional continuous control with **quantile twin-Q critics**
- configurable **n-step real replay targets** for continuous critics
- conservative target quantile bootstrapping with EMA target critics
- multi-step real-environment planner-vs-actor branch evaluation
- planner-benefit gate fitting now accepts configurable branch horizons and explicit compute cost
- backend-neutral cache contract upgraded with **native-cache hooks** and bounded-recompute fallback
- Mamba adapter probes native `step` / inference-cache capabilities without depending on a single private API layout
- multi-seed environment qualification with per-seed failure reporting and snapshot round-trip checks
- all v1.7 calibrated uncertainty, learned arbitration, discrete/continuous control, memory, expert, checkpoint, and MaleCNS features retained

## Architecture

```text
observation
   ↓
encoder
   ↓
stochastic belief state ───────── episodic memory
   ↓
fast dynamics ──→ slow context
   ↓
optional sparse dynamics experts
   ↓
reward / continuation / value / latent predictions
   ↓
action-conditioned uncertainty ensemble
   ↓
actor ─────────────── CEM latent planner
  │                     │
  │             measured planning benefit
  │                     ↓
  └────────── learned arbitration
                 ↓
       calibrated reliability gate
                 ↓
       validator / reflex layer
                 ↓
               action
```

For continuous control the critic path can use scalar twin-Q or distributional quantile twin-Q:

```text
belief + continuous action
         ↓
      Q1 / Q2 or quantile distributions
         ↓
n-step target + EMA target critic
         ↓
critic update + optional actor regularization
```

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1
pip install -e '.[dev]'

awa-smoke
awa-continuous-smoke
pytest -q
```

Research configurations:

```bash
awa-train --config configs/v1_8_full.yaml --out runs/v1_8_discrete
awa-train --config configs/v1_8_continuous.yaml --out runs/v1_8_continuous
```

## v1.8 research tools

Calibrate transition uncertainty using realized next-state prediction errors:

```bash
awa-calibrate \
  --config configs/v1_8_continuous.yaml \
  --checkpoint runs/v1_8_continuous/checkpoint_500.pt \
  --episodes 10 \
  --out runs/calibrator.pt
```

Fit the learned actor/planner gate from measured real-environment branch outcomes:

```bash
awa-gate-fit \
  --config configs/v1_8_continuous.yaml \
  --episodes 10 \
  --branch-horizon 5 \
  --out runs/arbitrator.pt
```

Check whether an environment adapter matches the configured observation/action contract:

```bash
awa-env-check --config configs/v1_8_continuous.yaml --steps 20
awa-env-multiseed --config configs/v1_8_continuous.yaml --seeds 1,2,3,4,5 --steps 20
```

## Optional environments/backends

```bash
pip install -e '.[gym]'       # Gymnasium
pip install -e '.[dmcontrol]' # dm_control
pip install -e '.[metaworld]' # Meta-World
pip install -e '.[faiss]'     # scalable episodic memory
pip install -e '.[mamba]'     # Mamba sequence experiments
pip install -e '.[hf]'        # Hugging Face perception adapters
```

The cached sequence API is backend-neutral. v1.8 first probes backend-native incremental/inference-cache hooks and falls back to bounded context recomputation when the installed backend does not expose a compatible native path.

## MaleCNS lane

MaleCNS remains a parallel architecture-mining program rather than a compulsory cognition substrate:

```text
MaleCNS/FlyWire
      ↓
normalize graph
      ↓
biological circuit ──→ matched random/group/weight controls
      ↓
matched benchmark
      ↓
KEEP / DELETE
```

The repository does not redistribute connectome datasets.

## Scope

v1.8 is a serious experimental framework, not a claim of AGI or parity with frontier world-model systems. Its purpose is to make architecture changes testable, resumable, and attributable.


## v2.1 representation milestone

The v2 agent can now inject a frozen pretrained representation backbone instead of being tied to the MLP encoder. Expensive frozen features can be cached deterministically before the trainable projection, offline transition datasets carry SHA-256 provenance, and a geometry-preserving projection trainer supports label-free projection warm-up. The base install remains runnable without downloading a large checkpoint.

Useful commands:

```bash
awa-v2-repr-check
awa-v2-cache
awa-v2-cache path/to/offline_dataset.npz --cache-dir .awa-representation-cache
```

See `docs/UPGRADE_2_1.md` for the exact cache and dataset contracts.


## v2.13 durable training campaign

v2.13 adds the operational layer needed to run Aether as a cumulative, resumable training experiment rather than a collection of one-off commands. A single campaign can grow append-only replay through 100k, 250k, 500k and 1M transition gates, train the same temporal-belief/world/actor stack at each gate, qualify candidates, preserve the last stable checkpoints, and resume after interruption without regenerating committed replay.

The safe default is planning only:

```bash
awa-v2-training-campaign --config configs/v2_13_training_campaign.yaml
```

Actual collection/training requires an explicit execution flag:

```bash
awa-v2-training-campaign --config configs/v2_13_training_campaign.yaml --out-dir runs/v2_13_campaign --execute
```

Use `awa-v2-training-campaign-smoke` to qualify collection, training, checkpoint promotion, integrity checks and exact resume on a tiny CPU run. See `docs/UPGRADE_2_13.md` for the contracts and limitations.
