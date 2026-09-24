# Aether World Agent v2.30 — Empirical Pivot

v2.30 freezes the intelligence architecture and moves the project from framework expansion toward externally grounded evidence. It is intentionally not another cognitive-feature release.

## Why this release exists

Aether already had strong experiment governance, resumable campaigns, model-based planning, adaptive compute, and a preregistered seven-system ablation ladder. The remaining weakness was empirical: most release validation still happened in synthetic/2-D environments or through fake/injected ViZDoom and V-JEPA test doubles. v2.30 makes that boundary explicit and adds the missing infrastructure required to cross it without weakening reproducibility.

## 1. Fail-closed empirical maturity ladder

`awa.v2.empirical_status` defines tiers:

- T0 — continuous-point mathematical sanity;
- T1 — procedural 2-D arena integration;
- T2 — real ViZDoom structured-state execution;
- T3 — real ViZDoom RGB execution;
- T4 — real V-JEPA visual-representation execution;
- T5 — unseen ViZDoom transfer;
- T6 — compositional adaptation/generalization.

The shipped `EMPIRICAL_STATUS.md` and `EMPIRICAL_STATUS.json` do not infer real-world maturity from unit tests. T2–T5 remain `UNEXECUTED` in the release artifact. T6 task generation is implemented but not empirically qualified.

Use:

```bash
awa-v2-empirical-status \
  --evidence-root runs \
  --json-output runs/EMPIRICAL_STATUS.json \
  --markdown-output runs/EMPIRICAL_STATUS.md
```

A tier is promoted only by matching evidence artifacts.

## 2. Real ViZDoom qualification path

`awa-v2-vizdoom-real-qualify` deliberately refuses the injected `DoomGame` objects used by unit tests. A qualifying backend must be created through the installed `vizdoom` package and expose native provenance.

Structured qualification:

```bash
pip install -e ".[dev,vizdoom]"
awa-v2-vizdoom-real-qualify \
  --out-dir runs/real-doom-structured \
  --scenario my_way_home \
  --track structured \
  --episodes 8
```

Pixel + V-JEPA qualification:

```bash
pip install -e ".[dev,doom-vjepa]"
awa-v2-vizdoom-real-qualify \
  --out-dir runs/real-doom-pixel \
  --scenario my_way_home \
  --track pixel \
  --episodes 8 \
  --require-vjepa \
  --device cuda
```

`real_vizdoom_qualification.json` records backend module/class/package provenance, dataset SHA-256, actual transition count, and—when requested—the non-injected V-JEPA class/fingerprint/feature shape. The receipt proves that the real path executed; it does not claim benchmark superiority.

## 3. Factorized environment/data factory

`FactorizedEnvironmentFactory` separates compositional evaluation from the old stage curriculum. The controlled factors are:

- navigation;
- memory;
- combat;
- scarcity;
- key/door dependency;
- hazards;
- dynamics shift.

The canonical benchmark gives train, heldout-composition, and transfer-composition sets disjoint exact factor signatures. Every generated task is instantiated in `ProceduralArenaEnv` and checked for finite observations/goals and reachable objective targets before entering the manifest.

Generate it with:

```bash
awa-v2-compositional-benchmark \
  --output runs/compositional_benchmark.json
```

The release ships `COMPOSITIONAL_BENCHMARK_2_30.json` as the canonical task manifest. It is a task-generation artifact, not a performance result.

## 4. Real planner schedule benchmark

v2.29 separated logical planner work from physical world-model forward calls. v2.30 adds a direct hardware benchmark runner for a trained procedural checkpoint:

```bash
awa-v2-planner-hardware-benchmark \
  --world-checkpoint runs/model/game_world.pt \
  --actor-checkpoint runs/model/game_actor.pt \
  --device cuda \
  --batch-limits 0,32,8,4,1 \
  --output runs/planner_schedule_gpu.json
```

For every batch limit it reports matched logical work, physical forward count, mean/median/P95 latency, peak CUDA allocation, effective world-model batch size and an action-sequence hash. Energy remains `not_measured`; Aether does not derive joules from TDP or wall time.

## 5. Dead-wood quarantine

The old v2 prototype modules `memory.py`, `hierarchy.py`, `context_replay.py`, and `decision.py` are no longer treated as part of the stable v2.30 spine. Their historical implementations live under `awa.experimental.legacy_v2`; old imports remain compatibility shims and emit `DeprecationWarning`.

The historical Drosophila/connectome implementation is similarly moved under `awa.experimental.connectome`, with `awa.connectome.*` retained as compatibility shims. This preserves reproducibility while making the active architecture clearer.

The v1 runtime remains in this release because several regression tests and compatibility CLIs still depend on it. It is explicitly compatibility-only, not evidence for the v2.30 architecture.

## 6. Preregistered ablation retained

The v2.29 logical/physical compute contract is retained and versioned to the v2.30 protocol. The canonical config is:

```text
configs/v2_30_empirical_pivot.yaml
```

Run the 25K/100K matrix before adding another cognitive module:

```bash
awa-v2-ablation-campaign \
  --config configs/v2_30_empirical_pivot.yaml \
  --out-dir runs/v2_30_ablation \
  --execute
```

## What v2.30 does not claim

The release does not contain:

- a completed five-seed 25K/100K ablation matrix;
- measured A100/H100/4090 schedule results;
- measured GPU energy;
- a real ViZDoom benchmark result;
- a real V-JEPA downstream training result;
- a five-seed compositional-transfer result;
- an external world-model-agent comparison.

Those are now the next experiments rather than additional framework code.
