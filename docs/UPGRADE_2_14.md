# Aether v2.14 — Empirical Qualification & Real Game Bridge

v2.14 does not change the core belief/world/actor/planner algorithms. It adds the missing boundary and qualification systems needed to move from the built-in procedural arena to external games while keeping experiments auditable.

## External game bridge

`awa.v2.game.bridge_protocol` defines the versioned `aether.game.v1` JSONL protocol. Required operations are `hello`, `reset`, `step`, and `close`. Snapshot/restore is optional but required for exact counterfactual branches. The Python client exposes a Gym-like `BridgeEnvironmentAdapter`.

Use:

```bash
awa-v2-game-bridge-check --host 127.0.0.1 --port 8765
```

See `docs/GAME_BRIDGE_PROTOCOL.md` and `integrations/unity/AetherBridgeReference.cs`.

## Rollout processes

`ProcessArenaRolloutPool` runs built-in arena episodes in true processes rather than threads. It uses serialized `TaskSpec` inputs and built-in teacher/random policies so opaque callables are not pickled across workers.

## Telemetry

`TelemetryRecorder` writes append-only, fsync'd JSONL events. v2.13 training campaigns now emit stage start, collection completion, training completion, and promotion events by default. Non-finite numeric telemetry is rejected.

## Failure triage

`triage_failure()` performs evidence-based heuristic classification into perception, world-model, memory, actor, planner, skill, goal, risk, environment, or unknown. It is diagnostic evidence, not causal proof.

## Structured vs pixel qualification

`RepresentationContract` and `BenchmarkManifest` prevent structured-state and pixel benchmarks from being silently combined. Pixel tracks require an explicit encoder name/fingerprint, frame shape, and clip length.

## Ablations

`run_ablation_suite()` standardizes paired-seed ablations for temporal memory, planner, skills, ensemble, replay priority, counterfactual data, hindsight, and VOC. Results include bootstrap confidence intervals and paired deltas versus the full system.

## Milestone reports

`awa-v2-qualification-report` consumes JSONL rows for 100k/250k/500k/1M checkpoints and produces JSON + Markdown scorecards with bootstrap intervals and desired trend checks for success, held-out performance, prediction error, planner dependency, skill reuse, adaptation speed, and compute per success.

## Methodological boundary

v2.14 qualifies interfaces and measurement infrastructure. It does not claim that any specific 100k–1M training curve has been achieved until an actual campaign produces those results.
