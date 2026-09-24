# Aether World Agent v2.38 — Progressive Execution Orchestrator

v2.38 does not add action-time cognition. The `awa.v2.agent_runtime` Python tree is pinned byte-for-byte to v2.37. The release converts the v2.37 staged empirical roadmap into a one-phase-at-a-time, resumable execution surface intended for CUDA hosts such as RunPod.

## What changed

- `awa-v2-progressive-executor` reads the progressive evidence graph and plans only the first blocked phase.
- Execution uses argv arrays with `shell=False`; arbitrary shell fragments are not accepted.
- A transition-target guard prevents accidentally jumping from a small bring-up to a larger stage.
- P4, P6 and P7 now use focused five-seed, 25K pairwise ablations rather than requiring the entire seven-system 25K/100K matrix before the rest of bring-up can continue.
- P5 remains fail-closed until a real held-out world-model horizon curve exists.
- P9 automatically resolves stable structured world/actor checkpoints from real ViZDoom receipts and uses the empirically qualified horizon for the GPU planner schedule sweep.
- P11 DREAM-RSI is marked expensive and requires a second explicit `--allow-expensive` acknowledgement.
- P10/P12/P13/P15 remain evidence/manual gates because v2.38 will not fabricate compositional, unseen-map, external-baseline or scaling-authorization results.
- `deploy/runpod/bootstrap.sh` installs the project and runs the first plan on a CUDA pod without starting training automatically.

## RunPod path

```bash
unzip aether-world-agent-v2.38.0-full-upgraded.zip
cd aether-world-agent-v2.38.0
bash deploy/runpod/bootstrap.sh
```

Inspect the next phase:

```bash
awa-v2-progressive-executor \
  --config configs/v2_38_progressive_executor.yaml \
  --evidence-root runs/v2_38
```

Execute exactly one eligible phase:

```bash
awa-v2-progressive-executor \
  --config configs/v2_38_progressive_executor.yaml \
  --evidence-root runs/v2_38 \
  --execute-next
```

DREAM-RSI additionally requires `--allow-expensive`.

## Scientific boundary

Focused pairwise gates are development/architecture promotion gates. P14 explicitly requires the final canonical seven-system 25K/100K preregistered matrix before P15 can authorize 250K+ scaling. Real ViZDoom, real V-JEPA, planner hardware, transfer, DREAM-RSI and external-baseline results remain unexecuted until produced on the target hardware.
