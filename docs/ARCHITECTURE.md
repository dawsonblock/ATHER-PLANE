# Architecture

## Core principle

The runtime is intentionally split into:

1. semantic cognition: understands goals and decomposes tasks,
2. predictive cognition: models consequences in latent state,
3. action layer: fast actor plus expensive latent MPC,
4. memory: recurrent working state plus explicit episodic store,
5. safety/reflex: deterministic authority below learned systems.

MaleCNS is not required for the core agent. It is a source of candidate network structures.

## Data flow

```text
observation
  -> encoder
  -> belief state
  -> fast dynamics
  -> [slow dynamics]
  -> reward/value/continuation predictions
  -> actor and/or planner
  -> reliability gate
  -> action validator/reflex
  -> environment
```

## Why latent prediction

The system predicts compact task-relevant representations rather than reconstructing every pixel. Pixel decoders may be added only for debugging or specialized objectives.

## Why actor + planner

The actor provides low-latency behavior. MPC is reserved for states where deliberate search is worthwhile. Planner solutions can later be distilled into procedural skills.

## Why uncertainty

Long imagined rollouts are dangerous when the learned dynamics are outside their training coverage. Ensemble disagreement is exposed explicitly to arbitration.

## Why biology is optional

A biological circuit is accepted only if it beats parameter/degree-matched controls under the same training budget.
