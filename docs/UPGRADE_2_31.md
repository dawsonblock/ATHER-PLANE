# Aether World Agent v2.31.0 — Proper System Split

v2.31 is an architectural-boundary release. It does not add a new cognitive mechanism.
Its purpose is to stop treating the agent, the machinery that trains the agent, and the
research governance around both as one undifferentiated runtime.

## Canonical layers

### 1. Agent Runtime

`awa.v2.agent_runtime` is the only canonical inference-time cognition surface.
It owns perception adapters, temporal belief, actor execution, learned world dynamics,
selective MPPI/iCEM planning, value-of-computation, uncertainty/risk handling,
hybrid action semantics, and the final safety gate.

It must not own or import dataset construction, optimizer updates, curriculum mutation,
experiment promotion, provenance ledgers, or release governance.

### 2. Learning System

`awa.v2.learning_system` owns experience collection, datasets, training, grounded
hindsight replay, factorized environment/task generation, and curriculum allocation.
It may consume the agent runtime because training and autonomous collection need the
agent's models. The reverse dependency is forbidden.

### 3. Research OS

`awa.v2.research_os` owns preregistration, provenance, resumable experiment execution,
train-once/evaluate-many qualification, compute accounting, ablation consolidation,
empirical maturity status, and keep/remove decisions.

The research OS may inspect both lower layers, but action-time agent code cannot depend on it.
Evaluation governance is explicitly prohibited from mutating agent behavior while evidence is
being collected.

## Dependency direction

```text
agent_runtime
     ↑
learning_system
     ↑
 research_os

experimental families are outside the canonical chain
```

More precisely:

```text
agent_runtime   -> agent_runtime + shared only
learning_system -> agent_runtime + learning_system + shared
research_os     -> agent_runtime + learning_system + research_os + shared
```

`awa-v2-split-check` validates this contract over the new canonical split packages and fails
closed on a forbidden import edge.

## Compatibility strategy

v2.31 does not perform a risky repository-wide move of every historical flat module. The
existing `awa.v2.*` implementation modules remain compatibility internals, while all new
canonical code should enter through one of the three split packages. This gives the project a
stable migration boundary without invalidating the v2.30 empirical baseline before the first
real benchmark campaign.

## What is deliberately not in the canonical agent

The following remain experimental/quarantined and are not credited as part of the agent:

- skill discovery/composition machinery
- Engram-style memory experiments
- sparse-MoE / modular-world experiments
- broad reusable-learning engine
- connectome experiments
- legacy v2 prototype families

They can return only through an isolated executable ablation that beats the reduced core.

## Canonical configuration

`configs/v2_31_system_split.yaml` preserves the v2.30 empirical-pivot experiment while
adding explicit `agent_runtime`, `learning_system`, and `research_os` responsibility sections.
The 25K/100K seven-system ablation remains unchanged in scientific intent.

## Validation

The release adds a dedicated split-contract test suite covering version/config binding,
dependency direction, public-surface separation, and CLI fail-closed behavior.
