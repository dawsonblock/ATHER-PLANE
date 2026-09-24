# Aether v2.28 — Remaining Work

The architecture should remain frozen until the v2.28 evidence program produces results.

## 1. Run the canonical 25K/100K campaign

Execute all 70 training jobs and 140 frozen-checkpoint evaluation cells using the five preregistered seeds. Do not modify thresholds after looking at outcomes. Preserve the campaign directory because every result is content/protocol bound.

## 2. Run the matched collector study

After selecting a stable learner, compare equal transition budgets from teacher, random, coverage, Aether actor, and Aether+planner collection. Use a frozen source checkpoint for Aether collectors and separate downstream learner seeds. The component-ablation study intentionally holds collection constant and therefore cannot answer this question.

## 3. Dedicated planner/VOC utility qualification

From identical saved environment states, measure actor-only, fixed MPPI/iCEM budgets, and adaptive VOC. Record realized return gain, planner/model calls, inference latency, risk, and how planner dependence changes with familiarity.

## 4. Compositional transfer benchmark

Create factor-isolated training sets and withheld combinations for memory, combat, resource scarcity, key/door dependency, moving hazards, and changed dynamics. Measure zero-shot success and episodes-to-recovery.

## 5. Evidence-driven reduction release

Use the v2.28 adjacent marginal report to decide the stable v2.29 spine. Delete or move to `awa.experimental` mechanisms that fail the preregistered value thresholds. Do not keep a component solely because it is architecturally interesting.

Likely cleanup candidates after evidence exists remain:

```text
skills/
training/reusable_engine.py
engram.py
ensemble.py
modular_world.py
training/modes.py
memory.py
hierarchy.py
context_replay.py
decision.py
```

## 6. Retire legacy package and CLI surface

Once the winning spine is known, move the small actor/critic primitives still consumed by v2 into neutral core modules, archive the obsolete v1 runtime behind release tags, and collapse historical entrypoints into a stable `awa` subcommand interface.

## 7. External validation

Compare the reduced winner against a strong external world-model baseline under matched transition and GPU budgets. Only then move the surviving mechanisms into structured ViZDoom and finally RGB + frozen V-JEPA.
