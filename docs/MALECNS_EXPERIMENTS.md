# MaleCNS Experimental Lane

Never assume the exact fly topology is superior.

For every extracted candidate circuit:

1. Biological topology.
2. Degree-preserving randomized topology.
3. Generic sparse learned baseline.
4. Fully learned topology where feasible.

Match:
- units,
- parameter count,
- input/output dimensions,
- training data,
- optimizer,
- seeds,
- wall-clock or optimizer-step budget.

Measure:
- final task success,
- learning speed,
- robustness to sensor dropout,
- perturbation tolerance,
- compute,
- generalization.

Only promote a biological circuit if it wins on a declared metric.
