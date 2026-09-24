# v1.6.0 — Continuous Control

v1.6 closes the largest known functional gap in v1.5: continuous control.

## Added

- `ActionSpec` unifying discrete and bounded-continuous actions.
- Squashed Gaussian (`TanhGaussianActor`) continuous actor.
- Bounded continuous CEM trajectory optimizer.
- Replay buffers supporting scalar discrete actions and vector continuous actions.
- Sequence overshooting for continuous recorded action trajectories.
- Generalized imagination actor/critic updates.
- Continuous action validation/clipping at the runtime authority boundary.
- Dependency-free `ContinuousPointEnv` for continuous CI/integration tests.
- Gymnasium Box-action adapter.
- dm_control continuous adapter.
- Continuous checkpoint/resume metadata.
- Continuous evaluation including terminal distance where available.

## Deliberate limits

This is a correct end-to-end continuous-control path, not a claim of TD-MPC2 parity. The world model remains the same RSSM-like stochastic latent model, and the included continuous benchmark is intentionally small. MuJoCo/dm_control are optional dependencies.

## Required comparisons

For serious experiments compare:

1. continuous actor only,
2. actor + continuous CEM,
3. GRU vs SSM dynamics,
4. fixed vs adaptive planning budget,
5. uncertainty gate on/off,
6. slow state on/off.
