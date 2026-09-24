# Aether Empirical Status

This file separates implementation maturity from empirical evidence. A passing unit test or synthetic smoke is not treated as proof of real-world control performance.

| Tier | Capability | Status | Evidence / boundary |
|---|---|---|---|
| T0 | continuous-point mathematical sanity | **SOFTWARE_VALIDATED** | source tests and smoke coverage. Useful for mathematical/controller sanity only; not evidence of complex embodied control. |
| T1 | procedural 2-D arena integration | **SOFTWARE_VALIDATED** | native procedural tests and campaign smokes. The shipped release does not include the canonical 25K/100K five-seed empirical result. |
| T2 | real ViZDoom structured-state control | **UNEXECUTED** | none. Requires a qualification receipt produced by the non-injected real ViZDoom path. |
| T3 | real ViZDoom RGB control | **UNEXECUTED** | none. Requires a real ViZDoom pixel-track qualification receipt. |
| T4 | real V-JEPA visual representation path | **UNEXECUTED** | none. Requires a non-injected V-JEPA model and real ViZDoom RGB frames. |
| T5 | unseen ViZDoom scenario/map transfer | **UNEXECUTED** | none. Requires a heldout real-ViZDoom transfer evaluation; no shipped result is assumed. |
| T6 | compositional adaptation/generalization | **IMPLEMENTED** | factorized compositional benchmark generator. Task generation is implemented; a complete multi-seed result is not shipped. |

Boundary: Implementation, unit tests, software smokes, hardware execution and empirical qualification are separate states. The status report never promotes a tier without corresponding evidence artifacts.
