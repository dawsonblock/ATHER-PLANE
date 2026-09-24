# v1.5.0 Upgrade

v1.5 turns the v1.4 scaffold into a stricter experimental system.

Key upgrades:
- burn-in aware contiguous sequence training
- multi-step latent overshooting objective
- environment factory with trainable Gymnasium discrete adapters
- adaptive uncertainty-dependent MPC budget
- optional logistic uncertainty calibration
- learned actor-vs-planner gate interface
- bounded/deduplicating FAISS-or-NumPy episodic vector memory
- experiment-matrix runner with CSV/JSON/Markdown and 95% CIs
- additional connectome controls: weight permutation and group-preserving rewiring
- differentiable fixed-topology connectome circuit exporter
- source-file provenance manifests with SHA-256

The core policy remains: biology and architectural complexity must beat matched controls before promotion.
