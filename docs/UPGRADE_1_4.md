# v1.4 upgrade

v1.4 moves AWA from a synthetic-only architecture scaffold toward a research harness that can accept real environments, large pretrained perception, scalable memory, and controlled connectome ingestion without making those optional dependencies mandatory.

Key changes:

- EMA target value network for more stable imagined-return bootstrapping.
- Continuation-weighted, advantage-normalized imagination actor objective.
- Optional Gymnasium discrete environment adapter.
- Optional dm_control observation adapter (continuous actor intentionally deferred).
- Optional Mamba sequence backend with an explicit non-drop-in contract.
- FAISS-backed vector episodic memory with NumPy fallback.
- Learned actor/planner arbitration module for later calibration/training.
- Generalization suite: longer memory horizon, noise, dropout, combined shift.
- Automated Markdown ablation reports.
- MaleCNS/FlyWire connection-table alias autodetection.

The default online dynamics remains the dependency-free GRU/SSM cell. This is deliberate: a Mamba sequence block is not falsely presented as equivalent to a one-step recurrent state update.
