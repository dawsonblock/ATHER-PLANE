# Aether v2.1 — Representation and Offline Data Milestone

v2.1 implements the first post-reset milestone: make perception a reproducible external dependency rather than a hard-coded toy encoder.

## Added

- `FrozenBackboneProjector`: frozen pretrained feature extractor + trainable Aether projection.
- `HuggingFaceFrozenBackbone`: optional local Transformers/AutoModel adapter; network downloads remain opt-in through caller configuration.
- `build_pretrained_representation`: provider factory for local modules and HF/DINO-family AutoModel-compatible backbones.
- `RepresentationCache`: content-addressed, atomic on-disk cache keyed by input bytes + backbone fingerprint.
- `CachedRepresentation`: caches expensive frozen features before the trainable projection.
- `OfflineTransitionDataset`: validated NPZ offline transition ingestion with dataset SHA-256 provenance.
- `GeometryPreservingProjectionTrainer`: label-free projection pretraining that preserves backbone similarity geometry and avoids collapsed dimensions.
- `AetherV2Agent(..., encoder=...)`: injection point for real pretrained representations.
- `awa-v2-cache`: deterministic feature-cache CLI.
- `awa-v2-repr-check`: representation contract/freeze smoke test.

## Why cache before projection?

Backbone inference is normally the expensive operation. Projection weights are intentionally trainable and may change frequently. Caching the frozen feature tensor allows projection/world-model experiments to iterate without repeatedly invoking the large vision backbone and without stale projected representations.

## Dataset contract

An offline `.npz` must contain:

- `observations`
- `actions`
- `rewards`
- `next_observations`
- `dones`

All arrays must have equal transition counts. `next_observations` must match the observation shape. A SHA-256 manifest binds experiments to exact dataset bytes.

## Boundary

v2.1 provides real encoder plumbing and deterministic representation caching. It does not bundle or download a large pretrained checkpoint, and it does not claim that any particular visual backbone is best. Backbone choice must be benchmarked on downstream prediction/control.
