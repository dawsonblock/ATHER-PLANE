from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import inspect
import torch
from torch import nn


@dataclass
class SequenceCache:
    context: torch.Tensor | None = None
    tokens_seen: int = 0
    native_state: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CachedSequenceAdapter(nn.Module):
    """Stable online cache interface for sequence-native backends.

    The generic fallback keeps a bounded detached context and recomputes it.
    If a backend exposes an incremental/native cache API, set ``prefer_native``
    and the adapter will use it when possible while preserving the same public
    ``step(token, cache)`` contract.
    """
    def __init__(self, backend: nn.Module, max_context: int = 256, prefer_native: bool = True):
        super().__init__()
        self.backend = backend
        self.max_context = int(max_context)
        self.prefer_native = bool(prefer_native)
        if self.max_context <= 0:
            raise ValueError("max_context must be positive")

    def initial_cache(self, batch_size: int | None = None, device=None, dtype=None) -> SequenceCache:
        native = None
        if self.prefer_native and hasattr(self.backend, "initial_cache"):
            native = self.backend.initial_cache(batch_size=batch_size, device=device, dtype=dtype)
        return SequenceCache(native_state=native)

    def forward(self, sequence: torch.Tensor):
        return self.backend(sequence)

    def _native_step(self, token: torch.Tensor, cache: SequenceCache):
        if not self.prefer_native or not hasattr(self.backend, "step_cached"):
            return None
        out, native_state = self.backend.step_cached(token, cache.native_state)
        return out, SequenceCache(
            context=None,
            tokens_seen=cache.tokens_seen + 1,
            native_state=native_state,
            metadata={**cache.metadata, "backend": "native"},
        )

    def step(self, token: torch.Tensor, cache: SequenceCache | None = None):
        if token.ndim == 2:
            token = token.unsqueeze(1)
        if token.ndim != 3 or token.shape[1] != 1:
            raise ValueError("token must be [B,D] or [B,1,D]")
        cache = cache or self.initial_cache(batch_size=token.shape[0], device=token.device, dtype=token.dtype)

        native = self._native_step(token, cache)
        if native is not None:
            return native

        context = token if cache.context is None else torch.cat([cache.context, token], dim=1)
        if context.shape[1] > self.max_context:
            context = context[:, -self.max_context :]
        out = self.backend(context)
        return out[:, -1], SequenceCache(
            context=context.detach(),
            tokens_seen=cache.tokens_seen + 1,
            native_state=None,
            metadata={**cache.metadata, "backend": "bounded_recompute"},
        )
