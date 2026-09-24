from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import inspect
import torch
from torch import nn


@dataclass
class NativeMambaState:
    inference_params: Any = None
    conv_state: Any = None
    ssm_state: Any = None
    offset: int = 0


class MambaSequenceAdapter(nn.Module):
    """Optional production Mamba backend with native-cache probing.

    ``mamba_ssm`` has evolved across releases. This adapter does not hard-code a
    single undocumented cache layout. It probes for supported public/common
    incremental APIs and falls back to bounded-context recomputation through
    ``CachedSequenceAdapter`` when a compatible native path is unavailable.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as e:
            raise ImportError("Install with `pip install -e '.[mamba]'`") from e
        self.d_model = int(d_model)
        self.block = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3:
            raise ValueError("MambaSequenceAdapter expects [batch,time,features]")
        return self.block(sequence)

    def native_cache_capability(self) -> dict:
        return {
            "has_step": callable(getattr(self.block, "step", None)),
            "has_allocate_inference_cache": callable(getattr(self.block, "allocate_inference_cache", None)),
            "forward_accepts_inference_params": "inference_params" in inspect.signature(self.block.forward).parameters,
        }

    def initial_cache(self, batch_size: int | None = None, device=None, dtype=None):
        if batch_size is None:
            return NativeMambaState()
        alloc = getattr(self.block, "allocate_inference_cache", None)
        if callable(alloc):
            try:
                state = alloc(batch_size, max_seqlen=1, dtype=dtype)
                return NativeMambaState(inference_params=state)
            except TypeError:
                try:
                    state = alloc(batch_size, 1, dtype=dtype)
                    return NativeMambaState(inference_params=state)
                except Exception:
                    pass
        return NativeMambaState()

    def step_cached(self, token: torch.Tensor, state: NativeMambaState | None = None):
        state = state or NativeMambaState()
        x = token[:, 0] if token.ndim == 3 else token
        step = getattr(self.block, "step", None)
        if callable(step):
            try:
                result = step(x, state.conv_state, state.ssm_state)
                if isinstance(result, tuple) and len(result) >= 3:
                    y, conv_state, ssm_state = result[:3]
                    if y.ndim == 2:
                        y = y.unsqueeze(1)
                    return y[:, -1], NativeMambaState(
                        inference_params=state.inference_params,
                        conv_state=conv_state,
                        ssm_state=ssm_state,
                        offset=state.offset + 1,
                    )
            except (TypeError, AttributeError):
                pass

        sig = inspect.signature(self.block.forward).parameters
        if "inference_params" in sig and state.inference_params is not None:
            try:
                out = self.block(token, inference_params=state.inference_params)
                return out[:, -1], NativeMambaState(
                    inference_params=state.inference_params,
                    conv_state=state.conv_state,
                    ssm_state=state.ssm_state,
                    offset=state.offset + 1,
                )
            except (TypeError, AttributeError, RuntimeError):
                pass
        raise RuntimeError("This installed mamba_ssm backend does not expose a compatible native cache API")

    def cached(self, max_context: int = 256, prefer_native: bool = True):
        from awa.dynamics.stateful_sequence import CachedSequenceAdapter
        return CachedSequenceAdapter(self, max_context=max_context, prefer_native=prefer_native)
