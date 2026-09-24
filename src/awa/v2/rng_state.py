from __future__ import annotations

import random
from typing import Any

import numpy as np
import torch


RNG_STATE_FORMAT = "awa-v2.27-rng-state-v1"


def capture_rng_state() -> dict[str, Any]:
    """Capture process RNG state required for deterministic training continuation.

    The payload is intended for torch.save checkpoints, not JSON. CUDA states are
    captured only when CUDA is available. DataLoader shuffling without an explicit
    generator consumes the torch CPU RNG and is therefore covered by torch_cpu.
    """
    return {
        "format": RNG_STATE_FORMAT,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state().clone(),
        "torch_cuda": [x.clone() for x in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> bool:
    """Restore a state emitted by :func:`capture_rng_state`.

    Returns False for a missing state so older checkpoints remain loadable. A
    present but incompatible state fails closed rather than silently changing the
    stochastic continuation.
    """
    if state is None:
        return False
    if state.get("format") != RNG_STATE_FORMAT:
        raise ValueError("unsupported RNG state format")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # torch.load(map_location="cuda") also maps the CPU generator state to CUDA.
    # Both RNG setter APIs require ByteTensors in host memory.
    torch.random.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda")
    if cuda_states is not None:
        if not torch.cuda.is_available():
            raise ValueError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    return True
