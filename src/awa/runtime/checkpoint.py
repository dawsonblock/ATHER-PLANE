from __future__ import annotations
from pathlib import Path
import random
import numpy as np
import torch


def _rng_payload():
    return {
        "python_rng": random.getstate(),
        "numpy_rng": np.random.get_state(),
        "torch_rng": torch.random.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(payload):
    random.setstate(payload["python_rng"])
    np.random.set_state(payload["numpy_rng"])
    torch.random.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])


class CheckpointManager:
    @staticmethod
    def save(path, model, optimizer=None, extra=None):
        """Legacy single-model checkpoint."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model": model.state_dict(), **_rng_payload(), "extra": extra or {}}
        if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
        torch.save(payload, path)

    @staticmethod
    def load(path, model, optimizer=None, map_location="cpu"):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model.load_state_dict(payload["model"])
        if optimizer is not None and "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
        _restore_rng(payload)
        return payload.get("extra", {})

    @staticmethod
    def save_bundle(path, modules: dict, optimizers: dict | None = None, extra=None):
        """Release-grade checkpoint of the complete trainable agent state."""
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "awa-bundle-v1",
            "modules": {name: module.state_dict() for name, module in modules.items()},
            "optimizers": {name: opt.state_dict() for name, opt in (optimizers or {}).items()},
            **_rng_payload(),
            "extra": extra or {},
        }
        torch.save(payload, path)

    @staticmethod
    def load_bundle(path, modules: dict, optimizers: dict | None = None, map_location="cpu", strict=True):
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("format") != "awa-bundle-v1":
            raise ValueError("Not an AWA bundle checkpoint")
        for name, module in modules.items():
            if name not in payload["modules"]:
                if strict: raise KeyError(f"Missing module in checkpoint: {name}")
                continue
            module.load_state_dict(payload["modules"][name], strict=strict)
        for name, opt in (optimizers or {}).items():
            if name in payload.get("optimizers", {}): opt.load_state_dict(payload["optimizers"][name])
            elif strict: raise KeyError(f"Missing optimizer in checkpoint: {name}")
        _restore_rng(payload)
        return payload.get("extra", {})
