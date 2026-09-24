from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json

import torch
from torch import nn


def _stable_tensor_bytes(x: torch.Tensor) -> bytes:
    y = x.detach().cpu().contiguous()
    header = f"{str(y.dtype)}|{tuple(y.shape)}|".encode()
    return header + y.numpy().tobytes()


def tensor_sha256(x: torch.Tensor) -> str:
    return hashlib.sha256(_stable_tensor_bytes(x)).hexdigest()


def module_fingerprint(module: nn.Module) -> str:
    h = hashlib.sha256()
    h.update(module.__class__.__module__.encode())
    h.update(module.__class__.__qualname__.encode())
    for name, value in sorted(module.state_dict().items()):
        h.update(name.encode())
        h.update(_stable_tensor_bytes(value))
    return h.hexdigest()


@dataclass(frozen=True)
class RepresentationRecord:
    key: str
    input_sha256: str
    encoder_fingerprint: str
    shape: tuple[int, ...]
    dtype: str


class RepresentationCache:
    """Content-addressed on-disk cache for deterministic representation embeddings.

    A cache key binds the raw tensor bytes, the encoder fingerprint and an optional
    namespace. Cached tensors are written atomically, so interrupted writers do not
    leave valid-looking partial entries.
    """

    def __init__(self, root: str | Path, namespace: str = "default"):
        self.root = Path(root)
        self.namespace = namespace
        self.data_dir = self.root / namespace / "tensors"
        self.meta_dir = self.root / namespace / "metadata"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def make_key(input_sha256: str, encoder_fingerprint: str, namespace: str = "default") -> str:
        raw = f"awa-repr-v1|{namespace}|{encoder_fingerprint}|{input_sha256}".encode()
        return hashlib.sha256(raw).hexdigest()

    def key_for(self, x: torch.Tensor, encoder_fingerprint: str) -> tuple[str, str]:
        input_hash = tensor_sha256(x)
        return self.make_key(input_hash, encoder_fingerprint, self.namespace), input_hash

    def get(self, x: torch.Tensor, encoder_fingerprint: str) -> torch.Tensor | None:
        key, _ = self.key_for(x, encoder_fingerprint)
        path = self.data_dir / f"{key}.pt"
        meta = self.meta_dir / f"{key}.json"
        if not path.exists() or not meta.exists():
            return None
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("encoder_fingerprint") != encoder_fingerprint:
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def put(self, x: torch.Tensor, encoder_fingerprint: str, embedding: torch.Tensor) -> RepresentationRecord:
        key, input_hash = self.key_for(x, encoder_fingerprint)
        target = self.data_dir / f"{key}.pt"
        tmp = target.with_suffix(".tmp")
        torch.save(embedding.detach().cpu(), tmp)
        tmp.replace(target)
        record = RepresentationRecord(
            key=key,
            input_sha256=input_hash,
            encoder_fingerprint=encoder_fingerprint,
            shape=tuple(int(v) for v in embedding.shape),
            dtype=str(embedding.dtype),
        )
        meta_target = self.meta_dir / f"{key}.json"
        meta_tmp = meta_target.with_suffix(".tmp")
        meta_tmp.write_text(json.dumps(record.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        meta_tmp.replace(meta_target)
        return record

    def get_hashed(self, input_sha256: str, encoder_fingerprint: str) -> torch.Tensor | None:
        """Load an entry when the caller already computed a stable input digest."""
        key = self.make_key(input_sha256, encoder_fingerprint, self.namespace)
        path = self.data_dir / f"{key}.pt"
        meta = self.meta_dir / f"{key}.json"
        if not path.exists() or not meta.exists():
            return None
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("encoder_fingerprint") != encoder_fingerprint or record.get("input_sha256") != input_sha256:
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def put_hashed(self, input_sha256: str, encoder_fingerprint: str, embedding: torch.Tensor) -> RepresentationRecord:
        """Store an entry using a caller-supplied digest for structured/multimodal inputs."""
        key = self.make_key(input_sha256, encoder_fingerprint, self.namespace)
        target = self.data_dir / f"{key}.pt"
        tmp = target.with_suffix(".tmp")
        torch.save(embedding.detach().cpu(), tmp)
        tmp.replace(target)
        record = RepresentationRecord(
            key=key, input_sha256=input_sha256, encoder_fingerprint=encoder_fingerprint,
            shape=tuple(int(v) for v in embedding.shape), dtype=str(embedding.dtype),
        )
        meta_target = self.meta_dir / f"{key}.json"
        meta_tmp = meta_target.with_suffix(".tmp")
        meta_tmp.write_text(json.dumps(record.__dict__, indent=2, sort_keys=True), encoding="utf-8")
        meta_tmp.replace(meta_target)
        return record

    def get_by_key(self, key: str) -> torch.Tensor | None:
        """Load an entry by a previously validated content-addressed key."""
        path = self.data_dir / f"{key}.pt"
        meta = self.meta_dir / f"{key}.json"
        if not path.exists() or not meta.exists():
            return None
        record = json.loads(meta.read_text(encoding="utf-8"))
        if record.get("key") != key:
            return None
        return torch.load(path, map_location="cpu", weights_only=True)

    def stats(self) -> dict[str, int]:
        return {
            "entries": len(list(self.data_dir.glob("*.pt"))),
            "metadata": len(list(self.meta_dir.glob("*.json"))),
        }


class FrozenBackboneProjector(nn.Module):
    """Frozen feature extractor plus a trainable projection into Aether latent space."""

    def __init__(self, backbone: nn.Module, feature_dim: int, latent_dim: int, normalize: bool = True):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = int(feature_dim)
        self.output_dim = int(latent_dim)
        self.normalize = bool(normalize)
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
        )
        self.freeze_backbone()

    def freeze_backbone(self) -> None:
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

    def _features(self, x: Any) -> torch.Tensor:
        with torch.no_grad():
            if hasattr(self.backbone, "encode"):
                z = self.backbone.encode(x)
            else:
                z = self.backbone(x)
        if not torch.is_tensor(z):
            raise TypeError("pretrained backbone must return a torch.Tensor")
        if z.ndim > 2:
            z = z.flatten(1)
        if z.shape[-1] != self.feature_dim:
            raise ValueError(f"backbone returned feature dim {z.shape[-1]}, expected {self.feature_dim}")
        return z.float()

    def encode_features(self, x: Any) -> torch.Tensor:
        z = self._features(x)
        if self.normalize:
            z = torch.nn.functional.normalize(z, dim=-1)
        return z

    def forward(self, x: Any) -> torch.Tensor:
        return self.projection(self.encode_features(x))


class CachedRepresentation(nn.Module):
    """Cache frozen backbone features while keeping the projection trainable.

    Caching occurs before the projection. This is important: changing projection
    weights does not invalidate expensive pretrained features.
    """

    def __init__(self, encoder: FrozenBackboneProjector, cache: RepresentationCache, fingerprint: str | None = None):
        super().__init__()
        self.encoder = encoder
        self.cache = cache
        self.encoder_fingerprint = fingerprint or module_fingerprint(encoder.backbone)
        self.output_dim = encoder.output_dim

    def frozen_features(self, x: torch.Tensor) -> torch.Tensor:
        rows = []
        batch = x if x.ndim > 1 else x.unsqueeze(0)
        for sample in batch:
            sample_batch = sample.unsqueeze(0)
            hit = self.cache.get(sample_batch, self.encoder_fingerprint)
            if hit is None:
                hit = self.encoder.encode_features(sample_batch).detach().cpu()
                self.cache.put(sample_batch, self.encoder_fingerprint, hit)
            rows.append(hit)
        return torch.cat(rows, dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = next(self.encoder.projection.parameters()).device
        features = self.frozen_features(x).to(device)
        return self.encoder.projection(features)


class CallableBackbone(nn.Module):
    """Small adapter useful for local pretrained models with a tensor callable."""

    def __init__(self, fn: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x)




class IdentityBackbone(nn.Module):
    """Deterministic state-vector feature encoder used for provenance-safe benchmarks.

    It contains no trainable parameters and returns flattened float features. Its
    module fingerprint is therefore stable across runs and can be stored in world
    checkpoints just like a frozen vision backbone.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.output_dim = int(feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.as_tensor(x, dtype=torch.float32)
        if y.ndim == 1:
            y = y.unsqueeze(0)
        y = y.reshape(y.shape[0], -1)
        if y.shape[-1] != self.output_dim:
            raise ValueError(f"identity backbone got dim {y.shape[-1]}, expected {self.output_dim}")
        return y


class HuggingFaceFrozenBackbone(nn.Module):
    """Optional Transformers vision backbone exposing unprojected frozen features.

    `model_name_or_path` can point at a fully local model. Network downloading is
    disabled by default so experiment provenance is explicit and reproducible.
    """

    def __init__(self, model_name_or_path: str, local_files_only: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise ImportError("Install with `pip install -e '.[hf]'` for Hugging Face backbones") from exc
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            hidden = getattr(self.model.config, "projection_dim", None)
        if hidden is None:
            raise ValueError("could not infer feature dimension from Hugging Face model config")
        self.output_dim = int(hidden)
        self.model_name_or_path = str(model_name_or_path)
        self.local_files_only = bool(local_files_only)

    def forward(self, images: Any) -> torch.Tensor:
        batch = self.processor(images=images, return_tensors="pt")
        device = next(self.model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            out = self.model(**batch)
        if getattr(out, "pooler_output", None) is not None:
            return out.pooler_output.float()
        hidden = out.last_hidden_state
        return hidden.mean(dim=1).float()


def build_pretrained_representation(
    provider: str,
    latent_dim: int,
    *,
    model_name_or_path: str | None = None,
    feature_dim: int | None = None,
    backbone: nn.Module | None = None,
    local_files_only: bool = True,
) -> FrozenBackboneProjector:
    """Factory for v2.1 representation backends.

    `provider='module'` accepts a supplied local nn.Module and explicit feature_dim.
    `provider='hf'` uses a local Hugging Face model by default. DINO-family and
    similar AutoModel-compatible encoders are therefore supported without Aether
    hard-coding a particular checkpoint.
    """
    key = provider.lower().strip()
    if key == "identity":
        if feature_dim is None:
            raise ValueError("identity provider requires feature_dim")
        return FrozenBackboneProjector(IdentityBackbone(feature_dim), feature_dim, latent_dim, normalize=False)
    if key == "module":
        if backbone is None or feature_dim is None:
            raise ValueError("module provider requires backbone and feature_dim")
        return FrozenBackboneProjector(backbone, feature_dim, latent_dim)
    if key in {"hf", "huggingface", "dino"}:
        if not model_name_or_path:
            raise ValueError("Hugging Face provider requires model_name_or_path")
        hf = HuggingFaceFrozenBackbone(model_name_or_path, local_files_only=local_files_only)
        return FrozenBackboneProjector(hf, hf.output_dim, latent_dim)
    if key in {"vjepa2", "vjepa2-hf", "hf-vjepa2"}:
        if not model_name_or_path:
            model_name_or_path = "facebook/vjepa2-vitl-fpc64-256"
        from awa.v2.video_representation import build_vjepa_game_representation
        return build_vjepa_game_representation(
            "vjepa2-hf", latent_dim, model_name_or_path=model_name_or_path,
            local_files_only=local_files_only, backbone=backbone,
        )
    raise ValueError(f"unknown representation provider: {provider}")
