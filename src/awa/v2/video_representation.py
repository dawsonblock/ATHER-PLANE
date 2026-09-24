from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable
import hashlib
import json

import numpy as np
import torch
from torch import nn

from awa.v2.representation import RepresentationCache, FrozenBackboneProjector


def _json_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _to_chw(frame: Any) -> torch.Tensor:
    """Convert one HWC/CHW game frame to contiguous CHW without normalizing pixels."""
    x = torch.as_tensor(frame)
    if x.ndim != 3:
        raise ValueError(f"game frame must have 3 dimensions, got {tuple(x.shape)}")
    if x.shape[0] in (1, 3, 4):
        y = x
    elif x.shape[-1] in (1, 3, 4):
        y = x.permute(2, 0, 1)
    else:
        raise ValueError("cannot infer image channel axis; expected CHW or HWC with 1/3/4 channels")
    if y.shape[0] == 4:  # drop alpha; V-JEPA is RGB
        y = y[:3]
    if y.shape[0] == 1:
        y = y.repeat(3, 1, 1)
    return y.contiguous()


@dataclass(frozen=True)
class VideoClipSpec:
    frames_per_clip: int = 64
    frame_stride: int = 1
    pad_mode: str = "repeat_first"

    def __post_init__(self):
        if int(self.frames_per_clip) <= 0:
            raise ValueError("frames_per_clip must be positive")
        if int(self.frame_stride) <= 0:
            raise ValueError("frame_stride must be positive")
        if self.pad_mode != "repeat_first":
            raise ValueError("only repeat_first padding is currently supported")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EpisodeClipBuilder:
    """Construct causal game-video clips without crossing episode boundaries.

    Current clips end at observations[i]. Next clips end at next_observations[i].
    Early-episode history is left-padded with the first frame of that episode.
    """

    def __init__(self, observations: Any, next_observations: Any, dones: Any, spec: VideoClipSpec):
        self.observations = observations
        self.next_observations = next_observations
        self.dones = np.asarray(dones, dtype=np.bool_).reshape(-1)
        self.spec = spec
        n = int(len(self.dones))
        if len(observations) != n or len(next_observations) != n:
            raise ValueError("video transition arrays must have equal lengths")
        starts = np.zeros(n, dtype=np.int64)
        start = 0
        for i in range(n):
            starts[i] = start
            if self.dones[i]:
                start = i + 1
        self.episode_starts = starts

    def _frame_for_virtual_index(self, transition_index: int, virtual_index: int, next_endpoint: bool) -> torch.Tensor:
        start = int(self.episode_starts[transition_index])
        if virtual_index <= start:
            return _to_chw(self.observations[start])
        if next_endpoint and virtual_index == transition_index + 1:
            return _to_chw(self.next_observations[transition_index])
        if virtual_index > transition_index:
            # This can only happen for the single next endpoint above.
            return _to_chw(self.next_observations[transition_index])
        return _to_chw(self.observations[virtual_index])

    def clip(self, transition_index: int, *, next_clip: bool = False) -> torch.Tensor:
        i = int(transition_index)
        if i < 0 or i >= len(self.dones):
            raise IndexError(i)
        end = i + (1 if next_clip else 0)
        count = int(self.spec.frames_per_clip)
        stride = int(self.spec.frame_stride)
        virtual = [end - (count - 1 - k) * stride for k in range(count)]
        frames = [self._frame_for_virtual_index(i, j, next_clip) for j in virtual]
        return torch.stack(frames, dim=0)  # T,C,H,W

    def transition_clips(self, transition_index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.clip(transition_index, next_clip=False), self.clip(transition_index, next_clip=True)


class VJEPA2HFBackbone(nn.Module):
    """Frozen V-JEPA 2 feature extractor using Hugging Face Transformers.

    The production path uses AutoVideoProcessor + AutoModel. For tests/offline
    environments, callers may inject a compatible model and processor and avoid
    installing Transformers entirely.

    Inputs are B,T,C,H,W game clips. Each clip is processed independently because
    representation caching is content-addressed per transition and this keeps the
    interface compatible across Transformers versions.
    """

    def __init__(
        self,
        model_name_or_path: str = "facebook/vjepa2-vitl-fpc64-256",
        *,
        local_files_only: bool = True,
        revision: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        pooling: str = "mean_tokens",
        model: nn.Module | None = None,
        processor: Callable[..., Any] | None = None,
    ):
        super().__init__()
        if pooling not in {"mean_tokens", "mean_spatiotemporal"}:
            raise ValueError(f"unsupported V-JEPA pooling: {pooling}")
        injected_components = model is not None or processor is not None
        if model is None or processor is None:
            try:
                from transformers import AutoModel, AutoVideoProcessor
            except ImportError as exc:
                raise ImportError(
                    "Install `aether-world-agent[vjepa]` for Hugging Face V-JEPA 2 support"
                ) from exc
            model = AutoModel.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                revision=revision,
            )
            processor = AutoVideoProcessor.from_pretrained(
                model_name_or_path,
                local_files_only=local_files_only,
                revision=revision,
            )
        self.model = model
        self.processor = processor
        self._awa_injected_components = bool(injected_components)
        self.model_name_or_path = str(model_name_or_path)
        self.local_files_only = bool(local_files_only)
        self.revision = revision
        self.pooling = pooling
        if device is not None:
            self.model.to(device)
        if dtype is not None:
            self.model.to(dtype=dtype)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        config = getattr(self.model, "config", SimpleNamespace())
        hidden = getattr(config, "hidden_size", None) or getattr(self.model, "embed_dim", None)
        if hidden is None:
            raise ValueError("could not infer V-JEPA feature dimension from model config")
        self.output_dim = int(hidden)
        self.frames_per_clip = int(getattr(config, "frames_per_clip", 64))
        self._commit_hash = getattr(config, "_commit_hash", None)
        self._awa_fingerprint = _json_sha256({
            "format": "awa-vjepa2-hf-v1",
            "model": self.model_name_or_path,
            "revision": self.revision,
            "commit_hash": self._commit_hash,
            "hidden_size": self.output_dim,
            "frames_per_clip": self.frames_per_clip,
            "pooling": self.pooling,
        })

    def fingerprint(self) -> str:
        return self._awa_fingerprint

    @staticmethod
    def _move_batch(batch: Any, device: torch.device) -> dict[str, torch.Tensor]:
        if hasattr(batch, "to"):
            batch = batch.to(device)
        if isinstance(batch, dict):
            return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        # Transformers BatchFeature behaves like a mapping.
        return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _one(self, clip: torch.Tensor) -> torch.Tensor:
        clip = torch.as_tensor(clip).detach().cpu()
        try:
            batch = self.processor(clip, return_tensors="pt")
        except TypeError:
            batch = self.processor(videos=clip, return_tensors="pt")
        device = next(self.model.parameters(), torch.empty(0)).device
        batch = self._move_batch(batch, device)
        with torch.no_grad():
            get_features = getattr(self.model, "get_vision_features", None)
            out = get_features(**batch) if callable(get_features) else self.model(**batch)
        hidden = out if torch.is_tensor(out) else getattr(out, "last_hidden_state", None)
        if hidden is None:
            if torch.is_tensor(out):
                hidden = out
            elif isinstance(out, (tuple, list)) and out and torch.is_tensor(out[0]):
                hidden = out[0]
            else:
                raise TypeError("V-JEPA model output has no last_hidden_state tensor")
        if hidden.ndim == 2:
            pooled = hidden
        else:
            pooled = hidden.reshape(hidden.shape[0], -1, hidden.shape[-1]).mean(dim=1)
        return pooled.float()

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(clips)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"V-JEPA clips must be B,T,C,H,W, got {tuple(x.shape)}")
        if x.shape[0] == 1:
            return self._one(x[0])
        # Prefer one processor/model invocation for real V-JEPA throughput.  Older
        # processors and lightweight test doubles are allowed to fall back to the
        # exact per-clip path so compatibility remains fail-safe.
        try:
            cpu=x.detach().cpu()
            try:
                batch=self.processor(videos=cpu,return_tensors="pt")
            except Exception:
                try:
                    batch=self.processor(cpu,return_tensors="pt")
                except Exception:
                    batch=self.processor(videos=[clip for clip in cpu],return_tensors="pt")
            device=next(self.model.parameters(),torch.empty(0)).device
            batch=self._move_batch(batch,device)
            with torch.no_grad():
                get_features=getattr(self.model,"get_vision_features",None)
                out=get_features(**batch) if callable(get_features) else self.model(**batch)
            hidden=out if torch.is_tensor(out) else getattr(out,"last_hidden_state",None)
            if hidden is None and isinstance(out,(tuple,list)) and out and torch.is_tensor(out[0]): hidden=out[0]
            if hidden is not None:
                pooled=hidden if hidden.ndim==2 else hidden.reshape(hidden.shape[0],-1,hidden.shape[-1]).mean(dim=1)
                if pooled.shape[0] == x.shape[0]:
                    return pooled.float()
        except Exception:
            pass
        rows=[self._one(clip) for clip in x]
        return torch.cat(rows,dim=0)


class VJEPA21TorchHubBackbone(nn.Module):
    """Local-first adapter for Meta's V-JEPA 2.1 PyTorch-Hub backbones.

    By default this requires a local checkout of facebookresearch/vjepa2 and uses
    source='local'. `allow_download=True` enables torch.hub's GitHub path explicitly.
    No video decoder is required because Aether supplies in-memory game-frame clips.
    """

    def __init__(
        self,
        repo_or_dir: str,
        *,
        model_name: str = "vjepa2_1_vit_base_384",
        preprocessor_name: str = "vjepa2_preprocessor",
        feature_dim: int | None = None,
        allow_download: bool = False,
        device: str | torch.device | None = None,
        model: nn.Module | None = None,
        preprocessor: Callable[[Any], Any] | None = None,
    ):
        super().__init__()
        source = "github" if allow_download else "local"
        if model is None:
            model = torch.hub.load(repo_or_dir, model_name, source=source)
        if preprocessor is None:
            preprocessor = torch.hub.load(repo_or_dir, preprocessor_name, source=source)
        self.model = model
        self.preprocessor = preprocessor
        self.repo_or_dir = str(repo_or_dir)
        self.model_name = str(model_name)
        self.allow_download = bool(allow_download)
        if device is not None:
            self.model.to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        inferred = feature_dim or getattr(self.model, "embed_dim", None) or getattr(self.model, "hidden_size", None)
        if inferred is None:
            raise ValueError("V-JEPA 2.1 TorchHub adapter requires feature_dim when the model does not expose embed_dim")
        self.output_dim = int(inferred)
        self._awa_fingerprint = _json_sha256({
            "format": "awa-vjepa21-torchhub-v1",
            "repo_or_dir": self.repo_or_dir,
            "model_name": self.model_name,
            "feature_dim": self.output_dim,
            "source": source,
        })

    def fingerprint(self) -> str:
        return self._awa_fingerprint

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(clips)
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError(f"V-JEPA 2.1 clips must be B,T,C,H,W, got {tuple(x.shape)}")
        rows = []
        device = next(self.model.parameters(), torch.empty(0)).device
        for clip in x:
            # Meta's hub preprocessor currently returns a list of eval views; the
            # first view is C,T,H,W. Some custom/local preprocessors accept T,C,H,W
            # directly, while others expect T,H,W,C, so support both deterministically.
            try:
                prepared = self.preprocessor(clip)
            except (TypeError, ValueError, RuntimeError):
                prepared = self.preprocessor(clip.permute(0, 2, 3, 1).cpu().numpy())
            if isinstance(prepared, (list, tuple)) and prepared and torch.is_tensor(prepared[0]):
                prepared = prepared[0]
            if torch.is_tensor(prepared):
                prepared = prepared.unsqueeze(0) if prepared.ndim == 4 else prepared
                prepared = prepared.to(device)
                args, kwargs = (prepared,), {}
            elif isinstance(prepared, dict):
                args, kwargs = (), {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in prepared.items()}
            else:
                raise TypeError("V-JEPA 2.1 preprocessor must return tensor, tensor view-list or mapping")
            with torch.no_grad():
                out = self.model(*args, **kwargs)
            if isinstance(out, (tuple, list)):
                out = out[0]
            if not torch.is_tensor(out):
                raise TypeError("V-JEPA 2.1 backbone must return a tensor or tensor tuple")
            if out.ndim == 2:
                pooled = out
            else:
                pooled = out.reshape(out.shape[0], -1, out.shape[-1]).mean(dim=1)
            rows.append(pooled.float())
        return torch.cat(rows, dim=0)


class ToyVideoBackbone(nn.Module):
    """Dependency-free deterministic video feature extractor used only for tests/smokes."""

    def __init__(self, feature_dim: int = 16):
        super().__init__()
        self.output_dim = int(feature_dim)
        self.proj = nn.Linear(6, self.output_dim, bias=False)
        g = torch.Generator().manual_seed(250)
        with torch.no_grad():
            self.proj.weight.copy_(torch.randn(self.output_dim, 6, generator=g) * 0.1)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        x = torch.as_tensor(clips).float()
        if x.ndim == 4:
            x = x.unsqueeze(0)
        if x.ndim != 5:
            raise ValueError("toy video backbone expects B,T,C,H,W")
        if x.max() > 1.5:
            x = x / 255.0
        # Per-channel mean/std across time and space gives six deterministic statistics.
        mean = x.mean(dim=(1, 3, 4))
        std = x.std(dim=(1, 3, 4), unbiased=False)
        stats = torch.cat([mean, std], dim=-1)
        if stats.shape[-1] != 6:
            raise ValueError("toy video backbone expects RGB clips")
        return self.proj(stats)


class StreamingGameFeatureEncoder:
    """Online causal game-frame encoder for deployment with a saved world checkpoint.

    It returns *frozen features*, not Aether latents. Feed the result to
    LoadedWorldRuntime.project_features so the exact projector trained with the world
    checkpoint is used at deployment.
    """

    def __init__(self, backbone: nn.Module, spec: VideoClipSpec, telemetry_dim: int = 0, normalize_visual: bool = True):
        self.backbone = backbone
        self.spec = spec
        self.telemetry_dim = int(telemetry_dim)
        self.normalize_visual = bool(normalize_visual)
        self.frames: deque[torch.Tensor] = deque(maxlen=(spec.frames_per_clip - 1) * spec.frame_stride + 1)

    def reset(self) -> None:
        self.frames.clear()

    def _sample(self) -> torch.Tensor:
        if not self.frames:
            raise RuntimeError("no frame has been appended")
        frames = list(self.frames)
        end = len(frames) - 1
        idx = [end - (self.spec.frames_per_clip - 1 - k) * self.spec.frame_stride for k in range(self.spec.frames_per_clip)]
        idx = [max(0, i) for i in idx]
        return torch.stack([frames[i] for i in idx], dim=0)

    @torch.no_grad()
    def step(self, frame: Any, telemetry: Any | None = None) -> torch.Tensor:
        self.frames.append(_to_chw(frame))
        clip = self._sample().unsqueeze(0)
        features = self.backbone(clip).float()
        if self.normalize_visual:
            features = torch.nn.functional.normalize(features, dim=-1)
        if self.telemetry_dim:
            if telemetry is None:
                raise ValueError("telemetry is required for this game encoder")
            t = torch.as_tensor(telemetry, dtype=torch.float32, device=features.device).reshape(1, -1)
            if t.shape[-1] != self.telemetry_dim:
                raise ValueError(f"telemetry dim {t.shape[-1]} != expected {self.telemetry_dim}")
            features = torch.cat([features, t], dim=-1)
        elif telemetry is not None:
            raise ValueError("telemetry was supplied but telemetry_dim=0")
        return features


def video_backbone_fingerprint(backbone: nn.Module) -> str:
    custom = getattr(backbone, "fingerprint", None)
    if callable(custom):
        return str(custom())
    from awa.v2.representation import module_fingerprint
    return module_fingerprint(backbone)


def _multi_input_sha256(clip: torch.Tensor, telemetry: torch.Tensor | None) -> str:
    h = hashlib.sha256()
    for name, value in (("clip", clip), ("telemetry", telemetry)):
        h.update(name.encode())
        if value is None:
            h.update(b"none")
            continue
        y = torch.as_tensor(value).detach().cpu().contiguous()
        h.update(f"{y.dtype}|{tuple(y.shape)}|".encode())
        h.update(y.numpy().tobytes())
    return h.hexdigest()


def precompute_game_video_cache(
    dataset: Any,
    backbone: nn.Module,
    cache: RepresentationCache,
    *,
    clip_spec: VideoClipSpec,
    index_path: str | Path | None = None,
    normalize_visual: bool = True,
    batch_size: int = 8,
) -> dict[str, Any]:
    """Cache causal pretrained video features (+ telemetry) for every transition.

    v2.16 batches cache misses through the frozen video backbone. Content hashes are
    still computed per transition, so batching changes throughput but not cache ABI.
    """
    if int(batch_size) <= 0: raise ValueError("batch_size must be positive")
    arrays=dataset.arrays
    builder=EpisodeClipBuilder(arrays["observations"],arrays["next_observations"],arrays["dones"],clip_spec)
    has_telemetry="telemetry" in arrays
    telemetry_dim=int(np.prod(arrays["telemetry"].shape[1:])) if has_telemetry else 0
    fp=video_backbone_fingerprint(backbone)
    observation_keys=[]; next_observation_keys=[]; hits=0; writes=0; peak_pending_clips=0
    pending: dict[str, tuple[str,torch.Tensor,torch.Tensor|None]] = {}

    def flush_pending():
        """Encode and persist a bounded chunk so long campaigns never retain all clips in RAM."""
        nonlocal writes
        if not pending:
            return
        items=list(pending.items())
        pending.clear()
        with torch.no_grad():
            for start_i in range(0,len(items),int(batch_size)):
                chunk=items[start_i:start_i+int(batch_size)]
                clips=torch.stack([row[1][1] for row in chunk],dim=0)
                feats=backbone(clips).float()
                if feats.shape[0] != len(chunk):
                    raise RuntimeError("video backbone returned unexpected batch size")
                if normalize_visual: feats=torch.nn.functional.normalize(feats,dim=-1)
                for j,(key,(input_sha,_clip,tel)) in enumerate(chunk):
                    feat=feats[j:j+1]
                    if tel is not None:
                        feat=torch.cat([feat,tel.float().reshape(1,-1).to(feat.device)],dim=-1)
                    cache.put_hashed(input_sha,fp,feat); writes+=1

    def register(clip: torch.Tensor, telemetry: torch.Tensor|None):
        nonlocal hits, peak_pending_clips
        input_sha=_multi_input_sha256(clip,telemetry)
        key=cache.make_key(input_sha,fp,cache.namespace)
        if cache.get_hashed(input_sha,fp) is not None:
            hits+=1
        elif key not in pending:
            pending[key]=(input_sha,clip,telemetry)
            peak_pending_clips=max(peak_pending_clips,len(pending))
            if len(pending)>=int(batch_size):
                flush_pending()
        return key

    for i in range(len(dataset)):
        current,nxt=builder.transition_clips(i)
        tel=torch.as_tensor(arrays["telemetry"][i]).float() if has_telemetry else None
        ntel=torch.as_tensor(arrays["next_telemetry"][i]).float() if has_telemetry else None
        observation_keys.append(register(current,tel)); next_observation_keys.append(register(nxt,ntel))

    flush_pending()

    first=cache.get_by_key(observation_keys[0])
    if first is None: raise RuntimeError("failed to read back first game representation cache entry")
    index={
        "format":"awa-game-video-representation-index-v1",
        "dataset_sha256":dataset.manifest.sha256,"encoder_fingerprint":fp,
        "encoder_class":f"{backbone.__class__.__module__}.{backbone.__class__.__qualname__}",
        "transitions":len(dataset),"feature_dim":int(first.reshape(1,-1).shape[-1]),
        "visual_feature_dim":int(first.reshape(1,-1).shape[-1]-telemetry_dim),"telemetry_dim":telemetry_dim,
        "clip_spec":clip_spec.to_dict(),"batch_size":int(batch_size),
        "observation_keys":observation_keys,"next_observation_keys":next_observation_keys,
        "hits":hits,"writes":writes,"peak_pending_clips":peak_pending_clips,
    }
    if index_path is not None:
        path=Path(index_path); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(index,indent=2,sort_keys=True),encoding="utf-8")
    return index


def build_vjepa_game_representation(
    provider: str,
    latent_dim: int,
    *,
    model_name_or_path: str = "facebook/vjepa2-vitl-fpc64-256",
    repo_or_dir: str | None = None,
    model_name: str = "vjepa2_1_vit_base_384",
    feature_dim: int | None = None,
    local_files_only: bool = True,
    device: str | torch.device | None = None,
    backbone: nn.Module | None = None,
    processor: Callable[..., Any] | None = None,
) -> FrozenBackboneProjector:
    key = provider.lower().strip()
    if key in {"vjepa2", "vjepa2-hf", "hf-vjepa2"}:
        bb = VJEPA2HFBackbone(
            model_name_or_path,
            local_files_only=local_files_only,
            device=device,
            model=backbone,
            processor=processor,
        )
    elif key in {"vjepa21", "vjepa2.1", "vjepa21-torchhub"}:
        if backbone is None and repo_or_dir is None:
            raise ValueError("V-JEPA 2.1 TorchHub provider requires --repo-or-dir or an injected backbone")
        bb = VJEPA21TorchHubBackbone(
            repo_or_dir or ".",
            model_name=model_name,
            feature_dim=feature_dim,
            allow_download=not local_files_only,
            device=device,
            model=backbone,
            preprocessor=processor,
        )
    elif key in {"toy-video", "toy"}:
        bb = ToyVideoBackbone(feature_dim=feature_dim or 16)
    else:
        raise ValueError(f"unknown game video provider: {provider}")
    return FrozenBackboneProjector(bb, int(bb.output_dim), int(latent_dim), normalize=True)
