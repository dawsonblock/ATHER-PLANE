from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

from awa.v2.benchmark_tracks import BenchmarkTrack, RepresentationContract
from awa.v2.video_representation import VideoClipSpec, video_backbone_fingerprint
from .vizdoom_env import VIZDOOM_TELEMETRY_DIM


DEFAULT_VJEPA2_MODEL = "facebook/vjepa2-vitl-fpc64-256"


@dataclass(frozen=True)
class ViZDoomVJEPAConfig:
    model_name_or_path: str = DEFAULT_VJEPA2_MODEL
    frames_per_clip: int = 64
    frame_stride: int = 1
    image_size: int = 256
    telemetry_dim: int = VIZDOOM_TELEMETRY_DIM
    normalize_visual: bool = True

    def clip_spec(self) -> VideoClipSpec:
        return VideoClipSpec(self.frames_per_clip,self.frame_stride)

    def to_dict(self): return asdict(self)


def validate_vjepa_clip_contract(backbone: Any, spec: VideoClipSpec) -> None:
    required = getattr(backbone, "frames_per_clip", None)
    if required is not None and int(required) != int(spec.frames_per_clip):
        raise ValueError(
            f"V-JEPA checkpoint expects {int(required)} frames per clip, got {int(spec.frames_per_clip)}"
        )


def vizdoom_pixel_representation_contract(backbone: Any, *, frame_shape: tuple[int,int,int], clip_spec: VideoClipSpec, telemetry_dim: int = VIZDOOM_TELEMETRY_DIM) -> RepresentationContract:
    validate_vjepa_clip_contract(backbone,clip_spec)
    return RepresentationContract(
        BenchmarkTrack.PIXEL,
        telemetry_dim=int(telemetry_dim),
        encoder_name=getattr(backbone,"model_name_or_path",backbone.__class__.__name__),
        encoder_fingerprint=video_backbone_fingerprint(backbone),
        frame_shape=tuple(int(x) for x in frame_shape),
        clip_length=int(clip_spec.frames_per_clip),
    )


def write_vizdoom_vjepa_manifest(path: str | Path, *, backbone: Any, frame_shape: tuple[int,int,int], clip_spec: VideoClipSpec, scenario: str, dataset_sha256: str, telemetry_dim: int = VIZDOOM_TELEMETRY_DIM) -> Path:
    contract=vizdoom_pixel_representation_contract(backbone,frame_shape=frame_shape,clip_spec=clip_spec,telemetry_dim=telemetry_dim)
    payload={
        "format":"awa-v2.15-vizdoom-vjepa-manifest-v1",
        "scenario":str(scenario),"dataset_sha256":str(dataset_sha256),
        "representation":contract.to_dict(),"representation_fingerprint":contract.fingerprint,
        "clip_spec":clip_spec.to_dict(),
    }
    p=Path(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n",encoding="utf-8"); return p
