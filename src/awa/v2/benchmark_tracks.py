from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any
import hashlib
import json


class BenchmarkTrack(str, Enum):
    STRUCTURED = "structured"
    PIXEL = "pixel"


@dataclass(frozen=True)
class RepresentationContract:
    track: BenchmarkTrack
    observation_dim: int | None = None
    telemetry_dim: int = 0
    encoder_name: str | None = None
    encoder_fingerprint: str | None = None
    frame_shape: tuple[int, int, int] | None = None
    clip_length: int | None = None

    def __post_init__(self):
        if self.track == BenchmarkTrack.STRUCTURED:
            if self.observation_dim is None or int(self.observation_dim) <= 0:
                raise ValueError("structured track requires observation_dim")
            if self.encoder_fingerprint is not None or self.frame_shape is not None:
                raise ValueError("structured track must not claim a pixel encoder")
        else:
            if not self.encoder_name or not self.encoder_fingerprint:
                raise ValueError("pixel track requires encoder name and fingerprint")
            if self.frame_shape is None or len(self.frame_shape) != 3 or min(self.frame_shape) <= 0:
                raise ValueError("pixel track requires positive HWC frame_shape")
            if self.clip_length is None or int(self.clip_length) <= 0:
                raise ValueError("pixel track requires clip_length")
        if int(self.telemetry_dim) < 0:
            raise ValueError("telemetry_dim cannot be negative")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["track"] = self.track.value
        if self.frame_shape is not None:
            data["frame_shape"] = list(self.frame_shape)
        return data

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RepresentationContract":
        data = dict(raw)
        data["track"] = BenchmarkTrack(data["track"])
        if data.get("frame_shape") is not None:
            data["frame_shape"] = tuple(int(x) for x in data["frame_shape"])
        return cls(**data)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class BenchmarkManifest:
    name: str
    representation: RepresentationContract
    dataset_sha256: str
    task_suite: str
    seed_set: tuple[int, ...]

    def __post_init__(self):
        if not self.name or not self.dataset_sha256 or not self.task_suite:
            raise ValueError("benchmark manifest identity fields cannot be empty")
        if len(self.dataset_sha256) != 64:
            raise ValueError("dataset_sha256 must be a SHA-256 hex digest")
        if not self.seed_set:
            raise ValueError("seed_set cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "awa-v2.14-benchmark-manifest-v1",
            "name": self.name,
            "representation": self.representation.to_dict(),
            "representation_fingerprint": self.representation.fingerprint,
            "dataset_sha256": self.dataset_sha256,
            "task_suite": self.task_suite,
            "seed_set": list(self.seed_set),
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path


def assert_representation_compatible(a: RepresentationContract, b: RepresentationContract) -> None:
    if a.fingerprint != b.fingerprint:
        raise ValueError(
            f"representation contract mismatch: {a.track.value}:{a.fingerprint[:12]} != "
            f"{b.track.value}:{b.fingerprint[:12]}"
        )
