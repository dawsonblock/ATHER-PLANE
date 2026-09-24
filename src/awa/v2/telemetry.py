from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Lock
from typing import Any
import json
import math
import os
import time

import torch


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("telemetry floats must be finite")
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"telemetry value is not JSON-safe: {type(value).__name__}")


@dataclass(frozen=True)
class TelemetrySummary:
    events: int
    event_counts: dict[str, int]
    latest_metrics: dict[str, float]
    numeric_means: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TelemetryRecorder:
    """Append-only JSONL telemetry suitable for long-running training campaigns.

    The recorder deliberately accepts generic numeric metrics rather than importing
    the trainer. This keeps instrumentation out of the learning trust path.
    """

    def __init__(self, path: str | Path, *, run_id: str = "aether"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        self._lock = Lock()

    def event(
        self,
        event_type: str,
        *,
        step: int | None = None,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event_type = str(event_type).strip()
        if not event_type:
            raise ValueError("event_type cannot be empty")
        row = {
            "format": "awa-v2.14-telemetry-event-v1",
            "time_unix": float(time.time()),
            "run_id": self.run_id,
            "event": event_type,
            "pid": int(os.getpid()),
            "step": None if step is None else int(step),
            "metrics": _json_safe(metrics or {}),
            "metadata": _json_safe(metadata or {}),
        }
        payload = json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
        return row

    def system_event(self, *, step: int | None = None) -> dict[str, Any]:
        metrics: dict[str, Any] = {}
        metadata: dict[str, Any] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
        }
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            metadata["cuda_device"] = str(props.name)
            metrics.update(
                cuda_allocated_bytes=float(torch.cuda.memory_allocated(device)),
                cuda_reserved_bytes=float(torch.cuda.memory_reserved(device)),
                cuda_max_allocated_bytes=float(torch.cuda.max_memory_allocated(device)),
            )
        return self.event("system", step=step, metrics=metrics, metadata=metadata)

    def rows(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows: list[dict[str, Any]] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
        return rows

    def summarize(self) -> TelemetrySummary:
        counts: dict[str, int] = {}
        latest: dict[str, float] = {}
        accum: dict[str, list[float]] = {}
        rows = self.rows()
        for row in rows:
            event = str(row.get("event", "unknown"))
            counts[event] = counts.get(event, 0) + 1
            for key, value in (row.get("metrics") or {}).items():
                if isinstance(value, (int, float)) and math.isfinite(float(value)):
                    latest[key] = float(value)
                    accum.setdefault(key, []).append(float(value))
        means = {k: float(sum(v) / len(v)) for k, v in accum.items() if v}
        return TelemetrySummary(len(rows), counts, latest, means)
