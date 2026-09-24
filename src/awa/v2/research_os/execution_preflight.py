"""Fail-closed hardware/dependency preflight for real Aether experiments.

The preflight is deliberately evidence-only.  It never installs packages,
downloads models, mutates configs, or relaxes requested requirements.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib.metadata
import math
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
from typing import Any

import torch


PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


@dataclass(frozen=True)
class PreflightRequirement:
    require_cuda: bool = True
    min_cuda_vram_gib: float = 16.0
    min_disk_free_gib: float = 20.0
    min_system_ram_gib: float = 0.0
    require_vizdoom: bool = True
    require_transformers: bool = False
    require_vjepa_cache: bool = False
    vjepa_model: str = "facebook/vjepa2-vitl-fpc64-256"
    persistent_volume_path: str = "/workspace"
    persistent_volume_size_env: str = "AWA_RUNPOD_VOLUME_SIZE_GB"
    persistent_volume_id_env: str = "AWA_RUNPOD_VOLUME_ID"
    require_persistent_volume_quota: bool = False

    def __post_init__(self) -> None:
        if self.min_cuda_vram_gib < 0 or self.min_disk_free_gib < 0 or self.min_system_ram_gib < 0:
            raise ValueError("minimum resource requirements must be non-negative")


@dataclass(frozen=True)
class PreflightCheck:
    name: str
    status: str
    required: bool
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _vizdoom_ipc_probe() -> tuple[bool, str | None]:
    # DoomGame can abort the process when its local IPC socket is denied.
    # Check the capability before calling the native extension.
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
            pass
        return True, None
    except OSError as exc:
        return False, str(exc)


def _gib(value: int | float) -> float:
    return float(value) / float(1024**3)


def _system_ram_bytes() -> int | None:
    try:
        total = int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, TypeError, AttributeError):
        return None
    # Containers can see host RAM through sysconf; honor their cgroup limit too.
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            raw = path.read_text(encoding="ascii").strip()
            if raw != "max":
                limit = int(raw)
                if 0 < limit < (1 << 60):
                    total = min(total, limit)
        except (OSError, ValueError):
            pass
    return total


def _hf_cache_roots() -> list[Path]:
    roots: list[Path] = []
    if os.getenv("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    if os.getenv("HUGGINGFACE_HUB_CACHE"):
        roots.append(Path(os.environ["HUGGINGFACE_HUB_CACHE"]))
    if os.getenv("TRANSFORMERS_CACHE"):
        roots.append(Path(os.environ["TRANSFORMERS_CACHE"]))
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    # Preserve order while deduplicating resolved-ish strings without requiring existence.
    seen: set[str] = set()
    out: list[Path] = []
    for root in roots:
        key = str(root.expanduser())
        if key not in seen:
            seen.add(key)
            out.append(root.expanduser())
    return out


def _hf_model_cache_candidates(model_id: str) -> list[Path]:
    slug = "models--" + model_id.replace("/", "--")
    return [root / slug for root in _hf_cache_roots()]


def _fingerprint(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def run_execution_preflight(
    output_dir: str | Path,
    requirement: PreflightRequirement | None = None,
) -> dict[str, Any]:
    """Inspect the current machine without altering it.

    ``ready`` is false whenever a required check fails. Optional missing resources
    are warnings, never silently upgraded to a pass.
    """

    req = requirement or PreflightRequirement()
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    checks: list[PreflightCheck] = []

    python_ok = sys.version_info >= (3, 11)
    checks.append(
        PreflightCheck(
            "python",
            PASS if python_ok else FAIL,
            True,
            {"version": platform.python_version(), "executable": sys.executable},
        )
    )

    cuda_available = bool(torch.cuda.is_available())
    cuda_details: dict[str, Any] = {
        "available": cuda_available,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "device_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        device = torch.device("cuda:0")
        props = torch.cuda.get_device_properties(device)
        cuda_details.update(
            {
                "device_name": torch.cuda.get_device_name(device),
                "total_memory_gib": _gib(int(props.total_memory)),
                "compute_capability": list(torch.cuda.get_device_capability(device)),
            }
        )
    available_vram = float(cuda_details.get("total_memory_gib", 0.0))
    cuda_ok = cuda_available and available_vram >= float(req.min_cuda_vram_gib)
    if not req.require_cuda:
        cuda_status = PASS if cuda_available else WARN
    else:
        cuda_status = PASS if cuda_ok else FAIL
    cuda_details["minimum_required_vram_gib"] = float(req.min_cuda_vram_gib)
    checks.append(PreflightCheck("cuda", cuda_status, req.require_cuda, cuda_details))

    ram_bytes = _system_ram_bytes()
    ram_gib = _gib(ram_bytes) if ram_bytes is not None else 0.0
    ram_ok = ram_bytes is not None and ram_gib >= float(req.min_system_ram_gib)
    checks.append(
        PreflightCheck(
            "system_ram",
            PASS if ram_ok or req.min_system_ram_gib == 0 else FAIL,
            req.min_system_ram_gib > 0,
            {"total_gib": ram_gib if ram_bytes is not None else None,
             "minimum_required_gib": float(req.min_system_ram_gib)},
        )
    )

    vizdoom_version = _package_version("vizdoom")
    ipc_available, ipc_error = _vizdoom_ipc_probe() if vizdoom_version else (False, "vizdoom is not installed")
    vizdoom_ok = bool(vizdoom_version and ipc_available)
    checks.append(
        PreflightCheck(
            "vizdoom",
            PASS if vizdoom_ok else (FAIL if req.require_vizdoom else WARN),
            req.require_vizdoom,
            {"package_version": vizdoom_version, "local_ipc_socket_available": ipc_available,
             "local_ipc_socket_error": ipc_error},
        )
    )

    transformers_version = _package_version("transformers")
    checks.append(
        PreflightCheck(
            "transformers",
            PASS if transformers_version else (FAIL if req.require_transformers else WARN),
            req.require_transformers,
            {"package_version": transformers_version},
        )
    )

    cache_candidates = _hf_model_cache_candidates(req.vjepa_model)
    cached = [str(path) for path in cache_candidates if path.exists()]
    cache_status = PASS if cached else (FAIL if req.require_vjepa_cache else WARN)
    checks.append(
        PreflightCheck(
            "vjepa_cache",
            cache_status,
            req.require_vjepa_cache,
            {
                "model": req.vjepa_model,
                "cached_paths": cached,
                "searched_paths": [str(path) for path in cache_candidates],
                "note": "Cache presence does not prove model integrity; real qualification loads and fingerprints the model.",
            },
        )
    )

    volume_path = Path(req.persistent_volume_path).expanduser().resolve()
    volume_size_raw = os.getenv(req.persistent_volume_size_env, "")
    volume_id = os.getenv(req.persistent_volume_id_env, "")
    try:
        provisioned_gib = float(volume_size_raw)
        quota_value_valid = math.isfinite(provisioned_gib) and provisioned_gib > 0
    except (TypeError, ValueError):
        provisioned_gib = 0.0
        quota_value_valid = False
    mount_ok = volume_path.is_dir() and os.path.ismount(volume_path)
    usage_bytes = None
    usage_error = None
    du = shutil.which("du")
    if du and volume_path.is_dir():
        try:
            result = subprocess.run(
                [du, "-sx", "--block-size=1", str(volume_path)],
                check=True, capture_output=True, text=True, timeout=120,
            )
            usage_bytes = int(result.stdout.split()[0])
        except (OSError, ValueError, subprocess.SubprocessError, IndexError) as exc:
            usage_error = repr(exc)
    used_gib = _gib(usage_bytes) if usage_bytes is not None else None
    remaining_gib = provisioned_gib - used_gib if used_gib is not None and quota_value_valid else None
    if req.require_persistent_volume_quota:
        disk_ok = bool(
            mount_ok and quota_value_valid and bool(volume_id) and remaining_gib is not None
            and remaining_gib >= float(req.min_disk_free_gib)
        )
        disk_details = {
            "path": str(volume_path),
            "is_mountpoint": mount_ok,
            "provisioned_quota_gib": provisioned_gib if quota_value_valid else None,
            "measured_used_gib": used_gib,
            "estimated_remaining_quota_gib": remaining_gib,
            "minimum_remaining_quota_gib": float(req.min_disk_free_gib),
            "volume_id": volume_id or None,
            "volume_id_env": req.persistent_volume_id_env,
            "volume_size_env": req.persistent_volume_size_env,
            "usage_measurement_error": usage_error,
            "note": "Quota comes from provisioned RunPod volume metadata passed at pod creation; statvfs backing-filesystem free space is not used.",
        }
    else:
        disk = shutil.disk_usage(out)
        free_gib = _gib(disk.free)
        disk_ok = free_gib >= float(req.min_disk_free_gib)
        disk_details = {"path": str(out), "free_gib": free_gib,
                        "minimum_required_gib": float(req.min_disk_free_gib),
                        "persistent_volume_quota_required": False}
    checks.append(
        PreflightCheck(
            "disk",
            PASS if disk_ok else FAIL,
            True,
            disk_details,
        )
    )

    probe = out / ".awa-preflight-write-test"
    try:
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
        write_status = PASS
        write_error = None
    except OSError as exc:
        write_status = FAIL
        write_error = repr(exc)
    checks.append(
        PreflightCheck(
            "output_write",
            write_status,
            True,
            {"path": str(out), "error": write_error},
        )
    )

    nvidia_smi = shutil.which("nvidia-smi")
    checks.append(
        PreflightCheck(
            "nvidia_smi",
            PASS if nvidia_smi else WARN,
            False,
            {"path": nvidia_smi},
        )
    )

    required_failures = [check.name for check in checks if check.required and check.status == FAIL]
    environment = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
    }
    fingerprint_payload = {
        "requirements": asdict(req),
        "environment": environment,
        "checks": [check.to_dict() for check in checks],
    }
    return {
        "format": "awa-v2.35-execution-preflight-v1",
        "ready": not required_failures,
        "required_failures": required_failures,
        "requirements": asdict(req),
        "environment": environment,
        "checks": [check.to_dict() for check in checks],
        "environment_fingerprint_sha256": _fingerprint(fingerprint_payload),
        "claim_boundary": (
            "Preflight proves only that requested dependencies/resources are observable on this machine. "
            "It does not prove benchmark performance, model correctness, or future resource availability."
        ),
    }


def write_execution_preflight(report: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
