from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .vizdoom_dataset import ViZDoomExplorerPolicy, collect_vizdoom_dataset
from .vizdoom_env import ViZDoomConfig, ViZDoomAetherEnv, ViZDoomScenario
from .vizdoom_vjepa import DEFAULT_VJEPA2_MODEL
from awa.v2.video_representation import VJEPA2HFBackbone


@dataclass(frozen=True)
class RealViZDoomQualificationConfig:
    scenario: str = "my_way_home"
    track: str = "structured"
    episodes: int = 4
    horizon: int = 600
    frame_skip: int = 4
    seed: int = 2300
    visible: bool = False
    require_vjepa: bool = False
    vjepa_model: str = DEFAULT_VJEPA2_MODEL
    vjepa_local_files_only: bool = True
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.track not in {"structured", "pixel"}:
            raise ValueError("track must be structured or pixel")
        if self.require_vjepa and self.track != "pixel":
            raise ValueError("V-JEPA qualification requires pixel track")
        if self.episodes <= 0 or self.horizon <= 0 or self.frame_skip <= 0:
            raise ValueError("episodes, horizon and frame_skip must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_backend(env: ViZDoomAetherEnv) -> dict[str, Any]:
    module = getattr(env, "vzd", None)
    game = getattr(env, "game", None)
    module_name = str(getattr(module, "__name__", ""))
    module_file = str(getattr(module, "__file__", ""))
    game_type = type(game)
    game_module = str(getattr(game_type, "__module__", ""))
    game_name = str(getattr(game_type, "__name__", ""))
    injected = not bool(getattr(env, "_owns_game", False))
    real = bool(
        not injected
        and module_name == "vizdoom"
        and module_file
        and "vizdoom" in game_module.lower()
        and "fake" not in game_name.lower()
    )
    try:
        version = importlib.metadata.version("vizdoom") if real else None
    except importlib.metadata.PackageNotFoundError:
        version = None
    return {
        "real_vizdoom": real,
        "injected_game": injected,
        "module_name": module_name,
        "module_file": module_file,
        "game_class": f"{game_module}.{game_name}",
        "package_version": version,
    }


def assert_real_backend(env: ViZDoomAetherEnv) -> dict[str, Any]:
    provenance = inspect_backend(env)
    if not provenance["real_vizdoom"]:
        raise RuntimeError(
            "real ViZDoom qualification refuses injected/fake backends; install ViZDoom and use the native DoomGame path"
        )
    return provenance


def inspect_vjepa_backbone(backbone: VJEPA2HFBackbone) -> dict[str, Any]:
    model_type = type(backbone.model)
    processor_type = type(backbone.processor)
    injected = bool(getattr(backbone, "_awa_injected_components", False))
    model_module = str(getattr(model_type, "__module__", ""))
    processor_module = str(getattr(processor_type, "__module__", ""))
    real = bool(
        not injected
        and model_module.startswith("transformers")
        and processor_module.startswith("transformers")
    )
    return {
        "real_backbone": real,
        "injected_components": injected,
        "model_name_or_path": backbone.model_name_or_path,
        "model_class": f"{model_module}.{model_type.__name__}",
        "processor_class": f"{processor_module}.{processor_type.__name__}",
        "fingerprint": backbone.fingerprint(),
        "feature_dim": int(backbone.output_dim),
        "frames_per_clip": int(backbone.frames_per_clip),
    }


def run_real_vizdoom_qualification(
    config: RealViZDoomQualificationConfig,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    scenario = ViZDoomScenario(config.scenario)

    # Probe backend provenance before collecting any evidence.
    probe = ViZDoomAetherEnv(
        ViZDoomConfig(
            scenario=scenario,
            track=config.track,
            frame_skip=config.frame_skip,
            seed=config.seed,
            visible=config.visible,
            max_episode_steps=config.horizon,
        )
    )
    try:
        backend = assert_real_backend(probe)
        frame_shape = probe.frame_shape
    finally:
        probe.close()

    def make_env() -> ViZDoomAetherEnv:
        return ViZDoomAetherEnv(
            ViZDoomConfig(
                scenario=scenario,
                track=config.track,
                frame_skip=config.frame_skip,
                seed=config.seed,
                visible=config.visible,
                max_episode_steps=config.horizon,
            )
        )

    dataset_path = out / f"real_vizdoom_{config.track}.npz"
    report = collect_vizdoom_dataset(
        make_env,
        dataset_path,
        episodes=config.episodes,
        horizon=config.horizon,
        base_seed=config.seed,
        policy=ViZDoomExplorerPolicy(config.seed),
    )

    vjepa: dict[str, Any] = {
        "requested": bool(config.require_vjepa),
        "real_backbone": False,
    }
    if config.require_vjepa:
        backbone = VJEPA2HFBackbone(
            config.vjepa_model,
            local_files_only=config.vjepa_local_files_only,
            device=config.device,
        )
        provenance = inspect_vjepa_backbone(backbone)
        if not provenance["real_backbone"]:
            raise RuntimeError("V-JEPA qualification refuses injected/test-double model components")
        with np.load(dataset_path, allow_pickle=False) as data:
            frames = np.asarray(data["observations"])
        if frames.ndim != 4 or frames.shape[-1] != 3 or len(frames) == 0:
            raise RuntimeError("pixel dataset does not contain HWC RGB frames")
        count = int(backbone.frames_per_clip)
        selected = frames[: min(len(frames), count)]
        if len(selected) < count:
            pad = np.repeat(selected[-1:,...], count - len(selected), axis=0)
            selected = np.concatenate([selected, pad], axis=0)
        clip = torch.from_numpy(selected).permute(0, 3, 1, 2).unsqueeze(0)
        with torch.no_grad():
            features = backbone(clip)
        if features.ndim != 2 or features.shape[0] != 1 or not torch.isfinite(features).all():
            raise RuntimeError("real V-JEPA qualification produced invalid features")
        vjepa = provenance | {
            "requested": True,
            "feature_shape": list(features.shape),
            "finite_features": True,
        }

    receipt = {
        "format": "awa-v2.30-real-vizdoom-qualification-v1",
        "scenario": scenario.value,
        "track": config.track,
        "config": config.to_dict(),
        "backend": backend,
        "collection": report.to_dict(),
        "frame_shape_probe": list(frame_shape) if frame_shape is not None else None,
        "vjepa": vjepa,
        "claim_boundary": (
            "This receipt proves execution through the real ViZDoom backend and, when requested, a non-injected V-JEPA path. "
            "It does not by itself prove benchmark superiority or general visual-control competence."
        ),
    }
    receipt_path = out / "real_vizdoom_qualification.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt
