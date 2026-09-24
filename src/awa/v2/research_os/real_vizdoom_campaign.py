"""Real-backend ViZDoom training campaign wrapper.

This module turns the historical ViZDoom campaign into a fail-closed hardware
qualification surface. It refuses injected Doom games and, for pixel campaigns,
refuses injected V-JEPA components before any benchmark receipt is committed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from awa.v2.game.vizdoom_campaign import ViZDoomTrainingCampaign
from awa.v2.game.vizdoom_env import ViZDoomAetherEnv, ViZDoomConfig, ViZDoomScenario
from awa.v2.game.vizdoom_qualification import assert_real_backend, inspect_vjepa_backbone
from awa.v2.video_representation import VJEPA2HFBackbone, VideoClipSpec


def _sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_config(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("ViZDoom campaign config must decode to a mapping")
    track = str(payload.get("track", "structured"))
    if track not in {"structured", "pixel"}:
        raise ValueError("ViZDoom campaign track must be structured or pixel")
    stages = payload.get("stages") or []
    if not isinstance(stages, list) or not stages:
        raise ValueError("ViZDoom campaign requires at least one stage")
    for row in stages:
        ViZDoomScenario(str(row["scenario"]))
    return payload


def build_real_vizdoom_campaign_plan(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    stages = [
        {
            "name": str(row["name"]),
            "scenario": str(row["scenario"]),
            "target_transitions": int(row["target_transitions"]),
        }
        for row in cfg["stages"]
    ]
    return {
        "format": "awa-v2.35-real-vizdoom-campaign-plan-v1",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "out_dir": str(Path(out_dir).resolve()),
        "track": str(cfg.get("track", "structured")),
        "stages": stages,
        "requires_real_vizdoom": True,
        "requires_real_vjepa": str(cfg.get("track", "structured")) == "pixel",
        "claim_boundary": "This is a plan only. No hardware or benchmark result is implied until a real receipt is written.",
    }


def run_real_vizdoom_training_campaign(
    config_path: str | Path,
    out_dir: str | Path,
    *,
    stop_after_stage: str | None = None,
    allow_vjepa_download: bool = False,
) -> dict[str, Any]:
    """Execute the existing training campaign only through genuine external backends."""

    config_path = Path(config_path).resolve()
    cfg = _load_config(config_path)
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    track = str(cfg.get("track", "structured"))
    env_cfg = cfg.get("environment") or {}
    runtime_cfg = cfg.get("runtime") or {}
    first_scenario = ViZDoomScenario(str(cfg["stages"][0]["scenario"]))

    def make_env(scenario: str, requested_track: str) -> ViZDoomAetherEnv:
        return ViZDoomAetherEnv(
            ViZDoomConfig(
                scenario=ViZDoomScenario(str(scenario)),
                track=requested_track,
                frame_skip=int(env_cfg.get("frame_skip", 4)),
                max_episode_steps=env_cfg.get("max_episode_steps"),
                visible=bool(env_cfg.get("visible", False)),
                sound=bool(env_cfg.get("sound", False)),
            )
        )

    probe = make_env(first_scenario.value, track)
    try:
        backend = assert_real_backend(probe)
    finally:
        probe.close()

    video_backbone = None
    vjepa: dict[str, Any] = {"requested": track == "pixel", "real_backbone": False}
    video_cfg = cfg.get("video") or {}
    clip_spec = VideoClipSpec(
        int(video_cfg.get("frames", 64)),
        int(video_cfg.get("stride", 1)),
    )
    if track == "pixel":
        model = str(video_cfg.get("model", "facebook/vjepa2-vitl-fpc64-256"))
        video_backbone = VJEPA2HFBackbone(
            model,
            local_files_only=not allow_vjepa_download,
            device=str(runtime_cfg.get("device", "cpu")),
        )
        vjepa = inspect_vjepa_backbone(video_backbone) | {"requested": True}
        if not bool(vjepa.get("real_backbone")):
            raise RuntimeError("real pixel campaign refuses injected/test-double V-JEPA components")

    campaign = ViZDoomTrainingCampaign(
        cfg,
        out,
        env_factory=make_env,
        video_backbone=video_backbone,
        clip_spec=clip_spec,
    )
    before = campaign.plan()
    result = campaign.run(stop_after_stage=stop_after_stage)
    registry_failures = campaign.registry.verify()
    if registry_failures:
        raise RuntimeError(f"checkpoint registry integrity failure: {registry_failures}")
    stable = result.get("stable")
    completed_stages = [row.get("stage") for row in result.get("results", [])]
    stable_artifacts: dict[str, Any] = {}
    if isinstance(stable, dict):
        for key in ("world", "actor"):
            checkpoint = Path(str(stable.get(key, "")))
            if checkpoint.exists():
                stable_artifacts[key] = {
                    "path": str(checkpoint.resolve()),
                    "sha256": _sha256(checkpoint),
                    "size_bytes": int(checkpoint.stat().st_size),
                    "registry_sha256": stable.get(f"{key}_sha256"),
                }

    receipt = {
        "format": "awa-v2.36-real-vizdoom-training-campaign-v2",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "track": track,
        "backend": backend,
        "vjepa": vjepa,
        "completed_stages": completed_stages,
        "transitions_before_resume": int(before.get("current_transitions", 0)),
        "transitions": int(result.get("transitions", 0)),
        "transition_delta_this_invocation": int(result.get("transitions", 0)) - int(before.get("current_transitions", 0)),
        "stable_checkpoint": stable,
        "stable_checkpoint_artifacts": stable_artifacts,
        "checkpoint_registry_verified": True,
        "result": result,
        "claim_boundary": (
            "This receipt proves that collection/training/evaluation executed through a real ViZDoom backend "
            "and, for the pixel track, a non-injected V-JEPA backbone. A single campaign is not a multi-seed "
            "generalization claim or proof of superiority over external baselines."
        ),
    }
    receipt_path = out / "real_vizdoom_training_receipt.json"
    temporary = receipt_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    temporary.replace(receipt_path)
    return receipt
