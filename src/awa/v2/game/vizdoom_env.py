from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable
import io
import json
import os
import struct
import tempfile

import numpy as np

from .action_codec import HybridGameActionCodec


VIZDOOM_ACTION_DIM = 4
VIZDOOM_GOAL_DIM = 8
SNAPSHOT_MAGIC = b"AWAVZD1\0"


class ViZDoomScenario(str, Enum):
    BASIC = "basic"
    MY_WAY_HOME = "my_way_home"
    HEALTH_GATHERING = "health_gathering"
    DEFEND_THE_CENTER = "defend_the_center"
    PREDICT_POSITION = "predict_position"
    TAKE_COVER = "take_cover"
    DEADLY_CORRIDOR = "deadly_corridor"
    DEATHMATCH = "deathmatch"


SCENARIO_CONFIGS: dict[ViZDoomScenario, str] = {
    ViZDoomScenario.BASIC: "basic.cfg",
    ViZDoomScenario.MY_WAY_HOME: "my_way_home.cfg",
    ViZDoomScenario.HEALTH_GATHERING: "health_gathering.cfg",
    ViZDoomScenario.DEFEND_THE_CENTER: "defend_the_center.cfg",
    ViZDoomScenario.PREDICT_POSITION: "predict_position.cfg",
    ViZDoomScenario.TAKE_COVER: "take_cover.cfg",
    ViZDoomScenario.DEADLY_CORRIDOR: "deadly_corridor.cfg",
    ViZDoomScenario.DEATHMATCH: "deathmatch.cfg",
}

# Stable telemetry ABI. Missing variables are encoded as zero so scenario-specific
# configs do not change the observation dimension.
TELEMETRY_FIELDS: tuple[tuple[str, float], ...] = (
    ("HEALTH", 100.0),
    ("ARMOR", 100.0),
    ("SELECTED_WEAPON", 10.0),
    ("SELECTED_WEAPON_AMMO", 100.0),
    ("KILLCOUNT", 20.0),
    ("ITEMCOUNT", 20.0),
    ("POSITION_X", 1024.0),
    ("POSITION_Y", 1024.0),
    ("POSITION_Z", 256.0),
    ("ANGLE", 360.0),
    ("PITCH", 180.0),
    ("ATTACK_READY", 1.0),
    ("ON_GROUND", 1.0),
    ("DEAD", 1.0),
)
VIZDOOM_TELEMETRY_DIM = len(TELEMETRY_FIELDS) + 3  # episode time, last reward, alive


@dataclass(frozen=True)
class ViZDoomConfig:
    scenario: ViZDoomScenario = ViZDoomScenario.MY_WAY_HOME
    scenario_path: str | None = None
    frame_skip: int = 4
    seed: int = 215
    track: str = "structured"  # structured | pixel
    visible: bool = False
    sound: bool = False
    screen_resolution: str = "RES_320X240"
    screen_format: str = "RGB24"
    turn_delta_degrees: float = 12.0
    move_delta_units: float = 12.0
    max_episode_steps: int | None = None
    normalize_telemetry: bool = True
    override_buttons: bool = True

    def __post_init__(self):
        if self.track not in {"structured", "pixel"}:
            raise ValueError("track must be 'structured' or 'pixel'")
        if int(self.frame_skip) <= 0:
            raise ValueError("frame_skip must be positive")
        if not np.isfinite(self.turn_delta_degrees) or self.turn_delta_degrees <= 0:
            raise ValueError("turn_delta_degrees must be finite and positive")
        if not np.isfinite(self.move_delta_units) or self.move_delta_units <= 0:
            raise ValueError("move_delta_units must be finite and positive")
        if self.max_episode_steps is not None and int(self.max_episode_steps) <= 0:
            raise ValueError("max_episode_steps must be positive")

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["scenario"] = self.scenario.value
        return row


@dataclass(frozen=True)
class ViZDoomStepInfo:
    success: bool
    player_dead: bool
    episode_time: int
    last_reward: float
    scenario: str
    frame_skip: int
    snapshot_fidelity: str = "world_state_only"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_vizdoom():
    try:
        import vizdoom as vzd
    except ImportError as exc:
        raise ImportError(
            "ViZDoom support is optional. Install `aether-world-agent[vizdoom]` or `pip install vizdoom`."
        ) from exc
    return vzd


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    text = str(value)
    return text.rsplit(".", 1)[-1]


def resolve_vizdoom_scenario_path(vzd: Any, scenario: ViZDoomScenario, explicit: str | None = None) -> Path:
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    filename = SCENARIO_CONFIGS[ViZDoomScenario(scenario)]
    candidates: list[Path] = []
    for attr in ("scenarios_path", "scenario_path"):
        base = getattr(vzd, attr, None)
        if base:
            candidates.append(Path(base) / filename)
    module_path = getattr(vzd, "__file__", None)
    if module_path:
        candidates.append(Path(module_path).resolve().parent / "scenarios" / filename)
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        f"could not locate ViZDoom scenario {filename}; pass scenario_path explicitly"
    )


class ViZDoomActionMapper:
    """Map Aether's 4-D hybrid ABI to ViZDoom's configured button vector.

    Aether action semantics for Doom are:
      action[0] = turn left/right in [-1,1]
      action[1] = move forward/backward in [-1,1]
      action[2] = attack (categorical)
      action[3] = use/interact (categorical)
    """

    def __init__(self, button_names: Iterable[str], *, turn_delta: float = 12.0, move_delta: float = 12.0, threshold: float = 0.25):
        self.button_names = tuple(str(x) for x in button_names)
        self.turn_delta = float(turn_delta)
        self.move_delta = float(move_delta)
        self.codec = HybridGameActionCodec(threshold=threshold)

    def map(self, action: Any) -> list[float]:
        hybrid = self.codec.decode(action)
        turn, forward = hybrid.movement
        out: list[float] = []
        for name in self.button_names:
            key = name.upper()
            if key == "TURN_LEFT_RIGHT_DELTA":
                out.append(float(turn * self.turn_delta))
            elif key == "MOVE_FORWARD_BACKWARD_DELTA":
                out.append(float(forward * self.move_delta))
            elif key == "TURN_LEFT":
                out.append(float(turn < -self.codec.threshold))
            elif key == "TURN_RIGHT":
                out.append(float(turn > self.codec.threshold))
            elif key == "MOVE_FORWARD":
                out.append(float(forward > self.codec.threshold))
            elif key == "MOVE_BACKWARD":
                out.append(float(forward < -self.codec.threshold))
            elif key == "ATTACK":
                out.append(float(hybrid.attack))
            elif key == "USE":
                out.append(float(hybrid.interact))
            else:
                out.append(0.0)
        return out


def _frame_to_hwc_rgb(frame: Any) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
        arr = np.moveaxis(arr[:3], 0, -1)
    elif arr.ndim == 3 and arr.shape[-1] in (1, 3, 4):
        arr = arr[..., :3]
    else:
        raise ValueError(f"unsupported ViZDoom frame shape: {arr.shape}")
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


class ViZDoomAetherEnv:
    """Direct ViZDoom adapter for Aether structured or pixel tracks.

    The class accepts an injected DoomGame-compatible object for tests and offline
    qualification. When no game is provided, the optional `vizdoom` package is used.
    """

    action_dim = VIZDOOM_ACTION_DIM
    goal_dim = VIZDOOM_GOAL_DIM
    telemetry_dim = VIZDOOM_TELEMETRY_DIM
    action_low = np.full(VIZDOOM_ACTION_DIM, -1.0, dtype=np.float32)
    action_high = np.full(VIZDOOM_ACTION_DIM, 1.0, dtype=np.float32)

    def __init__(self, config: ViZDoomConfig | None = None, *, game: Any | None = None, vizdoom_module: Any | None = None):
        self.config = config or ViZDoomConfig()
        self.vzd = vizdoom_module
        self.game = game
        self._owns_game = game is None
        self._initialized = False
        self._episode_steps = 0
        self._episode_return = 0.0
        self._last_reward = 0.0
        self._last_frame: np.ndarray | None = None
        self._last_telemetry = np.zeros(self.telemetry_dim, dtype=np.float32)
        self._available_variable_enums: dict[str, Any] = {}
        self._button_names: tuple[str, ...] = ()
        self._mapper: ViZDoomActionMapper | None = None
        self._scenario_semantics = None
        self._setup()
        from .vizdoom_semantics import ViZDoomScenarioSemantics
        self._scenario_semantics = ViZDoomScenarioSemantics(self.config.scenario)

    @property
    def observation_dim(self) -> int | None:
        return self.telemetry_dim if self.config.track == "structured" else None

    @property
    def frame_shape(self) -> tuple[int, int, int] | None:
        if self._last_frame is None:
            return None
        return tuple(int(x) for x in self._last_frame.shape)

    def _setup(self) -> None:
        if self.game is None:
            self.vzd = self.vzd or _load_vizdoom()
            self.game = self.vzd.DoomGame()
            scenario_path = resolve_vizdoom_scenario_path(self.vzd, self.config.scenario, self.config.scenario_path)
            self.game.load_config(str(scenario_path))
            # Explicit RGB contract for V-JEPA/pixel datasets.
            if hasattr(self.vzd, "ScreenResolution") and hasattr(self.vzd.ScreenResolution, self.config.screen_resolution):
                self.game.set_screen_resolution(getattr(self.vzd.ScreenResolution, self.config.screen_resolution))
            if hasattr(self.vzd, "ScreenFormat") and hasattr(self.vzd.ScreenFormat, self.config.screen_format):
                self.game.set_screen_format(getattr(self.vzd.ScreenFormat, self.config.screen_format))
            if hasattr(self.game, "set_window_visible"):
                self.game.set_window_visible(bool(self.config.visible))
            if hasattr(self.game, "set_sound_enabled"):
                self.game.set_sound_enabled(bool(self.config.sound))
            if self.config.override_buttons:
                buttons = []
                for name, max_value in (
                    ("TURN_LEFT_RIGHT_DELTA", self.config.turn_delta_degrees),
                    ("MOVE_FORWARD_BACKWARD_DELTA", self.config.move_delta_units),
                    ("ATTACK", 1.0),
                    ("USE", 1.0),
                ):
                    enum = getattr(getattr(self.vzd, "Button", object()), name, None)
                    if enum is not None:
                        buttons.append(enum)
                if buttons and hasattr(self.game, "set_available_buttons"):
                    self.game.set_available_buttons(buttons)
                    if hasattr(self.game, "set_button_max_value"):
                        for enum, max_value in zip(buttons, (self.config.turn_delta_degrees, self.config.move_delta_units, 1.0, 1.0)):
                            try:
                                self.game.set_button_max_value(enum, float(max_value))
                            except Exception:
                                pass
            # Add a stable superset of variables when available. Missing values are zero-filled.
            for name, _ in TELEMETRY_FIELDS:
                enum = getattr(getattr(self.vzd, "GameVariable", object()), name, None)
                if enum is not None:
                    self._available_variable_enums[name] = enum
                    try:
                        self.game.add_available_game_variable(enum)
                    except Exception:
                        pass
            self.game.set_seed(int(self.config.seed))
            self.game.init()
            self._initialized = True
        else:
            self.vzd = self.vzd or getattr(self.game, "vizdoom_module", None)
            self._initialized = True
            # Fake/embedded games can expose a mapping of names to variable handles.
            mapping = getattr(self.game, "variable_enums", None)
            if isinstance(mapping, dict):
                self._available_variable_enums.update(mapping)
        available = list(self.game.get_available_buttons()) if hasattr(self.game, "get_available_buttons") else []
        self._button_names = tuple(_enum_name(x) for x in available)
        if not self._button_names:
            # Test doubles and custom wrappers can implement this canonical order.
            self._button_names = ("TURN_LEFT_RIGHT_DELTA", "MOVE_FORWARD_BACKWARD_DELTA", "ATTACK", "USE")
        self._mapper = ViZDoomActionMapper(
            self._button_names,
            turn_delta=self.config.turn_delta_degrees,
            move_delta=self.config.move_delta_units,
        )

    def goal_vector(self) -> np.ndarray:
        vec = np.zeros(VIZDOOM_GOAL_DIM, dtype=np.float32)
        vec[list(ViZDoomScenario).index(self.config.scenario)] = 1.0
        return vec

    def _game_variable(self, name: str) -> float:
        if hasattr(self.game, "get_game_variable"):
            handle = self._available_variable_enums.get(name, name)
            try:
                value = float(self.game.get_game_variable(handle))
                return value if np.isfinite(value) else 0.0
            except Exception:
                pass
        values = getattr(self.game, "game_variables", None)
        if isinstance(values, dict):
            value = float(values.get(name, 0.0))
            return value if np.isfinite(value) else 0.0
        return 0.0

    def _telemetry(self) -> np.ndarray:
        values: list[float] = []
        for name, scale in TELEMETRY_FIELDS:
            value = self._game_variable(name)
            values.append(value / scale if self.config.normalize_telemetry else value)
        episode_time = float(self.game.get_episode_time()) if hasattr(self.game, "get_episode_time") else float(self._episode_steps)
        dead = bool(self.game.is_player_dead()) if hasattr(self.game, "is_player_dead") else self._game_variable("DEAD") > 0.5
        extra = [episode_time / 2100.0 if self.config.normalize_telemetry else episode_time,
                 self._last_reward / 100.0 if self.config.normalize_telemetry else self._last_reward,
                 0.0 if dead else 1.0]
        out = np.asarray(values + extra, dtype=np.float32)
        if not np.all(np.isfinite(out)):
            raise RuntimeError("ViZDoom telemetry contains non-finite values")
        self._last_telemetry = out
        return out

    def _read_state(self) -> tuple[np.ndarray | None, np.ndarray]:
        state = self.game.get_state() if hasattr(self.game, "get_state") else None
        if state is not None and getattr(state, "screen_buffer", None) is not None:
            self._last_frame = _frame_to_hwc_rgb(state.screen_buffer)
        telemetry = self._telemetry()
        return (None if self._last_frame is None else self._last_frame.copy()), telemetry

    def current_frame(self) -> np.ndarray:
        if self._last_frame is None:
            frame, _ = self._read_state()
            if frame is None:
                raise RuntimeError("ViZDoom has no screen buffer; ensure screen buffer is enabled")
        return self._last_frame.copy()  # type: ignore[union-attr]

    def telemetry_vector(self) -> np.ndarray:
        return self._last_telemetry.copy()

    def _observation(self) -> np.ndarray:
        frame, telemetry = self._read_state()
        if self.config.track == "structured":
            return telemetry
        if frame is None:
            raise RuntimeError("pixel track requires ViZDoom screen_buffer")
        return frame

    def reset(self, *, seed: int | None = None):
        if seed is not None and hasattr(self.game, "set_seed"):
            self.game.set_seed(int(seed))
        if hasattr(self.game, "new_episode"):
            self.game.new_episode()
        self._episode_steps = 0; self._episode_return = 0.0; self._last_reward = 0.0
        self._last_frame = None
        obs = self._observation()
        if self._scenario_semantics is not None:
            self._scenario_semantics.reset(self._game_variable)
        return obs, {"scenario": self.config.scenario.value, "seed": seed, "goal": self.goal_vector().copy()}

    def step(self, action: Any):
        if self._mapper is None:
            raise RuntimeError("ViZDoom action mapper is not initialized")
        mapped = self._mapper.map(action)
        reward = float(self.game.make_action(mapped, int(self.config.frame_skip)))
        if not np.isfinite(reward):
            raise RuntimeError("ViZDoom returned non-finite reward")
        self._episode_steps += 1; self._episode_return += reward; self._last_reward = reward
        finished = bool(self.game.is_episode_finished()) if hasattr(self.game, "is_episode_finished") else False
        player_dead = bool(self.game.is_player_dead()) if hasattr(self.game, "is_player_dead") else False
        truncated = bool(self.config.max_episode_steps is not None and self._episode_steps >= int(self.config.max_episode_steps) and not finished)
        terminated = bool(finished)
        # Terminal states can make get_state() return None. Preserve the last valid screen/telemetry;
        # next-observation values are ignored by bootstrapping when done=True.
        obs = self._observation()
        if self._scenario_semantics is not None:
            success, _ = self._scenario_semantics.update(
                self._game_variable, finished=finished, truncated=truncated, player_dead=player_dead,
                reward=reward, episode_return=self._episode_return,
            )
        else:
            success = bool(finished and not player_dead and reward > 0.0)
        info = ViZDoomStepInfo(
            success=success,
            player_dead=player_dead,
            episode_time=int(self.game.get_episode_time()) if hasattr(self.game, "get_episode_time") else self._episode_steps,
            last_reward=reward,
            scenario=self.config.scenario.value,
            frame_skip=int(self.config.frame_skip),
        ).to_dict()
        info["episode_return"] = float(self._episode_return)
        return obs, reward, terminated, truncated, info

    def constraint_labels(self) -> np.ndarray:
        if self._scenario_semantics is not None:
            return self._scenario_semantics.constraint_labels()
        health = self._game_variable("HEALTH")
        dead = bool(self.game.is_player_dead()) if hasattr(self.game, "is_player_dead") else self._game_variable("DEAD") > 0.5
        return np.asarray([float(dead), float(0.0 < health < 25.0), 0.0, 0.0], dtype=np.float32)

    def snapshot(self) -> bytes:
        if not hasattr(self.game, "save"):
            raise RuntimeError("this ViZDoom backend does not support save()")
        with tempfile.NamedTemporaryFile(suffix=".vzd", delete=False) as tmp:
            path = Path(tmp.name)
        try:
            self.game.save(str(path))
            state_bytes = path.read_bytes()
        finally:
            path.unlink(missing_ok=True)
        meta = json.dumps({
            "episode_steps": self._episode_steps,
            "episode_return": self._episode_return,
            "last_reward": self._last_reward,
            "scenario": self.config.scenario.value,
            # ViZDoom's native load restores world state but does not rewind internal
            # tic/total-reward accounting; callers must not claim perfect counterfactual time replay.
            "fidelity": "world_state_only",
        }, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return SNAPSHOT_MAGIC + struct.pack(">I", len(meta)) + meta + state_bytes

    def restore(self, snapshot: bytes):
        raw = bytes(snapshot)
        if not raw.startswith(SNAPSHOT_MAGIC) or len(raw) < len(SNAPSHOT_MAGIC) + 4:
            raise ValueError("invalid Aether ViZDoom snapshot")
        offset = len(SNAPSHOT_MAGIC)
        meta_len = struct.unpack(">I", raw[offset:offset+4])[0]; offset += 4
        meta = json.loads(raw[offset:offset+meta_len].decode("utf-8")); state_bytes = raw[offset+meta_len:]
        if meta.get("scenario") != self.config.scenario.value:
            raise ValueError("snapshot scenario mismatch")
        if not hasattr(self.game, "load"):
            raise RuntimeError("this ViZDoom backend does not support load()")
        with tempfile.NamedTemporaryFile(suffix=".vzd", delete=False) as tmp:
            path = Path(tmp.name); tmp.write(state_bytes); tmp.flush()
        try:
            self.game.load(str(path))
        finally:
            path.unlink(missing_ok=True)
        self._episode_steps = int(meta["episode_steps"])
        self._episode_return = float(meta["episode_return"])
        self._last_reward = float(meta["last_reward"])
        self._last_frame = None
        obs = self._observation()
        return obs, {"restored": True, "snapshot_fidelity": meta.get("fidelity", "unknown")}

    def close(self):
        if self._owns_game and self.game is not None and hasattr(self.game, "close"):
            self.game.close()
        self._initialized = False


def make_vizdoom_env(
    scenario: ViZDoomScenario | str = ViZDoomScenario.MY_WAY_HOME,
    *,
    track: str = "structured",
    seed: int = 215,
    scenario_path: str | None = None,
    frame_skip: int = 4,
    visible: bool = False,
) -> ViZDoomAetherEnv:
    return ViZDoomAetherEnv(ViZDoomConfig(
        scenario=ViZDoomScenario(scenario), track=track, seed=seed, scenario_path=scenario_path,
        frame_skip=frame_skip, visible=visible,
    ))
