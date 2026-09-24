from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Callable
import base64
import json
import math
import socket
import threading
import uuid

import numpy as np

PROTOCOL_VERSION = "aether.game.v1"
MAX_MESSAGE_BYTES = 8 * 1024 * 1024


def _finite_list(value: Any, *, name: str) -> list[float]:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return [float(x) for x in arr]


def _send_line(sock: socket.socket, obj: dict[str, Any]) -> None:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"
    if len(payload) > MAX_MESSAGE_BYTES:
        raise ValueError("bridge message exceeds maximum size")
    sock.sendall(payload)


def _recv_line(sock: socket.socket, buffer: bytearray) -> dict[str, Any]:
    while True:
        pos = buffer.find(b"\n")
        if pos >= 0:
            raw = bytes(buffer[:pos]); del buffer[:pos + 1]
            if not raw:
                continue
            return json.loads(raw.decode("utf-8"))
        chunk = sock.recv(65536)
        if not chunk:
            raise EOFError("bridge connection closed")
        buffer.extend(chunk)
        if len(buffer) > MAX_MESSAGE_BYTES:
            raise ValueError("bridge message exceeds maximum size")


@dataclass(frozen=True)
class BridgeCapabilities:
    observation_dim: int
    goal_dim: int
    action_dim: int
    supports_snapshot: bool = False
    supports_pixels: bool = False
    frame_shape: tuple[int, int, int] | None = None

    def __post_init__(self):
        if min(int(self.observation_dim), int(self.goal_dim), int(self.action_dim)) <= 0:
            raise ValueError("bridge dimensions must be positive")
        if self.supports_pixels and (self.frame_shape is None or len(self.frame_shape) != 3):
            raise ValueError("pixel bridge requires frame_shape")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        if self.frame_shape is not None:
            d["frame_shape"] = list(self.frame_shape)
        return d

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BridgeCapabilities":
        data = dict(raw)
        if data.get("frame_shape") is not None:
            data["frame_shape"] = tuple(int(x) for x in data["frame_shape"])
        return cls(**data)


@dataclass(frozen=True)
class BridgeTransition:
    observation: np.ndarray
    goal: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class JsonLineGameBridgeClient:
    """Dependency-free request/response bridge for Unity/Godot/custom simulators."""

    def __init__(self, host: str, port: int, *, timeout: float = 10.0):
        self.host = str(host); self.port = int(port); self.timeout = float(timeout)
        self.sock: socket.socket | None = None
        self._buffer = bytearray(); self._counter = 0
        self.capabilities: BridgeCapabilities | None = None

    def connect(self) -> BridgeCapabilities:
        if self.sock is not None:
            return self.capabilities  # type: ignore[return-value]
        self.sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        self.sock.settimeout(self.timeout)
        result = self._request("hello", {"protocol": PROTOCOL_VERSION})
        if result.get("protocol") != PROTOCOL_VERSION:
            raise RuntimeError("game bridge protocol mismatch")
        self.capabilities = BridgeCapabilities.from_dict(result["capabilities"])
        return self.capabilities

    def _request(self, op: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        if self.sock is None:
            raise RuntimeError("bridge client is not connected")
        self._counter += 1; req_id = f"{self._counter}-{uuid.uuid4().hex[:8]}"
        _send_line(self.sock, {"id": req_id, "op": op, "payload": payload or {}})
        row = _recv_line(self.sock, self._buffer)
        if row.get("id") != req_id:
            raise RuntimeError("bridge response id mismatch")
        if not row.get("ok", False):
            raise RuntimeError(str(row.get("error", "game bridge error")))
        return dict(row.get("result") or {})

    def reset(self, *, seed: int | None = None) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        result = self._request("reset", {"seed": seed})
        return (
            np.asarray(result["observation"], dtype=np.float32),
            np.asarray(result["goal"], dtype=np.float32),
            dict(result.get("info") or {}),
        )

    def step(self, action: Any) -> BridgeTransition:
        result = self._request("step", {"action": _finite_list(action, name="action")})
        reward = float(result["reward"])
        if not math.isfinite(reward):
            raise RuntimeError("bridge returned non-finite reward")
        return BridgeTransition(
            np.asarray(result["observation"], dtype=np.float32),
            np.asarray(result["goal"], dtype=np.float32),
            reward,
            bool(result.get("terminated", False)),
            bool(result.get("truncated", False)),
            dict(result.get("info") or {}),
        )

    def snapshot(self) -> bytes:
        result = self._request("snapshot")
        return base64.b64decode(result["snapshot_b64"].encode("ascii"), validate=True)

    def restore(self, snapshot: bytes) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        result = self._request("restore", {"snapshot_b64": base64.b64encode(snapshot).decode("ascii")})
        return np.asarray(result["observation"], dtype=np.float32), np.asarray(result["goal"], dtype=np.float32), dict(result.get("info") or {})

    def close(self) -> None:
        if self.sock is None:
            return
        try:
            self._request("close")
        except Exception:
            pass
        try:
            self.sock.close()
        finally:
            self.sock = None

    def __enter__(self):
        self.connect(); return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


class BridgeEnvironmentAdapter:
    """Gym-like facade over JsonLineGameBridgeClient."""

    def __init__(self, client: JsonLineGameBridgeClient):
        self.client = client
        caps = self.client.connect()
        self.obs_dim = caps.observation_dim; self.goal_dim = caps.goal_dim; self.action_dim = caps.action_dim
        self.action_low = np.full(self.action_dim, -1.0, dtype=np.float32)
        self.action_high = np.full(self.action_dim, 1.0, dtype=np.float32)
        self._goal = np.zeros(self.goal_dim, dtype=np.float32)

    def reset(self, *, seed: int | None = None):
        obs, goal, info = self.client.reset(seed=seed); self._goal = goal
        return obs, info

    def goal_vector(self) -> np.ndarray:
        return self._goal.copy()

    def step(self, action):
        tr = self.client.step(action); self._goal = tr.goal
        return tr.observation, tr.reward, tr.terminated, tr.truncated, tr.info

    def snapshot(self) -> bytes:
        return self.client.snapshot()

    def restore(self, snapshot: bytes):
        obs, goal, info = self.client.restore(snapshot); self._goal = goal
        return obs, info

    def close(self):
        self.client.close()


class BridgeSessionServer:
    """Small reference server used by tests and custom Python environments.

    Unity/Godot implementations need only mirror the documented JSONL operations.
    """

    def __init__(self, env_factory: Callable[[], Any], capabilities: BridgeCapabilities):
        self.env_factory = env_factory; self.capabilities = capabilities

    def _state(self, env, obs, info=None):
        goal = env.goal_vector() if hasattr(env, "goal_vector") else np.zeros(self.capabilities.goal_dim, np.float32)
        return {"observation": _finite_list(obs, name="observation"), "goal": _finite_list(goal, name="goal"), "info": dict(info or {})}

    def serve_socket(self, sock: socket.socket) -> None:
        env = self.env_factory(); buffer = bytearray(); running = True
        try:
            while running:
                req = _recv_line(sock, buffer); req_id = req.get("id"); op = req.get("op"); payload = req.get("payload") or {}
                try:
                    if op == "hello":
                        if payload.get("protocol") != PROTOCOL_VERSION:
                            raise RuntimeError("protocol mismatch")
                        result = {"protocol": PROTOCOL_VERSION, "capabilities": self.capabilities.to_dict()}
                    elif op == "reset":
                        out = env.reset(seed=payload.get("seed")) if payload.get("seed") is not None else env.reset()
                        obs, info = (out if isinstance(out, tuple) and len(out) == 2 else (out, {}))
                        result = self._state(env, obs, info)
                    elif op == "step":
                        action = np.asarray(payload["action"], dtype=np.float32)
                        out = env.step(action)
                        if len(out) == 5:
                            obs, reward, terminated, truncated, info = out
                        elif len(out) == 4:
                            obs, reward, done, info = out; terminated, truncated = bool(done), False
                        else:
                            raise RuntimeError("unsupported step tuple")
                        result = {**self._state(env, obs, info), "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated)}
                    elif op == "snapshot":
                        if not self.capabilities.supports_snapshot:
                            raise RuntimeError("snapshot unsupported")
                        snap = env.snapshot();
                        if not isinstance(snap, (bytes, bytearray)):
                            raise RuntimeError("bridge snapshot() must return bytes")
                        result = {"snapshot_b64": base64.b64encode(bytes(snap)).decode("ascii")}
                    elif op == "restore":
                        if not self.capabilities.supports_snapshot:
                            raise RuntimeError("restore unsupported")
                        snap = base64.b64decode(payload["snapshot_b64"].encode("ascii"), validate=True)
                        out = env.restore(snap)
                        obs, info = (out if isinstance(out, tuple) and len(out) == 2 else (out, {}))
                        result = self._state(env, obs, info)
                    elif op == "close":
                        result = {}; running = False
                    else:
                        raise RuntimeError(f"unsupported op: {op}")
                    _send_line(sock, {"id": req_id, "ok": True, "result": result})
                except Exception as exc:
                    _send_line(sock, {"id": req_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        except EOFError:
            pass
        finally:
            close = getattr(env, "close", None)
            if callable(close): close()
            try: sock.close()
            except OSError: pass


def serve_tcp(env_factory: Callable[[], Any], capabilities: BridgeCapabilities, host: str = "127.0.0.1", port: int = 0):
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((host, int(port))); listener.listen(1)
    actual_port = listener.getsockname()[1]
    server = BridgeSessionServer(env_factory, capabilities)
    stop = threading.Event()
    def run():
        try:
            while not stop.is_set():
                listener.settimeout(0.2)
                try: conn, _ = listener.accept()
                except socket.timeout: continue
                server.serve_socket(conn)
        finally:
            listener.close()
    thread = threading.Thread(target=run, daemon=True); thread.start()
    return actual_port, stop, thread
