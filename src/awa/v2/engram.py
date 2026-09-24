from __future__ import annotations

from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
import json
import numpy as np


@dataclass
class EngramEntry:
    key: str
    pattern: str
    vector: np.ndarray
    payload: dict
    score: float = 1.0
    uses: int = 0

    def to_dict(self):
        d = asdict(self)
        d["vector"] = self.vector.tolist()
        return d


class EngramLite:
    """Bounded sparse pattern memory for reusable regimes, tactics and failures."""

    def __init__(self, capacity=100_000, buckets=4096, seed=0):
        self.capacity = int(capacity)
        self.buckets = int(buckets)
        self.seed = int(seed)
        if self.capacity <= 0 or self.buckets <= 0:
            raise ValueError("Engram capacity and bucket count must be > 0")
        self._by_bucket = defaultdict(list)
        self._order = deque()

    def _bucket(self, vector):
        x = np.asarray(vector, dtype=np.float32).reshape(-1)
        signs = (x[: min(64, len(x))] >= 0).astype(np.uint8).tobytes()
        raw = (
            self.seed.to_bytes(8, "little", signed=False)
            + len(x).to_bytes(4, "little", signed=False)
            + signs
        )
        return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "little") % self.buckets

    @staticmethod
    def _unit(x):
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0 or not np.all(np.isfinite(x)):
            raise ValueError("Engram vectors must be non-empty and finite")
        return x / (np.linalg.norm(x) + 1e-8)

    def put(self, pattern, vector, payload, score=1.0, key=None):
        v = self._unit(vector)
        b = self._bucket(v)
        if key is None:
            blob = (
                json.dumps(payload, sort_keys=True, default=str).encode()
                + v.tobytes()
                + str(pattern).encode()
            )
            key = hashlib.sha1(blob).hexdigest()[:16]
        entry = EngramEntry(str(key), str(pattern), v, dict(payload), float(score), 0)
        self._by_bucket[b].append(entry)
        self._order.append((b, entry.key))
        while len(self._order) > self.capacity:
            ob, ok = self._order.popleft()
            self._by_bucket[ob] = [e for e in self._by_bucket[ob] if e.key != ok]
        return entry

    def query(self, vector, k=5, pattern=None):
        v = self._unit(vector)
        b = self._bucket(v)
        candidates = [e for e in self._by_bucket.get(b, []) if e.vector.shape == v.shape]
        if pattern is not None:
            candidates = [e for e in candidates if e.pattern == pattern]
        scored = []
        for e in candidates:
            sim = float(v @ e.vector)
            scored.append((sim * 0.8 + e.score * 0.2, e))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, e in scored[: int(k)]:
            e.uses += 1
            out.append((float(score), e))
        return out

    def __len__(self):
        return len(self._order)
