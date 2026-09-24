from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EpisodeRecord:
    embedding: np.ndarray
    goal_embedding: np.ndarray
    action: int
    reward: float
    outcome: str
    metadata: dict


class EpisodicMemory:
    """Dependency-free cosine retrieval. Swap with FAISS for large stores."""

    def __init__(self, max_records: int = 100_000):
        self.max_records = max_records
        self.records: list[EpisodeRecord] = []

    def add(self, record: EpisodeRecord):
        self.records.append(record)
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def query(self, embedding: np.ndarray, goal_embedding: np.ndarray | None = None, top_k: int = 5):
        if not self.records:
            return []
        q = embedding / (np.linalg.norm(embedding) + 1e-8)
        scored = []
        for r in self.records:
            e = r.embedding / (np.linalg.norm(r.embedding) + 1e-8)
            score = float(np.dot(q, e))
            if goal_embedding is not None:
                qg = goal_embedding / (np.linalg.norm(goal_embedding) + 1e-8)
                rg = r.goal_embedding / (np.linalg.norm(r.goal_embedding) + 1e-8)
                score = 0.75 * score + 0.25 * float(np.dot(qg, rg))
            scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[:top_k]
