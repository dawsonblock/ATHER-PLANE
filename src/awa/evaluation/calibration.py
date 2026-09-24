from __future__ import annotations
import numpy as np


def expected_calibration_error(confidence, correct, bins: int = 10) -> float:
    confidence = np.asarray(confidence, dtype=np.float64).reshape(-1)
    correct = np.asarray(correct, dtype=np.float64).reshape(-1)
    if len(confidence) != len(correct):
        raise ValueError("confidence and correct must have same length")
    if len(confidence) == 0:
        return 0.0
    confidence = np.clip(confidence, 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence); ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (confidence >= lo) & (confidence < hi if i < bins-1 else confidence <= hi)
        if not mask.any(): continue
        ece += (mask.sum()/total) * abs(confidence[mask].mean() - correct[mask].mean())
    return float(ece)


def brier_score(probability, outcome) -> float:
    p = np.asarray(probability, dtype=np.float64)
    y = np.asarray(outcome, dtype=np.float64)
    return float(np.mean((p - y) ** 2))
