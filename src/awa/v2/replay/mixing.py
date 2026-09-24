from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Mapping
import math
import numpy as np


@dataclass(frozen=True)
class ReplayMixtureReport:
    requested: int
    produced: int
    counts: dict[str, int]
    source_ids: dict[str, int]

    def to_dict(self):
        return asdict(self)


def _normalize_ratios(names: list[str], ratios: Mapping[str, float]) -> np.ndarray:
    vals = np.asarray([float(ratios.get(n, 0.0)) for n in names], dtype=np.float64)
    if np.any(~np.isfinite(vals)) or np.any(vals < 0):
        raise ValueError("replay ratios must be finite and non-negative")
    total = float(vals.sum())
    if total <= 0:
        raise ValueError("at least one replay ratio must be > 0")
    return vals / total


def _allocate_without_replacement(capacities: np.ndarray, probabilities: np.ndarray, target: int) -> np.ndarray:
    target = min(int(target), int(capacities.sum()))
    if target <= 0:
        return np.zeros_like(capacities)
    alloc = np.zeros_like(capacities)
    remaining = target
    active = capacities > 0
    # Iteratively allocate the remaining budget using largest-remainder rounding,
    # redistributing quota from sources that have insufficient capacity.
    while remaining > 0 and bool(active.any()):
        p = probabilities * active
        if p.sum() <= 0:
            p = active.astype(np.float64)
        p = p / p.sum()
        raw = p * remaining
        base = np.floor(raw).astype(np.int64)
        room = capacities - alloc
        base = np.minimum(base, room)
        alloc += base
        used = int(base.sum())
        remaining -= used
        room = capacities - alloc
        active = room > 0
        if remaining <= 0 or not bool(active.any()):
            break
        # Hand out residual items in descending fractional-priority order.
        frac = raw - np.floor(raw)
        order = np.argsort(-(frac + 1e-12 * probabilities))
        progressed = False
        for idx in order:
            if remaining <= 0:
                break
            if room[idx] <= 0:
                continue
            alloc[idx] += 1
            room[idx] -= 1
            remaining -= 1
            progressed = True
        if not progressed:
            break
        active = (capacities - alloc) > 0
    return alloc


def mix_replay_arrays(
    sources: Mapping[str, Mapping[str, np.ndarray]],
    ratios: Mapping[str, float],
    target_count: int,
    *,
    seed: int = 0,
    allow_replacement: bool = False,
) -> tuple[dict[str, np.ndarray], ReplayMixtureReport]:
    """Deterministically mix named replay sources while preserving episode boundaries.

    All sources must expose the same keys and first-dimension lengths. If an
    ``episode_ids`` field exists, ids are remapped so episodes from different
    sources cannot accidentally merge after concatenation.
    """
    if int(target_count) <= 0:
        raise ValueError("target_count must be > 0")
    if not sources:
        raise ValueError("at least one replay source is required")
    names = sorted(str(n) for n in sources)
    arrays_by_name: dict[str, dict[str, np.ndarray]] = {}
    schema = None
    capacities = []
    for name in names:
        arr = {k: np.asarray(v) for k, v in sources[name].items()}
        if not arr:
            raise ValueError(f"replay source {name!r} is empty")
        keys = tuple(sorted(arr))
        if schema is None:
            schema = keys
        elif keys != schema:
            raise ValueError("all replay sources must have identical array keys")
        n = len(next(iter(arr.values())))
        if any(len(v) != n for v in arr.values()):
            raise ValueError(f"replay source {name!r} contains mismatched lengths")
        if n <= 0:
            raise ValueError(f"replay source {name!r} contains zero transitions")
        arrays_by_name[name] = arr
        capacities.append(n)
    probabilities = _normalize_ratios(names, ratios)
    capacities_arr = np.asarray(capacities, dtype=np.int64)
    target = int(target_count)
    if allow_replacement:
        raw = probabilities * target
        alloc = np.floor(raw).astype(np.int64)
        for idx in np.argsort(-(raw - alloc))[: target - int(alloc.sum())]:
            alloc[idx] += 1
    else:
        alloc = _allocate_without_replacement(capacities_arr, probabilities, target)
    rng = np.random.default_rng(int(seed))
    selected: dict[str, np.ndarray] = {}
    pieces: dict[str, list[np.ndarray]] = {k: [] for k in schema or ()}
    source_rows = []
    episode_offset = 0
    source_ids = {name: i for i, name in enumerate(names)}
    counts: dict[str, int] = {}
    for i, name in enumerate(names):
        count = int(alloc[i])
        counts[name] = count
        if count <= 0:
            continue
        n = capacities[i]
        ids = rng.choice(n, size=count, replace=bool(allow_replacement and count > n))
        ids = np.asarray(ids, dtype=np.int64)
        selected[name] = ids
        arr = arrays_by_name[name]
        for key in schema or ():
            part = np.asarray(arr[key][ids])
            if key == "episode_ids":
                # Compact each source's selected episode ids into a globally unique range.
                unique = []
                lookup: dict[int, int] = {}
                remapped = np.empty(len(part), dtype=np.int64)
                for j, value in enumerate(np.asarray(part).reshape(-1)):
                    iv = int(value)
                    if iv not in lookup:
                        lookup[iv] = episode_offset + len(unique)
                        unique.append(iv)
                    remapped[j] = lookup[iv]
                episode_offset += len(unique)
                part = remapped
            pieces[key].append(part)
        source_rows.append(np.full(count, source_ids[name], dtype=np.int16))
    produced = int(sum(counts.values()))
    if produced <= 0:
        raise ValueError("replay mixture produced no transitions")
    mixed = {k: np.concatenate(v, axis=0) for k, v in pieces.items() if v}
    mixed["replay_source"] = np.concatenate(source_rows, axis=0)
    order = rng.permutation(produced)
    # Do not shuffle chain-sensitive replay tables when episode ids are present.
    # Sequence learners rely on contiguous next_observation -> observation chains.
    if "episode_ids" not in mixed:
        mixed = {k: v[order] for k, v in mixed.items()}
    report = ReplayMixtureReport(int(target_count), produced, counts, source_ids)
    return mixed, report
