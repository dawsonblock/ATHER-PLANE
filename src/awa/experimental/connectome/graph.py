from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import csv
import numpy as np


@dataclass
class CSRGraph:
    row_ptr: np.ndarray
    dst: np.ndarray
    weight: np.ndarray
    node_ids: np.ndarray

    @property
    def nodes(self):
        return len(self.node_ids)

    @property
    def edges(self):
        return len(self.dst)


def compile_edge_csv(path: str | Path, source_col="source", target_col="target", weight_col="weight") -> CSRGraph:
    """
    Compile a generic source,target,weight CSV to CSR.
    MaleCNS adapters should map official table columns to this generic form.
    """
    edges = []
    nodes = set()
    with open(path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            s = int(row[source_col]); t = int(row[target_col]); w = float(row[weight_col])
            edges.append((s, t, w))
            nodes.add(s); nodes.add(t)

    node_ids = np.array(sorted(nodes), dtype=np.int64)
    index = {int(v): i for i, v in enumerate(node_ids)}
    rows = [[] for _ in range(len(node_ids))]
    for s, t, w in edges:
        rows[index[s]].append((index[t], w))

    row_ptr = [0]
    dst = []
    weight = []
    for row in rows:
        for t, w in row:
            dst.append(t); weight.append(w)
        row_ptr.append(len(dst))

    return CSRGraph(
        np.asarray(row_ptr, dtype=np.int64),
        np.asarray(dst, dtype=np.int32),
        np.asarray(weight, dtype=np.float32),
        node_ids,
    )
