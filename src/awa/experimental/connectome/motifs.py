from __future__ import annotations
import numpy as np
from awa.experimental.connectome.graph import CSRGraph


def out_degrees(graph: CSRGraph) -> np.ndarray:
    return np.diff(graph.row_ptr)


def in_degrees(graph: CSRGraph) -> np.ndarray:
    return np.bincount(graph.dst, minlength=graph.nodes)


def graph_summary(graph: CSRGraph) -> dict:
    outd, ind = out_degrees(graph), in_degrees(graph)
    return {
        "nodes": int(graph.nodes), "edges": int(graph.edges),
        "mean_out_degree": float(outd.mean()) if len(outd) else 0.0,
        "max_out_degree": int(outd.max()) if len(outd) else 0,
        "max_in_degree": int(ind.max()) if len(ind) else 0,
        "sparsity": float(graph.edges / max(1, graph.nodes * graph.nodes)),
    }


def _edge_arrays(graph: CSRGraph):
    src = np.repeat(np.arange(graph.nodes, dtype=np.int32), np.diff(graph.row_ptr))
    return src, graph.dst.copy(), graph.weight.copy()


def directed_degree_preserving_rewire(graph: CSRGraph, swaps: int | None = None, seed: int = 0, allow_self=False) -> CSRGraph:
    """Directed double-edge swaps preserving exact in- and out-degree sequences."""
    rng = np.random.default_rng(seed)
    src, dst, weight = _edge_arrays(graph)
    if len(src) < 2: return CSRGraph(graph.row_ptr.copy(), dst, weight, graph.node_ids.copy())
    target_swaps = swaps if swaps is not None else min(len(src) * 2, 1_000_000)
    edges = {(int(s), int(d)) for s, d in zip(src, dst)}
    accepted = 0; attempts = 0; max_attempts = max(target_swaps * 20, 100)
    while accepted < target_swaps and attempts < max_attempts:
        attempts += 1
        i, j = rng.integers(0, len(src), size=2)
        if i == j: continue
        a,b = int(src[i]), int(dst[i]); c,d = int(src[j]), int(dst[j])
        if a == c or b == d: continue
        e1, e2 = (a,d), (c,b)
        if (not allow_self) and (a == d or c == b): continue
        if e1 in edges or e2 in edges: continue
        edges.remove((a,b)); edges.remove((c,d)); edges.add(e1); edges.add(e2)
        dst[i], dst[j] = d, b
        accepted += 1
    # Source ordering never changes, so CSR row_ptr remains valid and weights retain source-edge multiset.
    return CSRGraph(graph.row_ptr.copy(), dst.astype(np.int32), weight, graph.node_ids.copy())


def degree_preserving_stub(graph: CSRGraph, seed: int = 0) -> CSRGraph:
    """Backward-compatible alias now using strict directed degree-preserving rewiring."""
    return directed_degree_preserving_rewire(graph, seed=seed)
