from __future__ import annotations
import numpy as np
from awa.experimental.connectome.graph import CSRGraph
from awa.experimental.connectome.motifs import directed_degree_preserving_rewire, _edge_arrays


def permute_weights(graph: CSRGraph, seed: int=0) -> CSRGraph:
    rng=np.random.default_rng(seed); w=graph.weight.copy(); rng.shuffle(w)
    return CSRGraph(graph.row_ptr.copy(),graph.dst.copy(),w,graph.node_ids.copy())


def group_preserving_rewire(graph: CSRGraph, source_groups, target_groups=None, swaps: int | None=None,
                            seed: int=0, allow_self: bool=False) -> CSRGraph:
    """Degree-preserving swaps restricted to equal source and target group classes."""
    src,dst,w=_edge_arrays(graph); sg=np.asarray(source_groups); tg=sg if target_groups is None else np.asarray(target_groups)
    if len(sg)!=graph.nodes or len(tg)!=graph.nodes: raise ValueError("group arrays must match node count")
    rng=np.random.default_rng(seed); edges={(int(a),int(b)) for a,b in zip(src,dst)}
    target=int(swaps if swaps is not None else min(len(src)*2,250000)); accepted=0; attempts=0
    while accepted<target and attempts<max(100,target*40):
        attempts+=1; i,j=rng.integers(0,len(src),size=2)
        if i==j:continue
        a,b=int(src[i]),int(dst[i]); c,d=int(src[j]),int(dst[j])
        if sg[a]!=sg[c] or tg[b]!=tg[d]:continue
        if a==c or b==d:continue
        e1,e2=(a,d),(c,b)
        if (not allow_self) and (a==d or c==b):continue
        if e1 in edges or e2 in edges:continue
        edges.remove((a,b)); edges.remove((c,d)); edges.add(e1); edges.add(e2); dst[i],dst[j]=d,b; accepted+=1
    return CSRGraph(graph.row_ptr.copy(),dst.astype(np.int32),w.copy(),graph.node_ids.copy())
