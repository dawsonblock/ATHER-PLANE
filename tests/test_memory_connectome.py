import numpy as np
from awa.memory.episodic import EpisodicMemory, EpisodeRecord
from awa.connectome.graph import CSRGraph
from awa.connectome.motifs import graph_summary, degree_preserving_stub, in_degrees, out_degrees


def test_memory_query():
    mem = EpisodicMemory()
    mem.add(EpisodeRecord(np.array([1.,0.]), np.array([1.,0.]), 0, 1.0, "ok", {}))
    mem.add(EpisodeRecord(np.array([0.,1.]), np.array([0.,1.]), 1, 0.0, "x", {}))
    r = mem.query(np.array([1.,0.]), top_k=1)
    assert r[0][1].action == 0


def test_graph_summary():
    g = CSRGraph(
        np.array([0,2,3], dtype=np.int64),
        np.array([0,1,0], dtype=np.int32),
        np.ones(3, dtype=np.float32),
        np.array([10,20], dtype=np.int64),
    )
    s = graph_summary(g)
    assert s["nodes"] == 2
    assert s["edges"] == 3
    r = degree_preserving_stub(g, seed=1)
    assert np.array_equal(np.diff(r.row_ptr), np.diff(g.row_ptr))



def test_strict_rewire_preserves_in_out_degree():
    # Directed 4-node cycle plus diagonals: enough valid swap opportunities.
    src = np.array([0,0,1,1,2,2,3,3], dtype=np.int32)
    dst = np.array([1,2,2,3,3,0,0,1], dtype=np.int32)
    order = np.argsort(src)
    src, dst = src[order], dst[order]
    counts = np.bincount(src, minlength=4)
    row_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    g = CSRGraph(row_ptr, dst, np.ones(len(dst), np.float32), np.arange(4, dtype=np.int64))
    r = degree_preserving_stub(g, seed=3)
    assert np.array_equal(out_degrees(g), out_degrees(r))
    assert np.array_equal(in_degrees(g), in_degrees(r))
