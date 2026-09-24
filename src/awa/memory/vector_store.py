from __future__ import annotations
from dataclasses import dataclass
import time
import numpy as np


@dataclass
class SearchHit:
    score: float
    index: int
    metadata: dict


class VectorMemory:
    """FAISS-backed memory with NumPy fallback, deduplication and bounded retention."""
    def __init__(self, dim: int, use_faiss: bool=True, max_items: int | None=None, dedup_threshold: float=0.999):
        self.dim=int(dim); self.max_items=max_items; self.dedup_threshold=float(dedup_threshold)
        self.metadata=[]; self._vectors=[]; self._faiss_mod=None; self._faiss=None
        if use_faiss:
            try:
                import faiss
                self._faiss_mod=faiss; self._faiss=faiss.IndexFlatIP(self.dim)
            except ImportError: pass

    @staticmethod
    def _norm(x):
        x=np.asarray(x,dtype=np.float32).reshape(-1); n=np.linalg.norm(x); return x/max(float(n),1e-8)

    def _rebuild(self):
        if self._faiss_mod is not None:
            self._faiss=self._faiss_mod.IndexFlatIP(self.dim)
            if self._vectors: self._faiss.add(np.stack(self._vectors,axis=0).astype(np.float32))

    def _evict_if_needed(self):
        if self.max_items is None: return
        while len(self.metadata)>self.max_items:
            # Evict lowest importance; oldest breaks ties.
            i=min(range(len(self.metadata)),key=lambda j:(float(self.metadata[j].get("importance",0.0)),float(self.metadata[j].get("timestamp",0.0))))
            self.metadata.pop(i); self._vectors.pop(i)
        self._rebuild()

    def add(self, vector, metadata: dict | None=None, deduplicate: bool=True) -> int:
        v=self._norm(vector)
        if v.size!=self.dim: raise ValueError(f"expected dim {self.dim}, got {v.size}")
        meta=dict(metadata or {}); meta.setdefault("timestamp",time.time()); meta.setdefault("importance",0.0)
        if deduplicate and self.metadata:
            hit=self.search(v,1)
            if hit and hit[0].score>=self.dedup_threshold:
                idx=hit[0].index
                # Merge useful recency/importance instead of storing a duplicate.
                self.metadata[idx].update(meta)
                self.metadata[idx]["importance"]=max(float(self.metadata[idx].get("importance",0)),float(meta.get("importance",0)))
                return idx
        idx=len(self.metadata); self.metadata.append(meta); self._vectors.append(v)
        if self._faiss is not None: self._faiss.add(v[None,:])
        self._evict_if_needed(); return min(idx,len(self.metadata)-1)

    def search(self, query, top_k: int=5, predicate=None) -> list[SearchHit]:
        if not self.metadata: return []
        q=self._norm(query); k=min(max(int(top_k),1),len(self.metadata))
        # Predicate filtering uses exact NumPy path so IDs stay stable.
        if predicate is not None:
            candidates=[i for i,m in enumerate(self.metadata) if predicate(m)]
            if not candidates:return []
            mat=np.stack([self._vectors[i] for i in candidates]); scores=mat@q
            order=np.argsort(-scores)[:min(k,len(candidates))]
            return [SearchHit(float(scores[o]),int(candidates[o]),self.metadata[candidates[o]]) for o in order]
        if self._faiss is not None:
            scores,ids=self._faiss.search(q[None,:],k)
            return [SearchHit(float(s),int(i),self.metadata[int(i)]) for s,i in zip(scores[0],ids[0]) if i>=0]
        mat=np.stack(self._vectors); scores=mat@q; ids=np.argsort(-scores)[:k]
        return [SearchHit(float(scores[i]),int(i),self.metadata[int(i)]) for i in ids]

    def state_dict(self):
        return {"dim":self.dim,"max_items":self.max_items,"dedup_threshold":self.dedup_threshold,
                "vectors":[v.copy() for v in self._vectors],"metadata":[dict(m) for m in self.metadata]}

    def load_state_dict(self,state):
        if int(state["dim"])!=self.dim: raise ValueError("dimension mismatch")
        self.max_items=state.get("max_items"); self.dedup_threshold=float(state.get("dedup_threshold",self.dedup_threshold))
        self._vectors=[np.asarray(v,dtype=np.float32) for v in state["vectors"]]; self.metadata=[dict(m) for m in state["metadata"]]
        self._rebuild(); return self
