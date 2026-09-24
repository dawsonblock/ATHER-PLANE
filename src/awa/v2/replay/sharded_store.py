from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import json, hashlib
import numpy as np

@dataclass(frozen=True)
class ReplayShard:
    file: str
    transitions: int
    sha256: str
    keys: tuple[str,...]

class ShardedReplayStore:
    """Append-only NPZ replay shards with deterministic index and integrity hashes."""
    def __init__(self, root: str|Path, shard_size: int=100_000):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True); self.shard_size=max(1,int(shard_size))
        self.index_path=self.root/'index.json'; self.shards=[]
        if self.index_path.exists():
            raw=json.loads(self.index_path.read_text())
            self.shards=[ReplayShard(x['file'],int(x['transitions']),x['sha256'],tuple(x['keys'])) for x in raw.get('shards',[])]
    @staticmethod
    def _sha(path):
        h=hashlib.sha256();
        with Path(path).open('rb') as f:
            for c in iter(lambda:f.read(1<<20),b''): h.update(c)
        return h.hexdigest()
    def _write_index(self):
        payload={'format':'awa-v2.11-sharded-replay-v1','transitions':len(self),'shards':[asdict(x) for x in self.shards]}
        tmp=self.index_path.with_suffix('.tmp'); tmp.write_text(json.dumps(payload,indent=2,sort_keys=True)); tmp.replace(self.index_path)
    def append(self, arrays: dict[str,np.ndarray]):
        arrays={k:np.asarray(v) for k,v in arrays.items()}
        if not arrays: raise ValueError('arrays required')
        n=len(next(iter(arrays.values())))
        if n==0: return []
        if any(len(v)!=n for v in arrays.values()): raise ValueError('all arrays must have same first dimension')
        keys=tuple(sorted(arrays))
        if self.shards and keys != self.shards[0].keys:
            raise ValueError('replay shard schema mismatch; all shards must have identical keys')
        written=[]
        for start in range(0,n,self.shard_size):
            stop=min(n,start+self.shard_size); idx=len(self.shards); name=f'shard-{idx:06d}.npz'; path=self.root/name
            np.savez_compressed(path,**{k:v[start:stop] for k,v in arrays.items()})
            shard=ReplayShard(name,stop-start,self._sha(path),keys)
            self.shards.append(shard); written.append(shard)
        self._write_index(); return written
    def verify(self):
        bad=[]
        for s in self.shards:
            p=self.root/s.file
            if not p.exists() or self._sha(p)!=s.sha256: bad.append(s.file)
        return bad
    def iter_shards(self):
        for s in self.shards:
            with np.load(self.root/s.file,allow_pickle=False) as z:
                yield {k:np.asarray(z[k]) for k in z.files}

    def materialize(self, path: str|Path, *, max_transitions: int|None=None, preserve_first: int|None=None) -> Path:
        """Materialize append-only shards into one chain-preserving NPZ dataset.

        Shard order is preserved exactly. This is important for temporal learners: a
        terminal row remains between collection batches, so independent episodes
        cannot be spliced into one sequence.
        """
        if not self.shards:
            raise ValueError('cannot materialize an empty replay store')
        limit=None if max_transitions is None else int(max_transitions)
        if limit is not None and limit <= 0:
            raise ValueError('max_transitions must be > 0')
        if preserve_first is not None:
            head=int(preserve_first)
            if limit is None or not 0 < head < limit:
                raise ValueError('preserve_first requires 0 < preserve_first < max_transitions')
            if len(self) < limit:
                raise ValueError('replay does not contain max_transitions rows')
            # Collection can overshoot a stage target by whole episodes. Keep
            # the initial wiring experience and use the newest collected rows
            # for the remaining budget, so an overshoot cannot hide all recent
            # successful episodes from the learner.
            tail_start=len(self)-(limit-head)
            parts={k:[] for k in self.shards[0].keys}
            offset=0
            for shard in self.shards:
                end=offset+shard.transitions
                spans=((max(offset,0),min(end,head)),
                       (max(offset,tail_start),min(end,len(self))))
                if any(start<stop for start,stop in spans):
                    with np.load(self.root/shard.file,allow_pickle=False) as z:
                        for start,stop in spans:
                            if start<stop:
                                for key in parts:
                                    parts[key].append(np.asarray(z[key][start-offset:stop-offset]))
                offset=end
            merged={k:np.concatenate(v,axis=0) for k,v in parts.items()}
            if len(next(iter(merged.values()))) != limit:
                raise RuntimeError('stage materialization violated transition budget')
            # The first region must end at a true terminal; otherwise the
            # temporal model could run through a splice between scenarios.
            if 'dones' in merged and not bool(merged['dones'][head-1]):
                raise ValueError('preserved replay prefix ends inside an episode')
            target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.stem+'.tmp.npz')
            np.savez_compressed(tmp,**merged); tmp.replace(target)
            return target
        parts={k:[] for k in self.shards[0].keys}; remaining=limit
        for arrays in self.iter_shards():
            take=len(next(iter(arrays.values())))
            if remaining is not None:
                if remaining <= 0: break
                take=min(take,remaining)
            for k in parts: parts[k].append(np.asarray(arrays[k])[:take])
            if remaining is not None: remaining-=take
        merged={k:np.concatenate(v,axis=0) for k,v in parts.items() if v}
        target=Path(path); target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name(target.stem+'.tmp.npz')
        np.savez_compressed(tmp,**merged); tmp.replace(target)
        return target

    def __len__(self): return sum(s.transitions for s in self.shards)
