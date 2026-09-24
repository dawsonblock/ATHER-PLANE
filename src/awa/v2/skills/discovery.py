from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np
from .registry import SkillSpec

@dataclass(frozen=True)
class TrajectorySignature:
    key:str
    start:np.ndarray
    end:np.ndarray
    mean_action:np.ndarray
    length:int
    success:bool

class SkillDiscovery:
    """Lightweight trajectory clustering by learned-latent displacement/action signature."""
    def __init__(self,round_decimals=1,min_examples=5): self.round_decimals=int(round_decimals); self.min_examples=int(min_examples); self.groups={}
    def signature(self,states,actions,success=True):
        s=np.asarray(states,dtype=np.float32); a=np.asarray(actions,dtype=np.float32)
        if len(s)<2 or len(a)<1: raise ValueError('trajectory too short')
        vec=np.concatenate([s[-1]-s[0],a.mean(0)]); q=np.round(vec,self.round_decimals); key=hashlib.sha1(q.tobytes()).hexdigest()[:12]
        return TrajectorySignature(key,s[0],s[-1],a.mean(0),len(a),bool(success))
    def observe(self,states,actions,success=True):
        sig=self.signature(states,actions,success); self.groups.setdefault(sig.key,[]).append(sig); return sig
    def candidates(self):
        out=[]
        for key,rows in self.groups.items():
            if len(rows)<self.min_examples: continue
            sr=sum(r.success for r in rows)/len(rows); conf=min(1.0,len(rows)/(2*self.min_examples))
            if sr<.6: continue
            out.append(SkillSpec(
                f'auto_{key}',
                success_rate=sr,
                confidence=conf,
                uses=len(rows),
                metadata={
                    'examples':len(rows),
                    'mean_length':float(np.mean([r.length for r in rows])),
                    'start_centroid':np.mean(np.stack([r.start for r in rows]),axis=0).astype(np.float32).tolist(),
                    'end_centroid':np.mean(np.stack([r.end for r in rows]),axis=0).astype(np.float32).tolist(),
                    'mean_action':np.mean(np.stack([r.mean_action for r in rows]),axis=0).astype(np.float32).tolist(),
                },
            ))
        return out
