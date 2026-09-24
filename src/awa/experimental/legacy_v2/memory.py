from __future__ import annotations
from collections import deque
from dataclasses import dataclass
import numpy as np

class LongContextBuffer:
    def __init__(self,max_events:int=2048): self.events=deque(maxlen=max_events)
    def append(self,latent,action=None,reward=None,event=None):
        self.events.append({'latent':np.asarray(latent,dtype=np.float32),'action':None if action is None else np.asarray(action,dtype=np.float32),'reward':reward,'event':event})
    def recent_latents(self,n:int=128):
        xs=list(self.events)[-n:]
        return np.stack([x['latent'] for x in xs]) if xs else np.empty((0,0),dtype=np.float32)
    def __len__(self): return len(self.events)

@dataclass
class EpisodicItem:
    state: np.ndarray
    goal: np.ndarray
    outcome: float
    metadata: dict

class EpisodicStore:
    def __init__(self,capacity=100_000): self.capacity=int(capacity); self.items=[]
    def add(self,item:EpisodicItem):
        self.items.append(item)
        if len(self.items)>self.capacity: self.items.pop(0)
    def query(self,state,goal=None,k=5):
        if not self.items: return []
        s=np.asarray(state,dtype=np.float32); s=s/(np.linalg.norm(s)+1e-8); scored=[]
        for item in self.items:
            x=item.state/(np.linalg.norm(item.state)+1e-8); score=float(s@x)
            if goal is not None:
                g=np.asarray(goal,dtype=np.float32); g=g/(np.linalg.norm(g)+1e-8); ig=item.goal/(np.linalg.norm(item.goal)+1e-8); score=.75*score+.25*float(g@ig)
            scored.append((score,item))
        return sorted(scored,key=lambda z:z[0],reverse=True)[:k]
