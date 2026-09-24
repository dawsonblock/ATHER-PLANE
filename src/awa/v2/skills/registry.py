from __future__ import annotations
from dataclasses import dataclass, field, asdict

@dataclass
class SkillSpec:
    name:str
    preconditions:frozenset[str]=field(default_factory=frozenset)
    effects:frozenset[str]=field(default_factory=frozenset)
    success_rate:float=0.0
    confidence:float=0.0
    uses:int=0
    source:str='distilled'
    metadata:dict=field(default_factory=dict)
    def to_dict(self):
        d=asdict(self); d['preconditions']=sorted(self.preconditions); d['effects']=sorted(self.effects); return d

class SkillRegistry:
    def __init__(self): self._skills={}
    def register(self,skill:SkillSpec,min_success=.7,min_confidence=.5):
        if skill.success_rate<min_success or skill.confidence<min_confidence: return False
        self._skills[skill.name]=skill; return True
    def get(self,name): return self._skills.get(name)
    def applicable(self,facts:set[str]): return [s for s in self._skills.values() if s.preconditions.issubset(facts)]
    def __len__(self): return len(self._skills)
