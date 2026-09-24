from __future__ import annotations
import heapq

class SkillComposer:
    def __init__(self,registry): self.registry=registry
    def plan(self,current:set[str],goal:set[str],max_expansions=5000):
        start=frozenset(current); goal=set(goal); q=[(0,0,start,[])]; seen={start:0}; tie=0; n=0
        while q and n<max_expansions:
            cost,_,facts,path=heapq.heappop(q); n+=1
            if goal.issubset(facts): return path,cost
            for s in self.registry.applicable(set(facts)):
                nf=frozenset(set(facts)|set(s.effects)); nc=cost + 1.0/max(.05,s.success_rate)
                if nc>=seen.get(nf,float('inf')): continue
                seen[nf]=nc; tie+=1; heapq.heappush(q,(nc,tie,nf,path+[s.name]))
        return None,float('inf')
