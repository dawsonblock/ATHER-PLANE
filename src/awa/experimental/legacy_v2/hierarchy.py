from __future__ import annotations
from dataclasses import dataclass, field
import heapq

@dataclass(frozen=True)
class Subgoal:
    name: str
    preconditions: frozenset[str] = field(default_factory=frozenset)
    effects: frozenset[str] = field(default_factory=frozenset)
    cost: float = 1.0
    reachability: float = 1.0
    risk: float = 0.0

class SubgoalPlanner:
    """Small symbolic/skill-level uniform-cost planner with reachability/risk penalties."""
    def __init__(self,subgoals:list[Subgoal]): self.subgoals=list(subgoals)
    def plan(self,current:set[str],goal:set[str],max_expansions=10_000):
        start=frozenset(current); target=set(goal); q=[(0.0,0,start,[])]; seen={start:0.0}; n=0; tie=0
        while q and n<max_expansions:
            cost,_,state,path=heapq.heappop(q); n+=1
            if target.issubset(state): return path,cost
            for sg in self.subgoals:
                if not sg.preconditions.issubset(state): continue
                ns=frozenset(set(state)|set(sg.effects)); step=sg.cost + 2.0*(1.0-sg.reachability)+3.0*sg.risk; nc=cost+step
                if nc < seen.get(ns,float('inf')):
                    seen[ns]=nc; tie+=1; heapq.heappush(q,(nc,tie,ns,path+[sg]))
        return None,float('inf')
