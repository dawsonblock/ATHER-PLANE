from __future__ import annotations
from dataclasses import dataclass
import copy

@dataclass(frozen=True)
class ContextCheckpoint:
    step:int
    state:object

class BoundedContextReplay:
    """Checkpoint + bounded transition replay for reconstructing temporal state.

    This is a generic storage/compute trade-off inspired by bounded replay: store a
    periodic state snapshot and reconstruct intermediate state from only the recent
    transitions instead of persisting every recurrent/cache state.
    """
    def __init__(self,checkpoint_interval=128,max_transitions=4096):
        self.checkpoint_interval=int(checkpoint_interval); self.max_transitions=int(max_transitions)
        if self.checkpoint_interval<1: raise ValueError('checkpoint_interval must be >= 1')
        self.checkpoints={}; self.transitions={}
    def record(self,step:int,state,transition=None):
        step=int(step)
        if step%self.checkpoint_interval==0: self.checkpoints[step]=copy.deepcopy(state)
        if transition is not None: self.transitions[step]=copy.deepcopy(transition)
        floor=max(0,step-self.max_transitions)
        self.transitions={k:v for k,v in self.transitions.items() if k>=floor}
        # keep the nearest older checkpoint plus recent ones
        cps=sorted(self.checkpoints)
        older=[k for k in cps if k<=floor]
        keep_old=max(older) if older else None
        self.checkpoints={k:v for k,v in self.checkpoints.items() if k>=floor or k==keep_old}
    def nearest_checkpoint(self,target_step:int):
        eligible=[k for k in self.checkpoints if k<=int(target_step)]
        if not eligible: raise KeyError('no checkpoint at or before target step')
        k=max(eligible); return ContextCheckpoint(k,copy.deepcopy(self.checkpoints[k]))
    def reconstruct(self,target_step:int,apply_transition):
        cp=self.nearest_checkpoint(target_step); state=cp.state
        for step in range(cp.step,int(target_step)):
            if step not in self.transitions: raise KeyError(f'missing transition at step {step}')
            state=apply_transition(state,copy.deepcopy(self.transitions[step]))
        return state
