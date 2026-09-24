from __future__ import annotations
import torch
import torch.nn.functional as F
from awa.state.belief import Belief


class CEMPlanner:
    """CEM planner for discrete actions using imagined latent rollouts."""
    action_type = "discrete"
    def __init__(self, world_model, action_dim, horizon=8, candidates=128, elites=16, iterations=3):
        self.world_model=world_model; self.action_dim=action_dim; self.horizon=horizon; self.candidates=candidates; self.elites=elites; self.iterations=iterations

    @torch.no_grad()
    def plan_with_metadata(self, belief: Belief, task_context: torch.Tensor | None = None, horizon: int | None = None, candidates: int | None = None):
        device=belief.deterministic.device
        if belief.deterministic.shape[0]!=1: raise ValueError("CEMPlanner expects batch size 1")
        H=int(self.horizon if horizon is None else horizon); C=int(self.candidates if candidates is None else candidates)
        if H<=0 or C<=0: raise ValueError("horizon and candidates must be positive")
        elites=min(self.elites,C); probs=torch.full((H,self.action_dim),1.0/self.action_dim,device=device); final_scores=None
        for _ in range(self.iterations):
            acts_idx=torch.multinomial(probs,C,replacement=True).transpose(0,1); scores=torch.zeros(C,device=device)
            h=belief.deterministic.expand(C,-1).clone(); s=belief.stochastic.expand(C,-1).clone(); slow=None if belief.slow is None else belief.slow.expand(C,-1).clone(); b=Belief(h,s,slow)
            ctx=None if task_context is None else task_context.expand(C,-1); alive=torch.ones(C,device=device)
            for t in range(H):
                a=F.one_hot(acts_idx[:,t],self.action_dim).float(); b,reward,cont,value,_,_=self.world_model.imagine(b,a,step_index=t,task_context=ctx)
                continuation=torch.sigmoid(cont.squeeze(-1)); scores += alive*(reward.squeeze(-1)+0.05*value.squeeze(-1)); alive *= continuation
            elite_idx=scores.topk(elites).indices; elite_actions=acts_idx[elite_idx]
            probs=torch.stack([(torch.bincount(elite_actions[:,t],minlength=self.action_dim).float()+1.0) for t in range(H)]); probs=probs/probs.sum(dim=-1,keepdim=True); final_scores=scores
        action_idx=probs[0].argmax(); meta={"best_score":float(final_scores.max().item()),"mean_score":float(final_scores.mean().item()),"horizon":H,"candidates":C,"iterations":self.iterations}
        return F.one_hot(action_idx,self.action_dim).float().unsqueeze(0),int(action_idx.item()),probs,meta

    @torch.no_grad()
    def plan(self, belief: Belief, task_context: torch.Tensor | None = None, horizon: int | None = None, candidates: int | None = None):
        a,idx,p,_=self.plan_with_metadata(belief,task_context,horizon,candidates); return a,idx,p


class ContinuousCEMPlanner:
    """CEM trajectory optimizer for bounded continuous actions."""
    action_type = "continuous"
    def __init__(self, world_model, action_dim, low, high, horizon=8, candidates=256, elites=32, iterations=4,
                 min_std: float = 0.03, momentum: float = 0.10):
        self.world_model=world_model; self.action_dim=int(action_dim); self.horizon=int(horizon); self.candidates=int(candidates); self.elites=int(elites); self.iterations=int(iterations)
        self.low=torch.as_tensor(low,dtype=torch.float32).reshape(-1); self.high=torch.as_tensor(high,dtype=torch.float32).reshape(-1)
        if self.low.numel()!=self.action_dim or self.high.numel()!=self.action_dim: raise ValueError("continuous planner bounds mismatch")
        self.min_std=float(min_std); self.momentum=float(momentum)

    @torch.no_grad()
    def plan_with_metadata(self, belief: Belief, task_context: torch.Tensor | None = None, horizon: int | None = None, candidates: int | None = None):
        device=belief.deterministic.device
        if belief.deterministic.shape[0]!=1: raise ValueError("ContinuousCEMPlanner expects batch size 1")
        H=int(self.horizon if horizon is None else horizon); C=int(self.candidates if candidates is None else candidates); E=min(self.elites,C)
        low=self.low.to(device); high=self.high.to(device); center=(low+high)*0.5; scale=(high-low)*0.5
        mean=center.expand(H,-1).clone(); std=scale.expand(H,-1).clone().clamp_min(self.min_std); final_scores=None
        for _ in range(self.iterations):
            noise=torch.randn(C,H,self.action_dim,device=device); actions=(mean.unsqueeze(0)+std.unsqueeze(0)*noise).clamp(low,high); scores=torch.zeros(C,device=device)
            h=belief.deterministic.expand(C,-1).clone(); s=belief.stochastic.expand(C,-1).clone(); slow=None if belief.slow is None else belief.slow.expand(C,-1).clone(); b=Belief(h,s,slow)
            ctx=None if task_context is None else task_context.expand(C,-1); alive=torch.ones(C,device=device)
            for t in range(H):
                b,reward,cont,value,_,_=self.world_model.imagine(b,actions[:,t],step_index=t,task_context=ctx)
                continuation=torch.sigmoid(cont.squeeze(-1)); scores += alive*(reward.squeeze(-1)+0.05*value.squeeze(-1)); alive *= continuation
            elite=actions[scores.topk(E).indices]; new_mean=elite.mean(0); new_std=elite.std(0,unbiased=False).clamp_min(self.min_std)
            mean=self.momentum*mean+(1-self.momentum)*new_mean; std=self.momentum*std+(1-self.momentum)*new_std; final_scores=scores
        action=mean[0].clamp(low,high).unsqueeze(0); meta={"best_score":float(final_scores.max().item()),"mean_score":float(final_scores.mean().item()),"horizon":H,"candidates":C,"iterations":self.iterations,"mean_std":float(std.mean().item())}
        return action,action.squeeze(0).cpu().numpy(),{"mean":mean,"std":std},meta

    @torch.no_grad()
    def plan(self, belief: Belief, task_context: torch.Tensor | None = None, horizon: int | None = None, candidates: int | None = None):
        a,env_action,state,_=self.plan_with_metadata(belief,task_context,horizon,candidates); return a,env_action,state
