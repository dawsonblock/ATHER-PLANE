from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import torch
from torch import nn

from .world import MultimodalWorldModel


@dataclass(frozen=True)
class ComputeProfile:
    kind: str
    total_parameters: int
    active_parameters_approx: int
    trunk_active_macs: int
    trunk_total_macs: int
    hidden: int
    experts: int = 1
    top_k: int = 1
    expert_hidden: int | None = None

    def to_dict(self):
        return asdict(self)


class SparseMoETrunk(nn.Module):
    """Token-wise sparse MLP expert trunk.

    Only selected experts execute for each row. The straight-through routing weight keeps
    top-1 routing trainable while preserving a hard sparse forward path.
    """
    def __init__(self, input_dim:int, output_dim:int, *, experts:int=4, top_k:int=1, expert_hidden:int|None=None):
        super().__init__()
        if experts < 2: raise ValueError("experts must be >= 2")
        if top_k < 1 or top_k > experts: raise ValueError("top_k must be in [1, experts]")
        self.input_dim=int(input_dim); self.output_dim=int(output_dim); self.expert_count=int(experts); self.top_k=int(top_k)
        self.expert_hidden=int(expert_hidden or output_dim)
        self.gate=nn.Linear(self.input_dim,self.expert_count)
        self.experts=nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.input_dim,self.expert_hidden),nn.SiLU(),
                nn.Linear(self.expert_hidden,self.output_dim),nn.SiLU(),
            ) for _ in range(self.expert_count)
        ])
        self.register_buffer("routing_counts",torch.zeros(self.expert_count,dtype=torch.long),persistent=False)

    def forward(self,x:torch.Tensor):
        logits=self.gate(x)
        probs=torch.softmax(logits,-1)
        selected_probs,selected_idx=torch.topk(probs,self.top_k,dim=-1)
        normalized=selected_probs/selected_probs.sum(-1,keepdim=True).clamp_min(1e-8)
        # Forward value == normalized. Extra term supplies gate gradient even for top_k=1.
        weights=normalized+(selected_probs-selected_probs.detach())
        out=torch.zeros(x.shape[0],self.output_dim,device=x.device,dtype=x.dtype)
        counts=torch.zeros(self.expert_count,device=x.device,dtype=torch.long)
        for slot in range(self.top_k):
            ids=selected_idx[:,slot]
            w=weights[:,slot]
            for expert_id,expert in enumerate(self.experts):
                mask=ids==expert_id
                if mask.any():
                    out[mask]+=expert(x[mask])*w[mask].unsqueeze(-1)
                    counts[expert_id]+=mask.sum()
        if self.training:
            self.routing_counts.add_(counts.detach().to(self.routing_counts.device))
        return out

    def reset_routing_counts(self):
        self.routing_counts.zero_()

    def routing_load(self):
        total=self.routing_counts.sum().clamp_min(1)
        return (self.routing_counts.float()/total.float()).cpu()

    def active_macs(self):
        # Linear MAC estimate, not hardware timing: gate + selected expert matmuls.
        gate=self.input_dim*self.expert_count
        expert=self.input_dim*self.expert_hidden+self.expert_hidden*self.output_dim
        return int(gate+self.top_k*expert)

    def total_macs(self):
        gate=self.input_dim*self.expert_count
        expert=self.input_dim*self.expert_hidden+self.expert_hidden*self.output_dim
        return int(gate+self.expert_count*expert)


class ModularWorldModel(MultimodalWorldModel):
    """Aether multimodal world model with a sparse small-MoE dynamics trunk."""
    def __init__(self,belief_dim:int,action_dim:int,hidden:int=256,components:int=5,horizons=(1,5,10,20,50),*,experts:int=4,top_k:int=1,expert_hidden:int|None=None):
        super().__init__(belief_dim,action_dim,hidden=hidden,components=components,horizons=horizons)
        self.experts=int(experts); self.top_k=int(top_k)
        self.trunk=SparseMoETrunk(belief_dim+action_dim,hidden,experts=experts,top_k=top_k,expert_hidden=expert_hidden)
        self.expert_hidden=self.trunk.expert_hidden

    def compute_profile(self):
        total=sum(p.numel() for p in self.parameters())
        expert_params=[sum(p.numel() for p in e.parameters()) for e in self.trunk.experts]
        all_experts=sum(expert_params)
        shared=total-all_experts
        active=shared+sum(sorted(expert_params,reverse=True)[:self.top_k])
        return ComputeProfile(
            "moe",int(total),int(active),self.trunk.active_macs(),self.trunk.total_macs(),
            int(self.mean.in_features),self.experts,self.top_k,self.expert_hidden,
        )


def monolithic_compute_profile(model:MultimodalWorldModel)->ComputeProfile:
    hidden=int(model.mean.in_features); inp=int(model.belief_dim+model.action_dim)
    total=sum(p.numel() for p in model.parameters())
    trunk_macs=inp*hidden+hidden*hidden
    return ComputeProfile("monolithic",int(total),int(total),int(trunk_macs),int(trunk_macs),hidden,1,1,hidden)


def matched_expert_hidden(input_dim:int,hidden:int,experts:int=4,top_k:int=1)->int:
    """Choose expert width so sparse trunk active MACs approximately match a 2-layer MLP trunk."""
    target=input_dim*hidden+hidden*hidden
    gate=input_dim*experts
    denom=max(1,top_k*(input_dim+hidden))
    return max(4,int(round(max(1,target-gate)/denom)))


def make_compute_matched_pair(belief_dim:int,action_dim:int,*,hidden:int=128,components:int=3,horizons=(1,2,4),experts:int=4,top_k:int=1):
    mono=MultimodalWorldModel(belief_dim,action_dim,hidden=hidden,components=components,horizons=horizons)
    eh=matched_expert_hidden(belief_dim+action_dim,hidden,experts,top_k)
    moe=ModularWorldModel(belief_dim,action_dim,hidden=hidden,components=components,horizons=horizons,experts=experts,top_k=top_k,expert_hidden=eh)
    return mono,moe,monolithic_compute_profile(mono),moe.compute_profile()
