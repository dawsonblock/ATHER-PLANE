from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn


@dataclass
class SkillDataset:
    states: torch.Tensor
    actions: torch.Tensor


class DistilledDiscreteSkill(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int=128):
        super().__init__(); self.action_dim=action_dim
        self.net=nn.Sequential(nn.Linear(state_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,action_dim))
    def forward(self,state): return self.net(state)
    @torch.no_grad()
    def act(self,state): return self(state).argmax(dim=-1)


def distill_discrete_skill(dataset: SkillDataset, action_dim: int, hidden_dim: int=128,
                           epochs: int=100, lr: float=3e-4, batch_size: int=128, seed: int=0):
    if dataset.states.ndim!=2 or dataset.actions.ndim!=1 or len(dataset.states)!=len(dataset.actions):
        raise ValueError("states must be [N,D] and actions [N]")
    torch.manual_seed(seed); model=DistilledDiscreteSkill(dataset.states.shape[-1],action_dim,hidden_dim).to(dataset.states.device)
    opt=torch.optim.AdamW(model.parameters(),lr=lr); n=len(dataset.states)
    for _ in range(int(epochs)):
        perm=torch.randperm(n,device=dataset.states.device)
        for start in range(0,n,batch_size):
            idx=perm[start:start+batch_size]; loss=torch.nn.functional.cross_entropy(model(dataset.states[idx]),dataset.actions[idx])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad(): acc=(model(dataset.states).argmax(-1)==dataset.actions).float().mean().item()
    return model,{"train_accuracy":float(acc),"examples":n}
