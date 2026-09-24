from __future__ import annotations
import copy
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from awa.planning.actor import TanhGaussianActor
from awa.training.continuous_critic import TwinQCritic, TargetTwinQCritic


class LatentTransitionDataset(Dataset):
    """Continuous transition dataset in the exact latent space used by v2 planners."""
    def __init__(self, states, actions, rewards, next_states, dones):
        self.states=torch.as_tensor(states,dtype=torch.float32)
        self.actions=torch.as_tensor(actions,dtype=torch.float32)
        self.rewards=torch.as_tensor(rewards,dtype=torch.float32).reshape(-1)
        self.next_states=torch.as_tensor(next_states,dtype=torch.float32)
        self.dones=torch.as_tensor(dones,dtype=torch.float32).reshape(-1)
        n=len(self.states)
        if not all(len(v)==n for v in (self.actions,self.rewards,self.next_states,self.dones)):
            raise ValueError('transition arrays must have equal first dimension')
    def __len__(self): return len(self.states)
    def __getitem__(self,i):
        return self.states[i],self.actions[i],self.rewards[i],self.next_states[i],self.dones[i]


class WeightedLatentTransitionDataset(LatentTransitionDataset):
    def __init__(self, states, actions, rewards, next_states, dones, weights):
        super().__init__(states, actions, rewards, next_states, dones)
        self.weights=torch.as_tensor(weights,dtype=torch.float32).reshape(-1)
        if len(self.weights)!=len(self.states): raise ValueError('weights must match transition count')
        if torch.any(~torch.isfinite(self.weights)) or torch.any(self.weights<=0): raise ValueError('weights must be finite and > 0')
    def __getitem__(self,i):
        return (*super().__getitem__(i), self.weights[i])


@dataclass
class ActorBaselineReport:
    critic_loss: float
    actor_loss: float
    bc_loss: float
    q_value: float
    steps: int
    def to_dict(self): return asdict(self)


class OfflineActorCriticBaseline:
    """Compact TD3+BC-style offline actor baseline.

    This intentionally provides a strong no-search control without introducing a
    second world model. Critic targets are bootstrapped from a slowly updated
    target actor and target twin-Q critic; actor optimization combines conservative
    Q maximization with behavior cloning to remain anchored to the offline data.
    """
    def __init__(self, state_dim:int, action_dim:int, low, high, hidden:int=256,
                 gamma:float=.99, tau:float=.01, bc_weight:float=2.5, lr:float=3e-4,
                 device='cpu'):
        self.device=torch.device(device); self.gamma=float(gamma); self.tau=float(tau); self.bc_weight=float(bc_weight)
        self.state_dim=int(state_dim); self.action_dim=int(action_dim); self.hidden=int(hidden)
        self.actor=TanhGaussianActor(state_dim,action_dim,hidden,low,high).to(self.device)
        self.target_actor=copy.deepcopy(self.actor).to(self.device)
        for p in self.target_actor.parameters(): p.requires_grad=False
        self.critic=TwinQCritic(state_dim,action_dim,hidden,low,high).to(self.device)
        self.target_critic=TargetTwinQCritic(self.critic).to(self.device)
        self.actor_opt=torch.optim.Adam(self.actor.parameters(),lr=lr)
        self.critic_opt=torch.optim.Adam(self.critic.parameters(),lr=lr)
        self.steps=0

    @torch.no_grad()
    def _target_update(self):
        for t,s in zip(self.target_actor.parameters(),self.actor.parameters()): t.data.lerp_(s.data,self.tau)
        self.target_critic.update(self.critic,self.tau)

    def train_batch(self,batch):
        if len(batch)==6:
            s,a,r,ns,d,w=[x.to(self.device) for x in batch]
            w=w.reshape(-1); w=w/w.mean().clamp_min(1e-6)
        else:
            s,a,r,ns,d=[x.to(self.device) for x in batch]; w=torch.ones_like(r)
        with torch.no_grad():
            na=self.target_actor.deterministic_action(ns)
            nq=self.target_critic.minimum(ns,na)
            target=r+self.gamma*(1.0-d)*nq
        q1,q2=self.critic(s,a)
        critic_rows=torch.nn.functional.smooth_l1_loss(q1,target,reduction='none')+torch.nn.functional.smooth_l1_loss(q2,target,reduction='none')
        critic_loss=(critic_rows*w).mean()
        self.critic_opt.zero_grad(set_to_none=True); critic_loss.backward(); self.critic_opt.step()

        pi=self.actor.deterministic_action(s)
        q=self.critic.minimum(s,pi)
        bc_rows=(pi-a).square().mean(-1)
        bc=(bc_rows*w).mean()
        scale=((q.abs()*w).mean().detach()+1e-4).reciprocal()
        actor_loss=-(scale*(q*w).mean())+self.bc_weight*bc
        self.actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); self.actor_opt.step()
        self._target_update(); self.steps+=1
        return float(critic_loss.detach()),float(actor_loss.detach()),float(bc.detach()),float(q.mean().detach())

    def fit(self,dataset:Dataset,epochs:int=20,batch_size:int=128,shuffle:bool=True):
        if len(dataset)==0: raise ValueError('empty actor dataset')
        last=(0.,0.,0.,0.)
        for _ in range(int(epochs)):
            for batch in DataLoader(dataset,batch_size=min(batch_size,len(dataset)),shuffle=shuffle):
                last=self.train_batch(batch)
        return ActorBaselineReport(*last,self.steps)

    @torch.no_grad()
    def action(self,state):
        s=torch.as_tensor(state,dtype=torch.float32,device=self.device)
        if s.ndim==1: s=s.unsqueeze(0)
        return self.actor.deterministic_action(s)

    def state_dict(self):
        return {'actor':self.actor.state_dict(),'target_actor':self.target_actor.state_dict(),
                'critic':self.critic.state_dict(),'target_critic':self.target_critic.state_dict(),
                'actor_opt':self.actor_opt.state_dict(),'critic_opt':self.critic_opt.state_dict(),
                'steps':self.steps,'gamma':self.gamma,'tau':self.tau,'bc_weight':self.bc_weight}

    def load_state_dict(self, state, *, load_optimizers: bool = True):
        """Restore a complete offline actor/critic state for cumulative training.

        Older checkpoints already contain optimizer state, but historically Aether only
        consumed them for frozen evaluation.  v2.25 makes milestone training cumulative,
        so the training path needs an explicit, validated restore operation as well.
        """
        self.actor.load_state_dict(state['actor'], strict=True)
        self.target_actor.load_state_dict(state.get('target_actor', state['actor']), strict=True)
        self.critic.load_state_dict(state['critic'], strict=True)
        self.target_critic.load_state_dict(state['target_critic'], strict=True)
        if load_optimizers:
            if 'actor_opt' in state:
                self.actor_opt.load_state_dict(state['actor_opt'])
            if 'critic_opt' in state:
                self.critic_opt.load_state_dict(state['critic_opt'])
        self.steps = int(state.get('steps', 0))
        self.gamma = float(state.get('gamma', self.gamma))
        self.tau = float(state.get('tau', self.tau))
        self.bc_weight = float(state.get('bc_weight', self.bc_weight))
        return self

@torch.no_grad()
def latent_dataset_from_cached_features(dataset, cache, index:dict, projector:nn.Module):
    """Build the actor-only baseline from the same frozen-feature cache used by v2.2.

    This keeps actor-vs-planner comparisons in one representation space and avoids
    giving either path a different perception stack.
    """
    if index.get('dataset_sha256') != dataset.manifest.sha256:
        raise ValueError('representation index does not match dataset SHA-256')
    obs=[]; nxt=[]
    for ok,nk in zip(index['observation_keys'],index['next_observation_keys']):
        of=cache.get_by_key(ok); nf=cache.get_by_key(nk)
        if of is None or nf is None: raise FileNotFoundError('representation cache entry missing')
        obs.append(of.reshape(1,-1)); nxt.append(nf.reshape(1,-1))
    s=projector(torch.cat(obs,0)).detach().cpu(); ns=projector(torch.cat(nxt,0)).detach().cpu()
    return LatentTransitionDataset(s,dataset.arrays['actions'],dataset.arrays['rewards'],ns,dataset.arrays['dones'])
