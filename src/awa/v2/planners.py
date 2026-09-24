from __future__ import annotations
import torch
from awa.v2.interfaces import PlanResult
from awa.v2.compute_ledger import PlannerExecutionFingerprint, chunk_schedule


def _concat_world_outputs(parts):
    if len(parts) == 1:
        return parts[0]
    out = {}
    for key in parts[0]:
        values = [part[key] for part in parts]
        if all(isinstance(v, torch.Tensor) for v in values):
            out[key] = torch.cat(values, dim=0)
        else:
            out[key] = values[-1]
    return out


def _batched_imagine(world, belief, action, *, deterministic=True, world_batch_size=None):
    schedule = chunk_schedule(int(belief.shape[0]), world_batch_size)
    if len(schedule) <= 1:
        return world.imagine_step(belief, action, deterministic=deterministic), schedule
    parts = []
    start = 0
    for size in schedule:
        end = start + int(size)
        parts.append(world.imagine_step(belief[start:end], action[start:end], deterministic=deterministic))
        start = end
    return _concat_world_outputs(parts), schedule


def _score_rollout_profiled(world, belief, actions, risk_weight=1.0, gamma=0.99, *, world_batch_size=None):
    b = belief
    score = torch.zeros(actions.shape[0], device=belief.device)
    alive = torch.ones_like(score)
    logical = 0
    physical_batches = []
    last = None
    for t in range(actions.shape[1]):
        last, schedule = _batched_imagine(
            world, b, actions[:, t], deterministic=True, world_batch_size=world_batch_size
        )
        logical += actions.shape[0]
        physical_batches.extend(schedule)
        r = last['reward'].squeeze(-1)
        risk = last['risk']
        cont = last['continuation'].squeeze(-1)
        score += alive * (r - risk_weight * risk) * (gamma ** t)
        alive *= cont
        b = last['belief']
    if last is not None and hasattr(world, 'terminal_value'):
        score += alive * world.terminal_value(b) * (gamma ** actions.shape[1])
    fp = PlannerExecutionFingerprint.from_batches(
        physical_batches, logical_world_model_transitions=logical,
        world_batch_limit=world_batch_size, schedule_kind='candidate_batched',
    )
    return score, logical, fp


def _score_rollout(world, belief, actions, risk_weight=1.0, gamma=0.99, *, world_batch_size=None):
    score, logical, _ = _score_rollout_profiled(
        world, belief, actions, risk_weight, gamma, world_batch_size=world_batch_size
    )
    return score, logical


def _execution_metadata(fp: PlannerExecutionFingerprint, **extra):
    return fp.to_dict() | extra


def _actor_prior(world, actor, belief, horizon):
    seq = []
    b = belief
    for _ in range(int(horizon)):
        a = actor.deterministic_action(b)
        if isinstance(a, tuple):
            a = a[0]
        seq.append(a)
        b = world.imagine_step(b, a, deterministic=True)['belief']
    return torch.stack(seq, 1)


class NoSearchPlanner:
    name = 'actor'
    def __init__(self, actor): self.actor = actor
    @torch.no_grad()
    def plan(self, belief, actor=None, budget=None, goal=None):
        a = (actor or self.actor).deterministic_action(belief)
        if isinstance(a, tuple): a = a[0]
        return PlanResult(a, float('nan'), self.name, 0, _execution_metadata(PlannerExecutionFingerprint(),search=False))


class CEMPlanner:
    """Vanilla continuous CEM control.

    This is intentionally policy-free and exists as a clean control for policy-seeded
    search. `budget` means candidates per iteration; equal-compute conversion is
    handled by `awa.v2.arena`.
    """
    name = 'cem'
    def __init__(self, world, low, high, horizon=12, candidates=192, elites=24,
                 iterations=4, momentum=.2, risk_weight=1.0, init_std=.5, world_batch_size=None):
        self.world=world; self.horizon=int(horizon); self.candidates=int(candidates)
        self.elites=int(elites); self.iterations=int(iterations); self.momentum=float(momentum)
        self.risk_weight=float(risk_weight); self.init_std=float(init_std); self.world_batch_size=world_batch_size
        self.low=torch.as_tensor(low,dtype=torch.float32); self.high=torch.as_tensor(high,dtype=torch.float32)
    @torch.no_grad()
    def plan(self, belief, actor=None, budget=None, goal=None):
        H=self.horizon; C=max(1,int(budget or self.candidates)); E=min(self.elites,C)
        low=self.low.to(belief.device); high=self.high.to(belief.device)
        mean=((low+high)*.5).expand(H,-1).clone()
        std=((high-low)*self.init_std).expand(H,-1).clone()
        calls=0; best=float('-inf'); fp_total=PlannerExecutionFingerprint()
        for _ in range(self.iterations):
            acts=(mean.unsqueeze(0)+std.unsqueeze(0)*torch.randn(C,H,mean.shape[-1],device=belief.device)).clamp(low,high)
            scores,c,fp=_score_rollout_profiled(self.world,belief.expand(C,-1),acts,self.risk_weight,world_batch_size=self.world_batch_size); calls+=c
            fp_total=fp_total.plus(fp,schedule_kind='candidate_batched')
            top=scores.topk(E).indices; elite=acts[top]
            nm=elite.mean(0); ns=elite.std(0,unbiased=False).clamp_min(0.02*(high-low))
            mean=self.momentum*mean+(1-self.momentum)*nm
            std=self.momentum*std+(1-self.momentum)*ns
            best=max(best,float(scores.max()))
        return PlanResult(mean[0:1].clamp(low,high),best,self.name,calls,
                          _execution_metadata(fp_total,candidates=C,horizon=H,iterations=self.iterations,policy_seeded=False))


class MPPIPlanner:
    """Uninformed MPPI control using a neutral action trajectory as the nominal plan."""
    name='mppi'
    def __init__(self,world,low,high,horizon=12,candidates=256,temperature=1.0,noise_std=.35,risk_weight=1.0,world_batch_size=None):
        self.world=world; self.horizon=int(horizon); self.candidates=int(candidates)
        self.temperature=float(temperature); self.noise_std=float(noise_std); self.risk_weight=float(risk_weight); self.world_batch_size=world_batch_size
        self.low=torch.as_tensor(low,dtype=torch.float32); self.high=torch.as_tensor(high,dtype=torch.float32)
    @torch.no_grad()
    def plan(self,belief,actor=None,budget=None,goal=None):
        H=self.horizon; C=max(1,int(budget or self.candidates)); low=self.low.to(belief.device); high=self.high.to(belief.device)
        nominal=((low+high)*.5).expand(1,H,-1)
        noise=torch.randn(C,H,low.numel(),device=belief.device)*self.noise_std*(high-low)
        acts=(nominal+noise).clamp(low,high); acts[0]=nominal[0]
        scores,calls,fp=_score_rollout_profiled(self.world,belief.expand(C,-1),acts,self.risk_weight,world_batch_size=self.world_batch_size)
        w=torch.softmax((scores-scores.max())/max(self.temperature,1e-5),0)
        refined=(w[:,None,None]*acts).sum(0)
        return PlanResult(refined[0:1].clamp(low,high),float(scores.max()),self.name,calls,
                          _execution_metadata(fp,mean_score=float(scores.mean()),candidates=C,horizon=H,policy_seeded=False))


class PolicySeededMPPI:
    name='policy_mppi'
    def __init__(self,world,actor,low,high,horizon=12,candidates=256,temperature=1.0,noise_std=0.35,risk_weight=1.0,world_batch_size=None):
        self.world=world; self.actor=actor; self.horizon=int(horizon); self.candidates=int(candidates)
        self.temperature=float(temperature); self.noise_std=float(noise_std); self.risk_weight=float(risk_weight); self.world_batch_size=world_batch_size
        self.low=torch.as_tensor(low,dtype=torch.float32); self.high=torch.as_tensor(high,dtype=torch.float32)
    @torch.no_grad()
    def plan(self,belief,actor=None,budget=None,goal=None):
        H=self.horizon; C=max(1,int(budget or self.candidates)); prior=_actor_prior(self.world,actor or self.actor,belief,H)
        low=self.low.to(belief.device); high=self.high.to(belief.device)
        base=prior.expand(C,-1,-1); noise=torch.randn_like(base)*self.noise_std*(high-low)
        actions=(base+noise).clamp(low,high); actions[0]=prior[0]
        scores,calls,fp=_score_rollout_profiled(self.world,belief.expand(C,-1),actions,self.risk_weight,world_batch_size=self.world_batch_size)
        prior_fp=PlannerExecutionFingerprint.from_batches([1]*H,logical_world_model_transitions=H,schedule_kind='actor_prior')
        fp=prior_fp.plus(fp,schedule_kind='actor_prior+candidate_batched')
        weights=torch.softmax((scores-scores.max())/max(self.temperature,1e-5),0)
        refined=(weights[:,None,None]*actions).sum(0)
        return PlanResult(refined[0:1].clamp(low,high),float(scores.max()),self.name,calls,
                          _execution_metadata(fp,mean_score=float(scores.mean()),candidates=C,horizon=H,policy_seeded=True,search_logical_world_model_transitions=calls))


class PolicySeededICEM:
    name='policy_icem'
    def __init__(self,world,actor,low,high,horizon=12,candidates=192,elites=24,iterations=4,momentum=.2,risk_weight=1.0,world_batch_size=None):
        self.world=world; self.actor=actor; self.horizon=int(horizon); self.candidates=int(candidates)
        self.elites=int(elites); self.iterations=int(iterations); self.momentum=float(momentum); self.risk_weight=float(risk_weight); self.world_batch_size=world_batch_size
        self.low=torch.as_tensor(low,dtype=torch.float32); self.high=torch.as_tensor(high,dtype=torch.float32)
    @torch.no_grad()
    def plan(self,belief,actor=None,budget=None,goal=None):
        H=self.horizon; C=max(1,int(budget or self.candidates)); E=min(self.elites,C)
        low=self.low.to(belief.device); high=self.high.to(belief.device)
        mean=_actor_prior(self.world,actor or self.actor,belief,H)[0]
        std=((high-low)*.35).expand(H,-1).clone(); calls=0; best=float('-inf')
        fp_total=PlannerExecutionFingerprint.from_batches([1]*H,logical_world_model_transitions=H,schedule_kind='actor_prior')
        for _ in range(self.iterations):
            eps=torch.randn(C,H,mean.shape[-1],device=belief.device)
            if H>1: eps[:,1:]=0.6*eps[:,:-1]+0.4*eps[:,1:]
            acts=(mean.unsqueeze(0)+std.unsqueeze(0)*eps).clamp(low,high); acts[0]=mean
            scores,c,fp=_score_rollout_profiled(self.world,belief.expand(C,-1),acts,self.risk_weight,world_batch_size=self.world_batch_size); calls+=c
            fp_total=fp_total.plus(fp,schedule_kind='actor_prior+candidate_batched')
            elite=acts[scores.topk(E).indices]; nm=elite.mean(0); ns=elite.std(0,unbiased=False).clamp_min(0.02*(high-low))
            mean=self.momentum*mean+(1-self.momentum)*nm; std=self.momentum*std+(1-self.momentum)*ns
            best=max(best,float(scores.max()))
        return PlanResult(mean[0:1],best,self.name,calls,
                          _execution_metadata(fp_total,candidates=C,horizon=H,iterations=self.iterations,policy_seeded=True,search_logical_world_model_transitions=calls))


class GradientPlanner:
    name='gradient'
    def __init__(self,world,low,high,horizon=12,steps=24,lr=.08,risk_weight=1.0):
        self.world=world; self.low=torch.as_tensor(low,dtype=torch.float32); self.high=torch.as_tensor(high,dtype=torch.float32)
        self.horizon=int(horizon); self.steps=int(steps); self.lr=float(lr); self.risk_weight=float(risk_weight)
    def plan(self,belief,actor=None,budget=None,goal=None):
        H=self.horizon; low=self.low.to(belief.device); high=self.high.to(belief.device)
        if actor is not None:
            with torch.no_grad(): init=_actor_prior(self.world,actor,belief,H)[0]
        else: init=((low+high)/2).expand(H,-1).clone()
        calls=0; score=torch.tensor(0.0,device=belief.device)
        # Explicitly enable gradients because planners are frequently invoked from
        # inference/no-grad controllers. Gradients are requested only w.r.t. the
        # action trajectory; world-model parameters are never accumulated.
        with torch.enable_grad():
            raw=torch.atanh((2*(init-low)/(high-low)-1).clamp(-.99,.99)).detach().requires_grad_(True)
            for _ in range(max(1,int(budget or self.steps))):
                unit=torch.tanh(raw); acts=(low+(unit+1)*.5*(high-low)).unsqueeze(0)
                b=belief.detach(); score=torch.tensor(0.0,device=belief.device); alive=torch.tensor(1.0,device=belief.device)
                for t in range(H):
                    out=self.world.imagine_step(b,acts[:,t],deterministic=True); calls+=1
                    score=score+alive*(out['reward'].mean()-self.risk_weight*out['risk'].mean())*(.99**t)
                    alive=alive*out['continuation'].mean(); b=out['belief']
                (grad,)=torch.autograd.grad(-score,raw,retain_graph=False,create_graph=False)
                with torch.no_grad(): raw=(raw-self.lr*grad).detach().requires_grad_(True)
            with torch.no_grad():
                unit=torch.tanh(raw); acts=low+(unit+1)*.5*(high-low)
        prior_calls=H if actor is not None else 0
        physical_batches=[1]*(calls+prior_calls)
        fp=PlannerExecutionFingerprint.from_batches(physical_batches,logical_world_model_transitions=calls+prior_calls,schedule_kind='serial_gradient')
        return PlanResult(acts[0:1],float(score.detach()),self.name,calls,
                          _execution_metadata(fp,horizon=H,gradient_steps=max(1,int(budget or self.steps)),policy_seeded=actor is not None,search_logical_world_model_transitions=calls))


class BeamPlanner:
    """Discrete skill/action beam search over a latent world model."""
    name='beam'
    def __init__(self,world,action_dim:int,horizon:int=6,beam_width:int=16,risk_weight:float=1.0):
        self.world=world; self.action_dim=int(action_dim); self.horizon=int(horizon); self.beam_width=int(beam_width); self.risk_weight=float(risk_weight)
    @torch.no_grad()
    def plan(self,belief,actor=None,budget=None,goal=None):
        width=max(1,int(budget or self.beam_width)); device=belief.device
        beams=[(0.0,belief,[],1.0)]; calls=0
        for t in range(self.horizon):
            candidates=[]
            for score,b,path,alive in beams:
                for aidx in range(self.action_dim):
                    a=torch.nn.functional.one_hot(torch.tensor([aidx],device=device),self.action_dim).float()
                    out=self.world.imagine_step(b,a,deterministic=True); calls+=1
                    inc=float((out['reward'].squeeze()-self.risk_weight*out['risk'].squeeze()).item())*(.99**t)*alive
                    nalive=alive*float(out['continuation'].squeeze().item())
                    candidates.append((score+inc,out['belief'],path+[aidx],nalive))
            candidates.sort(key=lambda x:x[0],reverse=True); beams=candidates[:width]
        best=beams[0]; idx=best[2][0]
        action=torch.nn.functional.one_hot(torch.tensor([idx],device=device),self.action_dim).float()
        fp=PlannerExecutionFingerprint.from_batches([1]*calls,logical_world_model_transitions=calls,schedule_kind='serial_beam')
        return PlanResult(action,float(best[0]),self.name,calls,_execution_metadata(fp,horizon=self.horizon,beam_width=width,path=best[2]))


def _score_stochastic_rollout_profiled(world, belief, actions, *, samples=8, risk_weight=1.0,
                                       gamma=0.99, cvar_alpha=0.25, cvar_weight=0.5,
                                       epistemic_weight=0.0, world_batch_size=None):
    """Score sampled futures while recording logical work and physical call schedule."""
    C,H,_=actions.shape; S=max(1,int(samples)); alpha=min(1.0,max(1.0/S,float(cvar_alpha)))
    b=belief.unsqueeze(1).expand(C,S,-1).reshape(C*S,-1)
    scores=torch.zeros(C*S,device=belief.device); alive=torch.ones_like(scores); logical=0; physical_batches=[]
    for t in range(H):
        a=actions[:,t].unsqueeze(1).expand(C,S,-1).reshape(C*S,-1)
        out,schedule=_batched_imagine(world,b,a,deterministic=False,world_batch_size=world_batch_size)
        logical+=C*S; physical_batches.extend(schedule)
        r=out['reward'].squeeze(-1); risk=out['risk']; cont=out['continuation'].squeeze(-1)
        penalty=torch.zeros_like(r)
        if epistemic_weight>0 and hasattr(world,'disagreement'):
            try: penalty=world.disagreement(b,a).next_state
            except Exception: penalty=torch.zeros_like(r)
        scores += alive*(r-risk_weight*risk-epistemic_weight*penalty)*(gamma**t)
        alive*=cont; b=out['belief']
    if hasattr(world,'terminal_value'):
        scores += alive*world.terminal_value(b)*(gamma**H)
    rows=scores.view(C,S); mean=rows.mean(1)
    k=max(1,int(round(S*alpha))); cvar=rows.sort(1).values[:,:k].mean(1)
    robust=mean-float(cvar_weight)*(mean-cvar)
    fp=PlannerExecutionFingerprint.from_batches(physical_batches,logical_world_model_transitions=logical,world_batch_limit=world_batch_size,schedule_kind='stochastic_candidate_batched')
    return robust,logical,{'mean':mean,'cvar':cvar,'sample_std':rows.std(1,unbiased=False)},fp


def _score_stochastic_rollout(world, belief, actions, *, samples=8, risk_weight=1.0,
                              gamma=0.99, cvar_alpha=0.25, cvar_weight=0.5,
                              epistemic_weight=0.0, world_batch_size=None):
    robust,logical,stats,_=_score_stochastic_rollout_profiled(
        world,belief,actions,samples=samples,risk_weight=risk_weight,gamma=gamma,
        cvar_alpha=cvar_alpha,cvar_weight=cvar_weight,epistemic_weight=epistemic_weight,
        world_batch_size=world_batch_size,
    )
    return robust,logical,stats


class RiskAwarePolicySeededMPPI:
    """Policy-seeded MPPI scored over sampled futures and lower-tail CVaR."""
    name='risk_policy_mppi'
    def __init__(self,world,actor,low,high,horizon=12,candidates=128,temperature=1.0,
                 noise_std=.35,risk_weight=1.0,samples=8,cvar_alpha=.25,
                 cvar_weight=.5,epistemic_weight=.1,world_batch_size=None):
        self.world=world; self.actor=actor; self.horizon=int(horizon); self.candidates=int(candidates)
        self.temperature=float(temperature); self.noise_std=float(noise_std); self.risk_weight=float(risk_weight)
        self.samples=max(1,int(samples)); self.cvar_alpha=float(cvar_alpha); self.cvar_weight=float(cvar_weight)
        self.epistemic_weight=float(epistemic_weight); self.world_batch_size=world_batch_size
        self.low=torch.as_tensor(low,dtype=torch.float32); self.high=torch.as_tensor(high,dtype=torch.float32)
    @torch.no_grad()
    def plan(self,belief,actor=None,budget=None,goal=None):
        H=self.horizon; C=max(1,int(budget or self.candidates)); low=self.low.to(belief.device); high=self.high.to(belief.device)
        prior=_actor_prior(self.world,actor or self.actor,belief,H)
        base=prior.expand(C,-1,-1); noise=torch.randn_like(base)*self.noise_std*(high-low)
        actions=(base+noise).clamp(low,high); actions[0]=prior[0]
        scores,calls,stats,fp=_score_stochastic_rollout_profiled(
            self.world,belief.expand(C,-1),actions,samples=self.samples,risk_weight=self.risk_weight,
            cvar_alpha=self.cvar_alpha,cvar_weight=self.cvar_weight,epistemic_weight=self.epistemic_weight,
            world_batch_size=self.world_batch_size,
        )
        prior_fp=PlannerExecutionFingerprint.from_batches([1]*H,logical_world_model_transitions=H,schedule_kind='actor_prior')
        fp=prior_fp.plus(fp,schedule_kind='actor_prior+stochastic_candidate_batched')
        weights=torch.softmax((scores-scores.max())/max(self.temperature,1e-5),0)
        refined=(weights[:,None,None]*actions).sum(0)
        best=int(scores.argmax())
        return PlanResult(refined[0:1].clamp(low,high),float(scores[best]),self.name,calls,_execution_metadata(fp,
            mean_score=float(stats['mean'][best]),cvar_score=float(stats['cvar'][best]),
            sample_std=float(stats['sample_std'][best]),candidates=C,horizon=H,
            samples=self.samples,cvar_alpha=self.cvar_alpha,policy_seeded=True,search_logical_world_model_transitions=calls,
        ))

class HybridRiskAwarePolicySeededMPPI:
    """Risk-aware MPPI for Aether's hybrid game ABI.

    Movement dimensions 0:2 remain continuous. Attack/interact dimensions 2:4 are
    always exactly -1/+1 for every world-model rollout and for the returned action.
    This avoids training/planning on physically meaningless fractional button states.
    """
    name='hybrid_risk_policy_mppi'
    def __init__(self,world,actor,horizon=12,candidates=128,temperature=1.0,
                 movement_noise_std=.35,binary_flip_prob=.12,risk_weight=1.0,
                 samples=8,cvar_alpha=.25,cvar_weight=.5,epistemic_weight=.1,world_batch_size=None):
        self.world=world; self.actor=actor; self.horizon=int(horizon); self.candidates=int(candidates)
        self.temperature=float(temperature); self.movement_noise_std=float(movement_noise_std)
        self.binary_flip_prob=float(binary_flip_prob); self.risk_weight=float(risk_weight)
        self.samples=max(1,int(samples)); self.cvar_alpha=float(cvar_alpha); self.cvar_weight=float(cvar_weight)
        self.epistemic_weight=float(epistemic_weight); self.world_batch_size=world_batch_size

    @torch.no_grad()
    def plan(self,belief,actor=None,budget=None,goal=None):
        policy=actor or self.actor; H=self.horizon; C=max(1,int(budget or self.candidates))
        prior=_actor_prior(self.world,policy,belief,H)
        if prior.shape[-1] != 4:
            raise ValueError('hybrid Doom planner requires 4-D [turn,forward,attack,use] actions')
        base=prior.expand(C,-1,-1).clone()
        # Continuous movement search.
        base[:,:,0:2]=(base[:,:,0:2]+torch.randn_like(base[:,:,0:2])*self.movement_noise_std*2.0).clamp(-1,1)
        # Discrete controls are sampled as exact signs around the actor proposal.
        bits=torch.where(base[:,:,2:4]>=0,torch.ones_like(base[:,:,2:4]),-torch.ones_like(base[:,:,2:4]))
        flips=torch.rand_like(bits)<self.binary_flip_prob
        bits=torch.where(flips,-bits,bits)
        base[:,:,2:4]=bits
        base[0]=prior[0]
        base[0,:,2:4]=torch.where(base[0,:,2:4]>=0,torch.ones_like(base[0,:,2:4]),-torch.ones_like(base[0,:,2:4]))
        scores,calls,stats,fp=_score_stochastic_rollout_profiled(
            self.world,belief.expand(C,-1),base,samples=self.samples,risk_weight=self.risk_weight,
            cvar_alpha=self.cvar_alpha,cvar_weight=self.cvar_weight,epistemic_weight=self.epistemic_weight,
            world_batch_size=self.world_batch_size,
        )
        prior_fp=PlannerExecutionFingerprint.from_batches([1]*H,logical_world_model_transitions=H,schedule_kind='actor_prior')
        fp=prior_fp.plus(fp,schedule_kind='actor_prior+stochastic_candidate_batched')
        weights=torch.softmax((scores-scores.max())/max(self.temperature,1e-5),0)
        move=(weights[:,None,None]*base[:,:,:2]).sum(0)
        # Weighted majority vote preserves discrete semantics in the final plan.
        binary_score=(weights[:,None,None]*base[:,:,2:4]).sum(0)
        binary=torch.where(binary_score>=0,torch.ones_like(binary_score),-torch.ones_like(binary_score))
        refined=torch.cat([move,binary],dim=-1)
        best=int(scores.argmax())
        return PlanResult(refined[0:1],float(scores[best]),self.name,calls,_execution_metadata(fp,
            mean_score=float(stats['mean'][best]),cvar_score=float(stats['cvar'][best]),
            sample_std=float(stats['sample_std'][best]),candidates=C,horizon=H,
            samples=self.samples,cvar_alpha=self.cvar_alpha,policy_seeded=True,
            hybrid_action=True,binary_flip_prob=self.binary_flip_prob,search_logical_world_model_transitions=calls,
        ))
