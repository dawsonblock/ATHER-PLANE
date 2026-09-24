from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from awa.actions import action_spec_from_config, ActionSpec
from awa.model import WorldModel
from awa.planning.actor import CategoricalActor, TanhGaussianActor
from awa.planning.uncertainty import UncertaintyEnsemble
from awa.planning.calibrator import LogisticUncertaintyCalibrator
from awa.planning.learned_gate import LearnedArbitrator
from awa.planning.adaptive import AdaptivePlanningPolicy
from awa.planning.mpc import CEMPlanner, ContinuousCEMPlanner
from awa.runtime.engine import AgentEngine
from awa.memory.replay import EpisodeReplayBuffer, PrioritizedEpisodeReplayBuffer, Transition
from awa.environments.factory import make_environment
from awa.runtime.checkpoint import CheckpointManager
from awa.runtime.telemetry import JsonlTelemetry
from awa.training.returns import lambda_returns
from awa.training.target import TargetValueNetwork
from awa.training.sequence import continuation_mask, overshooting_latent_loss
from awa.training.continuous_critic import TwinQCritic, QuantileTwinQCritic, TargetTwinQCritic, RunningTargetScale, quantile_huber_loss


@dataclass
class Components:
    world_model: WorldModel
    actor: torch.nn.Module
    uncertainty: UncertaintyEnsemble
    planner: object | None
    engine: AgentEngine
    target_value: TargetValueNetwork
    action_spec: ActionSpec
    calibrator: LogisticUncertaintyCalibrator | None = None
    arbitrator: LearnedArbitrator | None = None
    q_critic: TwinQCritic | None = None
    target_q: TargetTwinQCritic | None = None
    q_scale: RunningTargetScale | None = None


def build_components(cfg, device) -> Components:
    m=cfg["model"]; spec=action_spec_from_config(cfg)
    wm=WorldModel(obs_dim=m["obs_dim"],latent_dim=m["latent_dim"],deterministic_dim=m["deterministic_dim"],
                  stochastic_dim=m["stochastic_dim"],action_dim=m["action_dim"],hidden_dim=m["hidden_dim"],
                  dynamics=m["dynamics"],slow_enabled=m.get("slow_enabled",False),slow_dim=m.get("slow_dim",128),
                  slow_stride=m.get("slow_stride",8),experts_enabled=m.get("experts_enabled",False),
                  experts_count=m.get("experts_count",4),experts_top_k=m.get("experts_top_k",2),
                  expert_context_dim=m.get("expert_context_dim",0)).to(device)
    if spec.kind=="discrete":
        actor=CategoricalActor(wm.belief_dim,m["action_dim"],m["hidden_dim"]).to(device)
    else:
        actor=TanhGaussianActor(wm.belief_dim,m["action_dim"],m["hidden_dim"],spec.low,spec.high,
                                float(m.get("actor_log_std_min",-5.0)),float(m.get("actor_log_std_max",2.0))).to(device)
    uncertainty=UncertaintyEnsemble(wm.belief_dim,m["latent_dim"],m["uncertainty_heads"],m["hidden_dim"],action_dim=m["action_dim"]).to(device)
    p=cfg["planner"]; planner=None
    if p["enabled"]:
        if spec.kind=="discrete":
            planner=CEMPlanner(wm,m["action_dim"],p["horizon"],p["candidates"],p["elites"],p["iterations"])
        else:
            planner=ContinuousCEMPlanner(wm,m["action_dim"],spec.low,spec.high,p["horizon"],p["candidates"],p["elites"],p["iterations"],
                                         float(p.get("min_std",0.03)),float(p.get("momentum",0.10)))
    calibrator=LogisticUncertaintyCalibrator().to(device) if p.get("calibrated_uncertainty",False) else None
    arbitrator=LearnedArbitrator(int(p.get("learned_gate_hidden",32))).to(device) if p.get("learned_gate",False) else None
    adaptive=None
    if p.get("adaptive",False):
        adaptive=AdaptivePlanningPolicy(p["horizon"],p["candidates"],float(p.get("uncertainty_low",0.25)),
                                        float(p.get("uncertainty_medium",0.60)),float(p.get("uncertainty_high",0.85)),
                                        int(p.get("min_horizon",2)),int(p.get("min_candidates",16)))
    q_critic=target_q=q_scale=None
    cq=cfg.get("training",{}).get("continuous_q",{})
    if spec.kind=="continuous" and cq.get("enabled",False):
        critic_type=str(cq.get("critic_type","twin_q"))
        if critic_type=="quantile_twin_q":
            q_critic=QuantileTwinQCritic(wm.belief_dim,m["action_dim"],int(cq.get("hidden_dim",m["hidden_dim"])),int(cq.get("quantiles",32)),spec.low,spec.high).to(device)
        elif critic_type=="twin_q":
            q_critic=TwinQCritic(wm.belief_dim,m["action_dim"],int(cq.get("hidden_dim",m["hidden_dim"])),spec.low,spec.high).to(device)
        else:
            raise ValueError(f"unknown continuous_q.critic_type: {critic_type}")
        target_q=TargetTwinQCritic(q_critic).to(device); q_scale=RunningTargetScale(float(cq.get("scale_momentum",0.01))).to(device)
    engine=AgentEngine(wm,actor,uncertainty,planner,spec,p.get("uncertainty_limit",1.5),calibrator,arbitrator,adaptive,p.get("learned_gate_threshold",0.5))
    return Components(wm,actor,uncertainty,planner,engine,TargetValueNetwork(wm.value_head).to(device),spec,calibrator,arbitrator,q_critic,target_q,q_scale)


def _weighted(values, mask, iw=None):
    if values.ndim==1: values=values[:,None]
    m=mask
    if m.ndim==1:m=m[:,None]
    if iw is not None:
        w=iw
        if w.ndim==1:w=w[:,None]
        m=m*w
    return (values*m).sum()/m.sum().clamp_min(1e-8)


def _batch_action(batch_actions: torch.Tensor, t: int, spec: ActionSpec) -> torch.Tensor:
    return spec.encode_tensor(batch_actions[:,t])


def _sequence_world_loss(c,batch,cfg):
    wm=c.world_model; B=batch.actions.shape[0]; T=batch.actions.shape[1]; A=cfg["model"]["action_dim"]; burn=int(cfg["training"].get("burn_in",0))
    b=wm.initial_belief(B,batch.observations.device); zero=c.action_spec.zero(B,batch.observations.device)
    first=wm.observe(b,zero,batch.observations[:,0],step_index=0); b=first.belief
    valid=continuation_mask(batch.dones).squeeze(-1); iw=batch.importance_weights
    losses={k:torch.zeros((),device=batch.observations.device) for k in ("latent","reward","cont","kl","uncertainty","routing")}
    posterior=[]; seq_error=torch.zeros(B,device=batch.observations.device); count=0; router_count=0
    calibration_u=[]; calibration_err=[]
    for t in range(T):
        a=_batch_action(batch.actions,t,c.action_spec); previous=b
        out=wm.observe(b,a,batch.observations[:,t+1],step_index=t+1); b=out.belief; posterior.append(b)
        if t<burn: continue
        mask=valid[:,t]; target=wm.encoder(batch.observations[:,t+1]).detach()
        latent_per=((wm.latent_prediction(b)-target)**2).mean(-1); reward_per=((out.reward-batch.rewards[:,t])**2).squeeze(-1)
        cont_per=F.binary_cross_entropy_with_logits(out.continuation_logit,1.0-batch.dones[:,t],reduction="none").squeeze(-1)
        kl_per=torch.distributions.kl_divergence(out.posterior,out.prior).mean(-1).clamp_min(float(cfg["training"].get("free_nats",0.1)))
        pred,raw_u=c.uncertainty(previous.vector,a); unc_per=((pred-target)**2).mean(-1)
        calibration_u.append(raw_u.detach()); calibration_err.append(unc_per.detach())
        for k,v in (("latent",latent_per),("reward",reward_per),("cont",cont_per),("kl",kl_per),("uncertainty",unc_per)): losses[k]=losses[k]+_weighted(v,mask,iw)
        seq_error += (latent_per.detach()+reward_per.detach()+cont_per.detach())*mask
        if out.router_weights is not None:
            mean_route=out.router_weights.mean(0); uniform=torch.full_like(mean_route,1.0/mean_route.numel()); losses["routing"] += F.mse_loss(mean_route,uniform); router_count+=1
        count+=1
    denom=max(count,1)
    for k in ("latent","reward","cont","kl","uncertainty"): losses[k]/=denom
    if router_count: losses["routing"]/=router_count
    overshoot=overshooting_latent_loss(wm,posterior,batch.observations,batch.actions,A,int(cfg["training"].get("overshoot_horizon",1)),burn,c.action_spec.kind)
    losses["overshoot"]=overshoot
    total=losses["latent"]+losses["reward"]+losses["cont"]+0.1*losses["kl"]+losses["uncertainty"]
    total += float(cfg["training"].get("expert_balance_coef",0.01))*losses["routing"]
    total += float(cfg["training"].get("overshoot_coef",0.0))*overshoot
    calibration=(torch.cat(calibration_u) if calibration_u else torch.empty(0,device=batch.observations.device),
                 torch.cat(calibration_err) if calibration_err else torch.empty(0,device=batch.observations.device))
    return total,losses,[x.detach() for x in posterior],seq_error/max(denom,1),calibration


def _actor_sample(actor, belief_vec, spec: ActionSpec):
    if spec.kind=="discrete":
        dist=actor.distribution(belief_vec); idx=dist.sample(); a=F.one_hot(idx,actor.action_dim).float()
        return a,dist.log_prob(idx),dist.entropy()
    return actor.sample_with_log_prob(belief_vec)


def _imagination_update(c,start_belief,actor_opt,value_opt,cfg):
    wm,actor=c.world_model,c.actor; tc=cfg["training"]; H=int(tc.get("imagination_horizon",8)); gamma=float(tc.get("discount",0.99)); lam=float(tc.get("lambda",0.95)); ent=float(tc.get("entropy_coef",0.003))
    belief=start_belief.detach(); logp=[]; ents=[]; rewards=[]; discounts=[]; values=[]; feats=[]
    for t in range(H):
        feat,_=wm.features(belief); feats.append(feat.detach()); a,lp,en=_actor_sample(actor,belief.vector.detach(),c.action_spec); logp.append(lp); ents.append(en)
        with torch.no_grad(): nxt,r,cont,_,_,_=wm.imagine(belief,a,step_index=t)
        rewards.append(r.squeeze(-1)); discounts.append(torch.sigmoid(cont.squeeze(-1))*gamma)
        with torch.no_grad(): values.append(c.target_value(feat).squeeze(-1))
        belief=nxt.detach()
    with torch.no_grad(): bf,_=wm.features(belief); bootstrap=c.target_value(bf).squeeze(-1)
    vs=torch.stack(values); rs=torch.stack(rewards); ds=torch.stack(discounts); returns=lambda_returns(rs,vs,ds,lam,bootstrap)
    lp=torch.stack(logp); en=torch.stack(ents); adv=returns-vs
    if tc.get("advantage_normalize",True): adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-6)
    weights=torch.cumprod(torch.cat([torch.ones_like(ds[:1]),ds[:-1]],0),0)
    actor_loss=-((lp*adv.detach())*weights.detach()).sum()/weights.detach().sum().clamp_min(1e-8)-ent*en.mean()
    if c.action_spec.kind=="continuous" and c.q_critic is not None:
        qa,qlp,_=c.actor.sample_with_log_prob(start_belief.vector.detach())
        for p in c.q_critic.parameters(): p.requires_grad_(False)
        q_value=c.q_critic.minimum(start_belief.vector.detach(),qa)
        q_actor=(float(tc.get("continuous_q",{}).get("entropy_alpha",0.01))*qlp-q_value).mean()
        actor_loss=actor_loss+float(tc.get("continuous_q",{}).get("actor_coef",0.25))*q_actor
        for p in c.q_critic.parameters(): p.requires_grad_(True)
    actor_opt.zero_grad(set_to_none=True); actor_loss.backward(); torch.nn.utils.clip_grad_norm_(actor.parameters(),tc["grad_clip"]); actor_opt.step()
    critic=torch.stack([wm.value_head(v).squeeze(-1) for v in feats]); critic_loss=((critic-returns.detach())**2*weights.detach()).sum()/weights.sum().clamp_min(1e-8)
    value_opt.zero_grad(set_to_none=True); critic_loss.backward(); torch.nn.utils.clip_grad_norm_(wm.value_head.parameters(),tc["grad_clip"]); value_opt.step(); c.target_value.update(wm.value_head,float(tc.get("target_tau",0.01)))
    return float(actor_loss.item()),float(critic_loss.item()),float(en.mean().item())



def _continuous_q_update(c,batch,q_opt,cfg):
    if c.q_critic is None or c.target_q is None or c.q_scale is None or q_opt is None:
        return 0.0
    tc=cfg["training"]; cq=tc.get("continuous_q",{}); gamma=float(tc.get("discount",0.99)); tau=float(cq.get("target_tau",0.01))
    n_step=max(1,int(cq.get("n_step",1)))
    wm=c.world_model; B=batch.actions.shape[0]; T=batch.actions.shape[1]; device=batch.observations.device

    # Reconstruct the posterior belief trajectory once. beliefs[t] is the state
    # before action[t]; beliefs[T] is the posterior after the last transition.
    with torch.no_grad():
        b=wm.initial_belief(B,device); zero=c.action_spec.zero(B,device)
        b=wm.observe(b,zero,batch.observations[:,0],step_index=0).belief
        beliefs=[b]
        for t in range(T):
            b=wm.observe(b,batch.actions[:,t].float(),batch.observations[:,t+1],step_index=t+1).belief
            beliefs.append(b)

    losses=[]; scalar_targets=[]
    for t in range(T):
        # n-step real replay return with termination-aware bootstrapping.
        with torch.no_grad():
            ret=torch.zeros(B,device=device); discount=torch.ones(B,device=device)
            steps=0
            for k in range(n_step):
                j=t+k
                if j>=T: break
                r=batch.rewards[:,j].squeeze(-1); done=batch.dones[:,j].squeeze(-1)
                ret += discount*r
                discount = discount*gamma*(1.0-done)
                steps += 1
            boot_index=min(t+steps,T)
            boot_belief=beliefs[boot_index]
            next_action=c.actor.deterministic_action(boot_belief.vector)
            if getattr(c.q_critic,'distributional',False):
                target_quantiles=c.target_q.conservative_quantiles(boot_belief.vector,next_action)
                target_samples=ret[:,None]+discount[:,None]*target_quantiles
                scalar_y=target_samples.mean(-1)
            else:
                target_q=c.target_q.minimum(boot_belief.vector,next_action)
                scalar_y=ret+discount*target_q
                target_samples=scalar_y[:,None]
            scalar_targets.append(scalar_y)

        action=batch.actions[:,t].float(); current=beliefs[t]
        q1,q2=c.q_critic(current.vector.detach(),action)
        if getattr(c.q_critic,'distributional',False):
            l1=quantile_huber_loss(q1,target_samples,c.q_critic.taus)
            l2=quantile_huber_loss(q2,target_samples,c.q_critic.taus)
            losses.append(l1+l2)
        else:
            losses.append((q1-scalar_y.detach(),q2-scalar_y.detach()))

    all_targets=torch.cat([x.reshape(-1) for x in scalar_targets]); c.q_scale.update(all_targets); scale=c.q_scale.std.detach()
    if getattr(c.q_critic,'distributional',False):
        loss=torch.stack(losses).mean()/scale.clamp_min(1e-4)
    else:
        loss=torch.stack([F.smooth_l1_loss(a/scale,torch.zeros_like(a))+F.smooth_l1_loss(b/scale,torch.zeros_like(b)) for a,b in losses]).mean()
    q_opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(c.q_critic.parameters(),tc["grad_clip"]); q_opt.step(); c.target_q.update(c.q_critic,tau)
    return float(loss.item())


def _collect_action(c, belief, cfg, step: int, device):
    tc=cfg["training"]; eps=max(tc.get("epsilon_final",0.05),tc.get("epsilon_start",0.4)*(1-step/max(1,tc["steps"])))
    if torch.rand(())<eps:
        model_action=c.action_spec.random(1,device)
        if c.action_spec.kind=="discrete": env_action=int(model_action.argmax(-1).item())
        else: env_action=model_action.squeeze(0).detach().cpu().numpy()
        return model_action,env_action
    if c.action_spec.kind=="discrete":
        model_action,idx=c.actor.sample_onehot(belief.vector); return model_action,int(idx.item())
    model_action,_,_=c.actor.sample_with_log_prob(belief.vector); return model_action,model_action.squeeze(0).detach().cpu().numpy()


def train(cfg,device,out_dir="runs/default",resume_path=None):
    tc=cfg["training"]; env=make_environment(cfg,cfg["seed"]); c=build_components(cfg,device)
    value_ids={id(p) for p in c.world_model.value_head.parameters()}; wm_params=[p for p in c.world_model.parameters() if id(p) not in value_ids]+list(c.uncertainty.parameters())
    wm_opt=torch.optim.AdamW(wm_params,lr=tc["lr"]); actor_opt=torch.optim.AdamW(c.actor.parameters(),lr=tc.get("actor_lr",tc["lr"])); value_opt=torch.optim.AdamW(c.world_model.value_head.parameters(),lr=tc.get("value_lr",tc["lr"]))
    q_opt=torch.optim.AdamW(c.q_critic.parameters(),lr=tc.get("continuous_q",{}).get("lr",tc.get("value_lr",tc["lr"]))) if c.q_critic is not None else None
    modules={"world_model":c.world_model,"actor":c.actor,"uncertainty":c.uncertainty,"target_value":c.target_value}; opts={"world_model":wm_opt,"actor":actor_opt,"value":value_opt}
    if c.q_critic is not None: modules.update({"q_critic":c.q_critic,"target_q":c.target_q,"q_scale":c.q_scale}); opts["q_critic"]=q_opt
    cal_opt=None
    if c.calibrator is not None:
        modules["calibrator"]=c.calibrator
        if tc.get("auto_calibrate_uncertainty",False): cal_opt=torch.optim.Adam(c.calibrator.parameters(),lr=float(tc.get("calibration_lr",0.03))); opts["calibrator"]=cal_opt
    if c.arbitrator is not None: modules["arbitrator"]=c.arbitrator
    replay=PrioritizedEpisodeReplayBuffer(tc["replay_capacity"],tc.get("priority_alpha",0.6),tc.get("priority_beta",0.4)) if tc.get("prioritized_replay",False) else EpisodeReplayBuffer(tc["replay_capacity"])
    start=0; episode=0; ep_reward=0.0; obs=env.reset(); prev=c.action_spec.zero(1,device); belief=c.world_model.initial_belief(1,device)
    if resume_path:
        extra=CheckpointManager.load_bundle(resume_path,modules,opts,map_location=device,strict=False); start=int(extra.get("step",0)); episode=int(extra.get("episode",0)); ep_reward=float(extra.get("episode_reward",0))
        if "replay" in extra: replay.load_state_dict(extra["replay"])
        if "env" in extra and hasattr(env,"load_state_dict"): env.load_state_dict(extra["env"])
        if "obs" in extra: obs=extra["obs"]
        if "prev_action" in extra: prev=extra["prev_action"].to(device)
        if "belief" in extra:
            bd=extra["belief"]; belief=type(belief)(bd["deterministic"].to(device),bd["stochastic"].to(device),None if bd.get("slow") is None else bd["slow"].to(device))
    telemetry=JsonlTelemetry(Path(out_dir)/"metrics.jsonl")
    for step in range(start+1,tc["steps"]+1):
        o=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0); post=c.world_model.observe(belief,prev,o,step_index=step); belief=post.belief.detach()
        model_action,env_action=_collect_action(c,belief,cfg,step,device)
        nxt,reward,done,info=env.step(env_action); stored_action=env_action if c.action_spec.kind=="discrete" else np.asarray(env_action,dtype=np.float32).copy()
        replay.add(Transition(obs,stored_action,reward,nxt,done)); ep_reward+=reward
        if replay.can_sample(tc["batch_size"],tc["sequence_length"]):
            batch=replay.sample_sequences(tc["batch_size"],tc["sequence_length"],device); wm_loss,losses,starts,err,calibration=_sequence_world_loss(c,batch,cfg)
            wm_opt.zero_grad(set_to_none=True); wm_loss.backward(); torch.nn.utils.clip_grad_norm_(wm_params,tc["grad_clip"]); wm_opt.step()
            if isinstance(replay,PrioritizedEpisodeReplayBuffer) and batch.sequence_ids is not None: replay.update_priorities(batch.sequence_ids,err.cpu().numpy())
            q_loss=_continuous_q_update(c,batch,q_opt,cfg)
            if cal_opt is not None and step%int(tc.get("calibration_interval",25))==0 and calibration[0].numel()>3:
                u,e=calibration; threshold=torch.quantile(e,float(tc.get("calibration_quantile",0.75))); labels=(e>=threshold).float(); pred=c.calibrator(u); cal_loss=F.binary_cross_entropy(pred,labels); cal_opt.zero_grad(set_to_none=True); cal_loss.backward(); cal_opt.step()
            a_loss,v_loss,entropy=_imagination_update(c,starts[-1],actor_opt,value_opt,cfg)
            if step%tc.get("log_every",25)==0:
                telemetry.log(step=step,action_type=c.action_spec.kind,world_loss=float(wm_loss.item()),actor_loss=a_loss,critic_loss=v_loss,entropy=entropy,
                              latent_loss=float(losses["latent"].item()),overshoot_loss=float(losses["overshoot"].item()),reward_loss=float(losses["reward"].item()),
                              cont_loss=float(losses["cont"].item()),kl=float(losses["kl"].item()),uncertainty_loss=float(losses["uncertainty"].item()),routing_loss=float(losses["routing"].item()),q_critic_loss=q_loss)
        if done:
            telemetry.log(step=step,episode=episode,episode_reward=ep_reward,success=bool(info.get("success",ep_reward>0.5))); episode+=1; ep_reward=0.0; obs=env.reset(); belief=c.world_model.initial_belief(1,device); prev.zero_()
        else: obs=nxt; prev=model_action.detach()
        if step%tc["checkpoint_every"]==0:
            bstate={"deterministic":belief.deterministic.detach().cpu(),"stochastic":belief.stochastic.detach().cpu(),"slow":None if belief.slow is None else belief.slow.detach().cpu()}
            extra={"step":step,"version":"1.8.0","action_type":c.action_spec.kind,"episode":episode,"episode_reward":ep_reward,"obs":obs,"prev_action":prev.detach().cpu(),"belief":bstate}
            if hasattr(env,"state_dict"): extra["env"]=env.state_dict()
            if tc.get("checkpoint_replay",True): extra["replay"]=replay.state_dict()
            CheckpointManager.save_bundle(Path(out_dir)/f"checkpoint_{step}.pt",modules,opts,extra=extra)
    if hasattr(env,"close"): env.close()
    return c


@torch.no_grad()
def evaluate(cfg,components,device,episodes=25,use_planner=False,deterministic_actor=True):
    successes=0; rewards=[]; final_distances=[]
    for ep in range(episodes):
        env=make_environment(cfg,cfg["seed"]+1000+ep); obs=env.reset(); belief=components.world_model.initial_belief(1,device); prev=components.action_spec.zero(1,device); total=0.0; done=False; t=0; final_info={}
        while not done:
            o=torch.as_tensor(obs,dtype=torch.float32,device=device).unsqueeze(0); out=components.world_model.observe(belief,prev,o,step_index=t); belief=out.belief
            if use_planner and components.planner is not None:
                model_action,env_action,_=components.planner.plan(belief)
            elif components.action_spec.kind=="discrete":
                model_action,idx=components.actor.deterministic_action(belief.vector) if deterministic_actor else components.actor.sample_onehot(belief.vector); env_action=int(idx.item())
            else:
                if deterministic_actor: model_action=components.actor.deterministic_action(belief.vector)
                else: model_action,_,_=components.actor.sample_with_log_prob(belief.vector)
                env_action=model_action.squeeze(0).cpu().numpy()
            obs,reward,done,final_info=env.step(env_action); total+=reward; prev=model_action; t+=1
        if hasattr(env,"close"): env.close()
        rewards.append(total); successes+=int(bool(final_info.get("success",total>0.5)))
        if "distance" in final_info: final_distances.append(float(final_info["distance"]))
    result={"episodes":episodes,"success_rate":successes/episodes,"mean_reward":sum(rewards)/episodes,"action_type":components.action_spec.kind}
    if final_distances: result["mean_final_distance"]=sum(final_distances)/len(final_distances)
    return result
