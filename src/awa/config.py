from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import yaml


class ConfigError(ValueError): pass

def _require(mapping: dict, key: str, path: str):
    if key not in mapping: raise ConfigError(f"Missing required config key: {path}.{key}")
    return mapping[key]


def validate_config(cfg: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(cfg, dict): raise ConfigError("Configuration must be a mapping.")
    cfg=copy.deepcopy(cfg)
    for section in ("env","model","planner","training"):
        if section not in cfg or not isinstance(cfg[section],dict): raise ConfigError(f"Missing required mapping: {section}")
    env,model,planner,training=cfg["env"],cfg["model"],cfg["planner"],cfg["training"]
    action_type=env.get("action_type","discrete")
    if action_type not in {"discrete","continuous"}: raise ConfigError("env.action_type must be 'discrete' or 'continuous'")
    env["action_type"]=action_type
    for k in ("obs_dim","action_dim","horizon"):
        v=_require(env,k,"env")
        if not isinstance(v,int) or v<=0: raise ConfigError(f"env.{k} must be a positive integer")
    if action_type=="continuous":
        dim=env["action_dim"]; lo=env.get("action_low",[-1.0]*dim); hi=env.get("action_high",[1.0]*dim)
        if isinstance(lo,(int,float)): lo=[float(lo)]*dim
        if isinstance(hi,(int,float)): hi=[float(hi)]*dim
        if len(lo)!=dim or len(hi)!=dim: raise ConfigError("continuous action bounds must match env.action_dim")
        if any(float(h)<=float(l) for l,h in zip(lo,hi)): raise ConfigError("each continuous action_high must exceed action_low")
        env["action_low"]=[float(x) for x in lo]; env["action_high"]=[float(x) for x in hi]
    for k in ("obs_dim","action_dim","latent_dim","deterministic_dim","stochastic_dim","hidden_dim"):
        v=_require(model,k,"model")
        if not isinstance(v,int) or v<=0: raise ConfigError(f"model.{k} must be a positive integer")
    if model["obs_dim"]!=env["obs_dim"]: raise ConfigError("model.obs_dim must equal env.obs_dim")
    if model["action_dim"]!=env["action_dim"]: raise ConfigError("model.action_dim must equal env.action_dim")
    if model.get("dynamics","gru") not in {"gru","ssm"}: raise ConfigError("model.dynamics must be 'gru' or 'ssm' in the online v1.8 runtime")
    if model.get("slow_enabled",False) and (int(model.get("slow_dim",0))<=0 or int(model.get("slow_stride",0))<=0): raise ConfigError("slow_dim and slow_stride must be positive when slow_enabled=true")
    if model.get("experts_enabled",False):
        if int(model.get("experts_count",0))<2: raise ConfigError("experts_count must be >=2 when experts_enabled=true")
        top_k=int(model.get("experts_top_k",1))
        if top_k<=0 or top_k>int(model["experts_count"]): raise ConfigError("experts_top_k must be in [1, experts_count]")
    for k in ("steps","batch_size","sequence_length","replay_capacity"):
        v=_require(training,k,"training")
        if not isinstance(v,int) or v<=0: raise ConfigError(f"training.{k} must be a positive integer")
    if not 0<float(training.get("discount",0.99))<=1: raise ConfigError("training.discount must be in (0,1]")
    lam=float(training.get("lambda",0.95))
    if not 0<=lam<=1: raise ConfigError("training.lambda must be in [0,1]")
    if training.get("prioritized_replay",False):
        alpha=float(training.get("priority_alpha",0.6)); beta=float(training.get("priority_beta",0.4))
        if not 0<=alpha<=1 or not 0<=beta<=1: raise ConfigError("priority_alpha and priority_beta must be in [0,1]")
    if planner.get("enabled",False):
        for k in ("horizon","candidates","elites","iterations"):
            v=_require(planner,k,"planner")
            if not isinstance(v,int) or v<=0: raise ConfigError(f"planner.{k} must be a positive integer")
        if planner["elites"]>planner["candidates"]: raise ConfigError("planner.elites cannot exceed planner.candidates")
        if action_type=="continuous" and float(planner.get("min_std",0.03))<=0: raise ConfigError("planner.min_std must be positive")
    burn=int(training.get("burn_in",0))
    if burn<0 or burn>=int(training["sequence_length"]): raise ConfigError("training.burn_in must be >=0 and < sequence_length")
    if int(training.get("overshoot_horizon",1))<1: raise ConfigError("training.overshoot_horizon must be >=1")
    cq=training.get("continuous_q",{})
    if cq.get("enabled",False):
        if action_type!="continuous": raise ConfigError("training.continuous_q requires env.action_type=continuous")
        if float(cq.get("target_tau",0.01))<=0 or float(cq.get("target_tau",0.01))>1: raise ConfigError("continuous_q.target_tau must be in (0,1]")
        if float(cq.get("actor_coef",0.25))<0: raise ConfigError("continuous_q.actor_coef must be >=0")
        ctype=str(cq.get("critic_type","twin_q"))
        if ctype not in {"twin_q","quantile_twin_q"}: raise ConfigError("continuous_q.critic_type must be twin_q or quantile_twin_q")
        if int(cq.get("n_step",1))<1: raise ConfigError("continuous_q.n_step must be >=1")
        if ctype=="quantile_twin_q" and int(cq.get("quantiles",32))<4: raise ConfigError("continuous_q.quantiles must be >=4")
    if training.get("auto_calibrate_uncertainty",False) and not planner.get("calibrated_uncertainty",False):
        raise ConfigError("auto_calibrate_uncertainty requires planner.calibrated_uncertainty=true")
    cfg.setdefault("seed",0); cfg.setdefault("device","auto")
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path,"r",encoding="utf-8") as f: cfg=yaml.safe_load(f)
    return validate_config(cfg)
