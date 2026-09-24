from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
import copy,json,time
import numpy as np
import torch
from torch.utils.data import Subset

from .world import MultimodalWorldModel
from .modular_world import ModularWorldModel,make_compute_matched_pair,monolithic_compute_profile
from .game.belief import GameBeliefEncoder,GameBeliefSequenceDataset,GameBeliefWorldTrainer
from .datasets import OfflineTransitionDataset
from .experiment_ledger import ExperimentLedger
from .evaluation_stats import bootstrap_ci


def resolve_device(device:str="auto"):
    if device!="auto": return device
    if torch.cuda.is_available(): return "cuda"
    if hasattr(torch.backends,"mps") and torch.backends.mps.is_available(): return "mps"
    return "cpu"


@dataclass(frozen=True)
class ScalingPoint:
    axis:str; value:str; model_kind:str; seed:int; train_sequences:int; eval_sequences:int
    total_parameters:int; active_parameters_approx:int; trunk_active_macs:int; train_seconds:float
    final_loss:float; horizon_rmse:dict; reward_rmse:float; value_rmse:float
    context:dict|None=None
    def to_dict(self): return asdict(self)


@dataclass(frozen=True)
class ScalingReport:
    format:str; dataset_sha256:str; device:str; precision:str; points:list[dict]; summary:dict
    def to_dict(self): return asdict(self)


def summarize_scaling_points(points,*,confidence=.95,resamples=2000,seed=0):
    groups={}
    for p in points:
        row=p.to_dict() if hasattr(p,"to_dict") else dict(p)
        key=(row["axis"],row["value"],row["model_kind"])
        groups.setdefault(key,[]).append(row)
    out={}
    for (axis,value,kind),rows in sorted(groups.items()):
        name=f"{axis}:{value}:{kind}"; metrics={}
        for metric in ("final_loss","reward_rmse","value_rmse","train_seconds"):
            metrics[metric]=bootstrap_ci([r[metric] for r in rows],confidence=confidence,resamples=resamples,seed=seed).to_dict()
        horizons=sorted({h for r in rows for h in r.get("horizon_rmse",{})},key=lambda x:int(x))
        metrics["horizon_rmse"]={h:bootstrap_ci([r["horizon_rmse"][h] for r in rows if h in r["horizon_rmse"]],confidence=confidence,resamples=resamples,seed=seed).to_dict() for h in horizons}
        metrics["total_parameters"]=rows[0]["total_parameters"]; metrics["active_parameters_approx"]=rows[0]["active_parameters_approx"]; metrics["trunk_active_macs"]=rows[0]["trunk_active_macs"]
        out[name]=metrics
    return out


class ControlledScalingRunner:
    """Matched-compute model/data/context sweeps on one declared game dataset."""
    def __init__(self,transitions:OfflineTransitionDataset,*,device="auto",precision="fp32",seed=0,components=3,horizons=(1,2,4),lr=7e-4):
        self.transitions=transitions; self.device=resolve_device(device); self.precision=str(precision); self.seed=int(seed)
        self.components=int(components); self.horizons=tuple(int(h) for h in horizons); self.lr=float(lr)

    def _encoder(self,*,event_slots=8,global_stride=4):
        return GameBeliefEncoder(latent_dim=32,local_dim=48,global_dim=48,goal_latent_dim=16,event_slots=int(event_slots),global_stride=int(global_stride)).to(self.device)

    def _subset(self,seq,fraction,seed):
        n=max(1,min(len(seq),int(round(len(seq)*float(fraction)))))
        rng=np.random.default_rng(int(seed)); ids=np.sort(rng.choice(len(seq),size=n,replace=False)).tolist()
        return Subset(seq,ids)

    def _train_one(self,seq,eval_seq,encoder,*,kind,hidden,epochs,batch_size,seed,experts=4,top_k=1,train_subset=None,context=None):
        torch.manual_seed(int(seed)); np.random.seed(int(seed))
        if kind=="monolithic":
            world=MultimodalWorldModel(encoder.belief_dim,4,hidden=int(hidden),components=self.components,horizons=self.horizons).to(self.device)
            profile=monolithic_compute_profile(world)
        elif kind=="moe":
            _,world,_,profile=make_compute_matched_pair(encoder.belief_dim,4,hidden=int(hidden),components=self.components,horizons=self.horizons,experts=int(experts),top_k=int(top_k))
            world=world.to(self.device)
        else: raise ValueError("kind must be monolithic or moe")
        ds=train_subset or seq
        trainer=GameBeliefWorldTrainer(encoder,world,lr=self.lr,precision=self.precision)
        start=time.perf_counter(); hist=trainer.fit(ds,epochs=int(epochs),batch_size=min(int(batch_size),len(ds))); elapsed=time.perf_counter()-start
        metrics=trainer.evaluate_horizons(eval_seq,self.horizons,batch_size=min(int(batch_size),len(eval_seq)))
        return ScalingPoint("","",kind,int(seed),len(ds),len(eval_seq),profile.total_parameters,profile.active_parameters_approx,profile.trunk_active_macs,float(elapsed),float(hist[-1].loss if hist else 0),metrics["horizon_rmse"],float(metrics["reward_rmse"]),float(metrics["value_rmse"]),context)

    def model_pair(self,*,sequence_length=4,hidden=64,epochs=1,batch_size=32,experts=4,top_k=1,seed=None):
        seed=self.seed if seed is None else int(seed); seq=GameBeliefSequenceDataset(self.transitions,int(sequence_length))
        base=self._encoder(); state=copy.deepcopy(base.state_dict())
        rows=[]
        for kind in ("monolithic","moe"):
            enc=self._encoder(); enc.load_state_dict(state)
            p=self._train_one(seq,seq,enc,kind=kind,hidden=hidden,epochs=epochs,batch_size=batch_size,seed=seed,experts=experts,top_k=top_k)
            rows.append(ScalingPoint("architecture","matched_active_compute",**{k:v for k,v in p.to_dict().items() if k not in {"axis","value"}}))
        return rows

    def data_curve(self,*,fractions=(.25,.5,1.0),sequence_length=4,hidden=64,epochs=1,batch_size=32,kind="monolithic",experts=4,top_k=1,seed=None):
        seed=self.seed if seed is None else int(seed); seq=GameBeliefSequenceDataset(self.transitions,int(sequence_length)); base=self._encoder(); state=copy.deepcopy(base.state_dict()); rows=[]
        for i,f in enumerate(fractions):
            enc=self._encoder(); enc.load_state_dict(state); subset=self._subset(seq,float(f),seed+i)
            p=self._train_one(seq,seq,enc,kind=kind,hidden=hidden,epochs=epochs,batch_size=batch_size,seed=seed,experts=experts,top_k=top_k,train_subset=subset)
            rows.append(ScalingPoint("data_fraction",f"{float(f):.4f}",**{k:v for k,v in p.to_dict().items() if k not in {"axis","value"}}))
        return rows

    def model_curve(self,*,hidden_sizes=(32,64,128),sequence_length=4,epochs=1,batch_size=32,kinds=("monolithic","moe"),experts=4,top_k=1,seed=None):
        seed=self.seed if seed is None else int(seed); seq=GameBeliefSequenceDataset(self.transitions,int(sequence_length)); base=self._encoder(); state=copy.deepcopy(base.state_dict()); rows=[]
        for hidden in hidden_sizes:
            for kind in kinds:
                enc=self._encoder(); enc.load_state_dict(state)
                p=self._train_one(seq,seq,enc,kind=kind,hidden=int(hidden),epochs=epochs,batch_size=batch_size,seed=seed,experts=experts,top_k=top_k)
                rows.append(ScalingPoint("model_hidden",str(int(hidden)),**{k:v for k,v in p.to_dict().items() if k not in {"axis","value"}}))
        return rows

    def context_curve(self,*,specs=((2,8,2),(4,8,4),(8,12,4)),hidden=64,epochs=1,batch_size=32,kind="monolithic",experts=4,top_k=1,seed=None):
        seed=self.seed if seed is None else int(seed); rows=[]
        for seq_len,event_slots,stride in specs:
            seq=GameBeliefSequenceDataset(self.transitions,int(seq_len)); enc=self._encoder(event_slots=event_slots,global_stride=stride)
            ctx={"sequence_length":int(seq_len),"event_slots":int(event_slots),"global_stride":int(stride)}
            p=self._train_one(seq,seq,enc,kind=kind,hidden=hidden,epochs=epochs,batch_size=batch_size,seed=seed,experts=experts,top_k=top_k,context=ctx)
            rows.append(ScalingPoint("context",f"seq{seq_len}-slots{event_slots}-stride{stride}",**{k:v for k,v in p.to_dict().items() if k not in {"axis","value"}}))
        return rows


def run_controlled_scaling_experiment(dataset_path:str|Path,out_dir:str|Path,config:dict):
    transitions=OfflineTransitionDataset(dataset_path); out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    ledger=ExperimentLedger(out,config,experiment_id="controlled_modular_scaling")
    runtime=config.get("runtime",{}); model=config.get("model",{}); train=config.get("training",{})
    runner=ControlledScalingRunner(transitions,device=runtime.get("device","auto"),precision=runtime.get("precision","fp32"),seed=int(config.get("seed",0)),components=int(model.get("components",3)),horizons=tuple(model.get("horizons",[1,2,4])),lr=float(train.get("lr",7e-4)))
    common=dict(hidden=int(model.get("hidden",64)),epochs=int(train.get("epochs",1)),batch_size=int(train.get("batch_size",32)),experts=int(model.get("experts",4)),top_k=int(model.get("top_k",1)))
    seeds=[int(x) for x in config.get("qualification",{}).get("seeds",[config.get("seed",0)])]
    points=[]
    def across(fn):
        rows=[]
        for sd in seeds: rows.extend(fn(sd))
        return rows
    def phase(name,fn):
        result=ledger.run(name,lambda:{"points":[p.to_dict() for p in fn()]}); points.extend(result["points"])
    phase("architecture",lambda:across(lambda sd:runner.model_pair(sequence_length=int(train.get("sequence_length",4)),seed=sd,**common)))
    phase("data_curve",lambda:across(lambda sd:runner.data_curve(fractions=tuple(config.get("data",{}).get("fractions",[.25,.5,1.0])),sequence_length=int(train.get("sequence_length",4)),kind=str(config.get("data",{}).get("model_kind","monolithic")),seed=sd,**common)))
    phase("model_curve",lambda:across(lambda sd:runner.model_curve(hidden_sizes=tuple(config.get("scaling",{}).get("hidden_sizes",[32,64,128])),sequence_length=int(train.get("sequence_length",4)),epochs=common["epochs"],batch_size=common["batch_size"],experts=common["experts"],top_k=common["top_k"],seed=sd)))
    specs=tuple(tuple(int(x) for x in sp) for sp in config.get("context",{}).get("specs",[[2,4,2],[4,8,4],[8,12,4]]))
    phase("context_curve",lambda:across(lambda sd:runner.context_curve(specs=specs,kind=str(config.get("context",{}).get("model_kind","monolithic")),seed=sd,**common)))
    qual=config.get("qualification",{})
    summary=summarize_scaling_points(points,confidence=float(qual.get("confidence",.95)),resamples=int(qual.get("bootstrap_resamples",2000)),seed=int(config.get("seed",0)))
    report=ScalingReport("awa-v2.12-controlled-scaling-v1",transitions.manifest.sha256,runner.device,runner.precision,points,summary)
    (out/"scaling_report.json").write_text(json.dumps(report.to_dict(),indent=2,sort_keys=True)+"\n",encoding="utf-8")
    return report,ledger.summary()
