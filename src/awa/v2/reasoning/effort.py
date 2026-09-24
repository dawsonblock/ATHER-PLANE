from __future__ import annotations
from dataclasses import dataclass, asdict
import math
import numpy as np
import torch

@dataclass(frozen=True)
class EffortLevel:
    name:str
    planner:str
    budget:int
    normalized_compute:float

@dataclass(frozen=True)
class EffortDecision:
    level:EffortLevel
    expected_gain:float
    expected_cost:float
    voc:float
    def to_dict(self):
        d=asdict(self); return d

class ReasoningEffortController:
    """Choose a planner/compute budget from predicted benefit minus compute cost.

    The predictor can be an Aether VOC model exposing predict(features)->[B,K], or a
    simple callable returning gains for each effort level.
    """
    def __init__(self,levels:list[EffortLevel],gain_predictor=None,cost_lambda=1.0):
        if not levels: raise ValueError('levels required')
        self.levels=list(levels); self.gain_predictor=gain_predictor; self.cost_lambda=float(cost_lambda)

    def choose(self,features, predicted_gains=None):
        if predicted_gains is None:
            if self.gain_predictor is None: raise ValueError('predicted_gains or gain_predictor required')
            predicted_gains=self.gain_predictor(features)
        if isinstance(predicted_gains, torch.Tensor):
            predicted_gains = predicted_gains.detach().cpu().numpy()
        g=np.asarray(predicted_gains,dtype=np.float32).reshape(-1)
        if g.size!=len(self.levels): raise ValueError('gain count must equal effort levels')
        voc=[]
        for gain,lvl in zip(g,self.levels): voc.append(float(gain)-self.cost_lambda*float(lvl.normalized_compute))
        i=int(np.argmax(voc)); lvl=self.levels[i]
        return EffortDecision(lvl,float(g[i]),self.cost_lambda*float(lvl.normalized_compute),float(voc[i]))

    @classmethod
    def default(cls,cost_lambda=.05):
        return cls([
            EffortLevel('reflex','actor',0,0.0),
            EffortLevel('shallow','policy_mppi',32,.25),
            EffortLevel('medium','policy_mppi',128,.50),
            EffortLevel('deep','policy_icem',256,.75),
            EffortLevel('strategic','beam',512,1.0),
        ],cost_lambda=cost_lambda)


@dataclass(frozen=True)
class EffortSignals:
    """State used to allocate deliberation compute without changing task semantics."""
    uncertainty: float
    novelty: float
    risk: float
    actor_confidence: float
    historical_planner_benefit: float
    goal_difficulty: float
    model_reliability: float = 1.0
    resource_pressure: float = 0.0

    def __post_init__(self):
        vals=asdict(self)
        if not all(math.isfinite(float(v)) for v in vals.values()):
            raise ValueError('effort signals must be finite')
        for name in ('novelty','risk','actor_confidence','goal_difficulty','model_reliability','resource_pressure'):
            if not 0.0 <= float(getattr(self,name)) <= 1.0:
                raise ValueError(f'{name} must lie in [0,1]')
        if self.uncertainty < 0:
            raise ValueError('uncertainty must be >= 0')

    def vector(self) -> np.ndarray:
        return np.asarray([
            self.uncertainty,self.novelty,self.risk,self.actor_confidence,
            self.historical_planner_benefit,self.goal_difficulty,
            self.model_reliability,self.resource_pressure,
        ],dtype=np.float32)


@dataclass(frozen=True)
class AdaptiveEffortDecision:
    level: EffortLevel
    expected_gain: float
    expected_risk: float
    expected_latency_ms: float
    compute_penalty: float
    latency_penalty: float
    risk_penalty: float
    deliberation_bonus: float
    utility: float
    masked_levels: tuple[str,...]

    def to_dict(self):
        return asdict(self)


class AdaptiveReasoningEffortController:
    """Allocate a graded reasoning budget from gain, risk, latency and reliability.

    This is intentionally a compute controller, not a safety bypass. The downstream
    SafeActionGuard/risk limits remain authoritative. Low world-model reliability
    can mask model-based planning entirely.
    """
    def __init__(
        self,
        levels:list[EffortLevel] | None = None,
        *,
        compute_lambda:float=.05,
        latency_lambda:float=.001,
        risk_lambda:float=.25,
        risk_deliberation_bonus:float=.10,
        minimum_model_reliability:float=.25,
    ):
        self.levels=list(levels or ReasoningEffortController.default().levels)
        if not self.levels: raise ValueError('levels required')
        for x in (compute_lambda,latency_lambda,risk_lambda,risk_deliberation_bonus,minimum_model_reliability):
            if not math.isfinite(float(x)) or float(x)<0: raise ValueError('controller penalties must be finite and >= 0')
        if minimum_model_reliability>1: raise ValueError('minimum_model_reliability must be <= 1')
        self.compute_lambda=float(compute_lambda); self.latency_lambda=float(latency_lambda)
        self.risk_lambda=float(risk_lambda); self.risk_deliberation_bonus=float(risk_deliberation_bonus)
        self.minimum_model_reliability=float(minimum_model_reliability)

    def choose(self, signals:EffortSignals, predicted_gains, predicted_risks=None, predicted_latencies_ms=None) -> AdaptiveEffortDecision:
        gains=np.asarray(predicted_gains,dtype=np.float64).reshape(-1)
        n=len(self.levels)
        if gains.size!=n or not np.all(np.isfinite(gains)): raise ValueError('predicted_gains must be finite and match effort levels')
        risks=np.full(n,float(signals.risk),dtype=np.float64) if predicted_risks is None else np.asarray(predicted_risks,dtype=np.float64).reshape(-1)
        lat=np.asarray([lvl.normalized_compute*10.0 for lvl in self.levels],dtype=np.float64) if predicted_latencies_ms is None else np.asarray(predicted_latencies_ms,dtype=np.float64).reshape(-1)
        if risks.size!=n or lat.size!=n: raise ValueError('risk/latency counts must match effort levels')
        masked=[]; utilities=[]; pieces=[]
        for i,lvl in enumerate(self.levels):
            risk=float(risks[i]); latency=float(lat[i])
            if not math.isfinite(risk) or not math.isfinite(latency) or risk<0 or latency<0:
                masked.append(lvl.name); utilities.append(float('-inf')); pieces.append((0,0,0,0)); continue
            if lvl.planner!='actor' and signals.model_reliability < self.minimum_model_reliability:
                masked.append(lvl.name); utilities.append(float('-inf')); pieces.append((0,0,0,0)); continue
            resource_multiplier=1.0+float(signals.resource_pressure)
            cp=self.compute_lambda*float(lvl.normalized_compute)*resource_multiplier
            lp=self.latency_lambda*latency
            rp=self.risk_lambda*risk
            # More deliberate levels receive a small bonus in high-risk states only
            # when the model is reliable. Safety still executes after this choice.
            db=(self.risk_deliberation_bonus*float(signals.risk)*float(signals.model_reliability)*float(lvl.normalized_compute))
            utility=float(gains[i])-cp-lp-rp+db
            utilities.append(utility); pieces.append((cp,lp,rp,db))
        if not any(math.isfinite(u) for u in utilities):
            raise RuntimeError('no valid reasoning-effort level')
        idx=int(np.argmax(np.asarray(utilities,dtype=np.float64)))
        cp,lp,rp,db=pieces[idx]
        return AdaptiveEffortDecision(self.levels[idx],float(gains[idx]),float(risks[idx]),float(lat[idx]),float(cp),float(lp),float(rp),float(db),float(utilities[idx]),tuple(masked))


@dataclass(frozen=True)
class EffortOutcome:
    level_name:str
    planner:str
    budget:int
    predicted_gain:float
    realized_gain:float
    world_model_calls:int
    latency_ms:float
    risk:float
    novelty:float
    success:bool


class EffortOutcomeLedger:
    """Training evidence for learning planner usefulness and measuring dependence."""
    def __init__(self): self.records:list[EffortOutcome]=[]
    def add(self,record:EffortOutcome):
        vals=(record.predicted_gain,record.realized_gain,record.latency_ms,record.risk,record.novelty)
        if not all(math.isfinite(float(v)) for v in vals): raise ValueError('effort outcome must be finite')
        if record.world_model_calls<0 or record.latency_ms<0: raise ValueError('compute measurements must be non-negative')
        self.records.append(record)
    def summary(self):
        if not self.records: return {'samples':0,'planner_fraction':0.0,'mean_realized_gain':0.0,'mean_world_model_calls':0.0}
        n=len(self.records); planner=sum(r.planner!='actor' for r in self.records)
        return {
            'samples':n,
            'planner_fraction':float(planner/n),
            'mean_realized_gain':float(np.mean([r.realized_gain for r in self.records])),
            'mean_world_model_calls':float(np.mean([r.world_model_calls for r in self.records])),
            'success_rate':float(np.mean([bool(r.success) for r in self.records])),
        }
    def training_arrays(self):
        if not self.records: raise ValueError('empty effort outcome ledger')
        x=np.asarray([[r.risk,r.novelty,r.world_model_calls/1000.0,r.latency_ms/1000.0] for r in self.records],dtype=np.float32)
        y=np.asarray([r.realized_gain for r in self.records],dtype=np.float32)
        return x,y
