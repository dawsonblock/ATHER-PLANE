from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable
import hashlib, json, math
import numpy as np

from .evaluation_stats import bootstrap_ci, paired_bootstrap_ci

REQUIRED_METRICS = (
    "success_rate", "episode_return", "constraint_violations",
    "planner_calls_per_episode", "world_model_calls_per_episode",
    "inference_latency_ms", "wall_clock_seconds",
)

@dataclass(frozen=True)
class RunRecord:
    system: str
    seed: int
    task: str
    split: str
    transitions: int
    metrics: dict[str, float]
    checkpoint_sha256: str = ""
    config_sha256: str = ""

    def __post_init__(self):
        if not self.system or not self.task: raise ValueError("system and task are required")
        if self.split not in {"train", "validation", "heldout", "transfer"}: raise ValueError("invalid split")
        if self.transitions < 0: raise ValueError("transitions must be non-negative")
        for key, value in self.metrics.items():
            if not math.isfinite(float(value)): raise ValueError(f"non-finite metric: {key}")

    def to_dict(self): return asdict(self)


def file_sha256(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: str | Path) -> list[RunRecord]:
    rows=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(),1):
        if not line.strip(): continue
        raw=json.loads(line)
        try: rows.append(RunRecord(**raw))
        except Exception as exc: raise ValueError(f"invalid record at line {n}: {exc}") from exc
    if not rows: raise ValueError("no run records")
    return rows


def _key(r: RunRecord): return (r.seed,r.task,r.split,r.transitions)


def summarize_runs(records: Iterable[RunRecord], *, primary_metric="success_rate", baseline="full", resamples=2000) -> dict[str,Any]:
    rows=list(records)
    if not rows: raise ValueError("records required")
    systems=sorted({r.system for r in rows})
    if baseline not in systems: raise ValueError("baseline system missing")
    grouped={s:[r for r in rows if r.system==s] for s in systems}
    base={_key(r):r for r in grouped[baseline]}
    summary={}
    for system in systems:
        vals=[]; base_vals=[]; missing=[]
        for r in grouped[system]:
            if primary_metric not in r.metrics: raise ValueError(f"{system} missing {primary_metric}")
            vals.append(float(r.metrics[primary_metric]))
            if system != baseline:
                b=base.get(_key(r))
                if b is None or primary_metric not in b.metrics: missing.append(_key(r))
                else: base_vals.append(float(b.metrics[primary_metric]))
        ci=bootstrap_ci(vals,resamples=resamples,seed=1700+len(system))
        item={"primary":ci.to_dict(),"runs":len(vals)}
        if system != baseline:
            if missing or len(base_vals)!=len(vals):
                item["paired_delta_vs_full"]={"status":"not_comparable","missing_pairs":[list(x) for x in missing]}
            else:
                d=paired_bootstrap_ci(vals,base_vals,resamples=resamples,seed=1717+len(system))
                item["paired_delta_vs_full"]={"status":"ok",**d.to_dict()}
        summary[system]=item
    return {"format":"awa-v2.17-empirical-closure-v1","primary_metric":primary_metric,"baseline":baseline,"systems":systems,"summary":summary}


def planner_dependence(records: Iterable[RunRecord], *, system="full") -> list[dict[str,float]]:
    rows=[r for r in records if r.system==system and "planner_calls_per_episode" in r.metrics]
    by={}
    for r in rows: by.setdefault(r.transitions,[]).append(float(r.metrics["planner_calls_per_episode"]))
    return [{"transitions":int(t),"mean_planner_calls_per_episode":float(np.mean(v)),"n":len(v)} for t,v in sorted(by.items())]


def qualification_gates(records: Iterable[RunRecord], *, minimum_seeds=5, required_metrics=REQUIRED_METRICS) -> dict[str,Any]:
    rows=list(records); systems=sorted({r.system for r in rows}); failures=[]
    for system in systems:
        seeds={r.seed for r in rows if r.system==system}
        if len(seeds)<minimum_seeds: failures.append(f"{system}: only {len(seeds)} seeds; require {minimum_seeds}")
        for r in [x for x in rows if x.system==system]:
            missing=[m for m in required_metrics if m not in r.metrics]
            if missing: failures.append(f"{system}/{r.seed}/{r.task}/{r.transitions}: missing {','.join(missing)}")
    heldout=any(r.split in {"heldout","transfer"} for r in rows)
    if not heldout: failures.append("no heldout or transfer evaluation records")
    return {"status":"PASS" if not failures else "FAIL","minimum_seeds":minimum_seeds,"systems":systems,"heldout_present":heldout,"failures":failures}


def build_evidence_bundle(records: Iterable[RunRecord], *, primary_metric="success_rate", minimum_seeds=5) -> dict[str,Any]:
    rows=list(records)
    return {
        "format":"awa-v2.17-evidence-bundle-v1",
        "qualification":qualification_gates(rows,minimum_seeds=minimum_seeds),
        "comparison":summarize_runs(rows,primary_metric=primary_metric),
        "planner_dependence":planner_dependence(rows),
        "records":[r.to_dict() for r in rows],
    }
