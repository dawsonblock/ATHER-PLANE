from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import hashlib, json, math

from .empirical_closure import RunRecord
from .evidence_integrity import RunProvenance, build_integrity_receipt


def _sha(value: Any) -> str:
    raw=json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class ExperimentProtocol:
    """Predeclared empirical matrix. Qualification fails closed on missing or extra cells."""
    protocol_id: str
    systems: tuple[str,...]
    seeds: tuple[int,...]
    tasks: tuple[str,...]
    splits: tuple[str,...]
    milestones: tuple[int,...]
    primary_metric: str = "success_rate"
    minimum_seeds: int = 5
    system_config_sha256: tuple[tuple[str,str], ...] = ()

    def __post_init__(self):
        if not self.protocol_id: raise ValueError("protocol_id required")
        for name in ("systems","seeds","tasks","splits","milestones"):
            v=getattr(self,name)
            if not v or len(v)!=len(set(v)): raise ValueError(f"{name} must be non-empty and unique")
        if any(s not in {"train","validation","heldout","transfer"} for s in self.splits): raise ValueError("invalid split")
        if self.minimum_seeds < 2 or len(self.seeds) < self.minimum_seeds: raise ValueError("insufficient preregistered seeds")
        if any(m < 0 for m in self.milestones): raise ValueError("milestones must be non-negative")
        if self.system_config_sha256:
            keys=[str(k) for k,_ in self.system_config_sha256]
            if len(keys)!=len(set(keys)): raise ValueError("system_config_sha256 keys must be unique")
            if set(keys)!=set(self.systems): raise ValueError("system config hashes must bind every preregistered system")
            for name,digest in self.system_config_sha256:
                if len(str(digest))!=64 or any(c not in "0123456789abcdef" for c in str(digest).lower()):
                    raise ValueError(f"invalid system config sha256 for {name}")

    def cells(self) -> set[tuple[str,int,str,str,int]]:
        return {(s,seed,t,sp,m) for s in self.systems for seed in self.seeds for t in self.tasks for sp in self.splits for m in self.milestones}

    def to_dict(self) -> dict[str,Any]:
        d=asdict(self)
        for k in ("systems","seeds","tasks","splits","milestones"): d[k]=list(d[k])
        d["system_config_sha256"]={k:v for k,v in self.system_config_sha256}
        return d

    @property
    def sha256(self) -> str: return _sha(self.to_dict())


def record_key(r: RunRecord) -> tuple[str,int,str,str,int]:
    return (r.system,r.seed,r.task,r.split,r.transitions)


def verify_protocol_completeness(protocol: ExperimentProtocol, records: Iterable[RunRecord]) -> dict[str,Any]:
    rows=list(records); expected=protocol.cells(); actual=[record_key(r) for r in rows]
    counts={k:actual.count(k) for k in set(actual)}
    duplicate=sorted(k for k,n in counts.items() if n>1)
    aset=set(actual); missing=sorted(expected-aset); extra=sorted(aset-expected)
    failures=[]
    if missing: failures.append(f"missing preregistered cells: {len(missing)}")
    if extra: failures.append(f"unexpected cells: {len(extra)}")
    if duplicate: failures.append(f"duplicate cells: {len(duplicate)}")
    return {"status":"PASS" if not failures else "FAIL","expected_cells":len(expected),"observed_records":len(rows),
            "missing":[list(x) for x in missing],"extra":[list(x) for x in extra],"duplicates":[list(x) for x in duplicate],"failures":failures}


def verify_pairing(protocol: ExperimentProtocol, records: Iterable[RunRecord], *, baseline: str="full") -> dict[str,Any]:
    rows=list(records); failures=[]
    if baseline not in protocol.systems: failures.append("baseline not preregistered")
    by={(r.system,r.seed,r.task,r.split,r.transitions):r for r in rows}
    for system in protocol.systems:
        if system==baseline: continue
        for seed in protocol.seeds:
            for task in protocol.tasks:
                for split in protocol.splits:
                    for milestone in protocol.milestones:
                        b=(baseline,seed,task,split,milestone); c=(system,seed,task,split,milestone)
                        if b in by and c in by:
                            if protocol.primary_metric not in by[b].metrics or protocol.primary_metric not in by[c].metrics:
                                failures.append(f"primary metric missing for pair: {c}")
    return {"status":"PASS" if not failures else "FAIL","baseline":baseline,"failures":failures}


def verify_metric_semantics(records: Iterable[RunRecord]) -> dict[str,Any]:
    failures=[]
    for r in records:
        m=r.metrics; key=record_key(r)
        sr=m.get("success_rate")
        if sr is not None and not (0.0 <= float(sr) <= 1.0): failures.append(f"success_rate outside [0,1]: {key}")
        cv=m.get("constraint_violations")
        if cv is not None and float(cv) < 0: failures.append(f"negative constraint_violations: {key}")
        for name in ("planner_calls_per_episode","world_model_calls_per_episode","inference_latency_ms","wall_clock_seconds"):
            if name in m and float(m[name]) < 0: failures.append(f"negative {name}: {key}")
        if any(not math.isfinite(float(v)) for v in m.values()): failures.append(f"non-finite metric: {key}")
    return {"status":"PASS" if not failures else "FAIL","failures":failures}


def build_protocol_receipt(protocol: ExperimentProtocol, records: Iterable[RunRecord], provenance: dict[str,RunProvenance], *, baseline: str="full") -> dict[str,Any]:
    rows=list(records)
    completeness=verify_protocol_completeness(protocol,rows)
    pairing=verify_pairing(protocol,rows,baseline=baseline)
    metrics=verify_metric_semantics(rows)
    integrity=build_integrity_receipt(rows,provenance)
    body={"format":"awa-v2.19-experiment-protocol-v1","protocol":protocol.to_dict(),"protocol_sha256":protocol.sha256,
          "completeness":completeness,"pairing":pairing,"metric_semantics":metrics,"integrity":integrity}
    body["status"]="PASS" if all(x["status"]=="PASS" for x in (completeness,pairing,metrics,integrity)) else "FAIL"
    body["receipt_sha256"]=_sha(body)
    return body
