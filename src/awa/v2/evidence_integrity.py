from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable
import hashlib, json, re

from .empirical_closure import RunRecord

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class RunProvenance:
    run_id: str
    source_sha256: str
    dataset_sha256: str
    environment_sha256: str
    dependency_lock_sha256: str
    scenario_ids: tuple[str, ...]

    def __post_init__(self):
        if not self.run_id: raise ValueError("run_id required")
        for name in ("source_sha256", "dataset_sha256", "environment_sha256", "dependency_lock_sha256"):
            if not _SHA256.fullmatch(getattr(self, name)): raise ValueError(f"{name} must be lowercase sha256")
        if not self.scenario_ids or len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("scenario_ids must be non-empty and unique")

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self); out["scenario_ids"] = list(self.scenario_ids); return out


def verify_record_integrity(records: Iterable[RunRecord], provenance: dict[str, RunProvenance], *, require_artifact_hashes: bool=True) -> dict[str, Any]:
    rows=list(records); failures=[]; seen=set()
    for r in rows:
        key=(r.system,r.seed,r.task,r.split,r.transitions)
        if key in seen: failures.append(f"duplicate run key: {key}")
        seen.add(key)
        run_id="|".join(map(str,key))
        p=provenance.get(run_id)
        if p is None: failures.append(f"missing provenance: {run_id}")
        if require_artifact_hashes:
            if not _SHA256.fullmatch(r.checkpoint_sha256): failures.append(f"invalid checkpoint_sha256: {run_id}")
            if not _SHA256.fullmatch(r.config_sha256): failures.append(f"invalid config_sha256: {run_id}")
    return {"status":"PASS" if not failures else "FAIL", "records":len(rows), "failures":failures}


def detect_split_leakage(provenance: dict[str, RunProvenance], records: Iterable[RunRecord]) -> dict[str, Any]:
    split_scenarios: dict[str,set[str]]={}
    failures=[]
    for r in records:
        run_id="|".join(map(str,(r.system,r.seed,r.task,r.split,r.transitions)))
        p=provenance.get(run_id)
        if p is None: continue
        split_scenarios.setdefault(r.split,set()).update(p.scenario_ids)
    train=split_scenarios.get("train",set()) | split_scenarios.get("validation",set())
    evals=split_scenarios.get("heldout",set()) | split_scenarios.get("transfer",set())
    overlap=sorted(train & evals)
    if overlap: failures.append(f"train/eval scenario leakage: {','.join(overlap)}")
    return {"status":"PASS" if not failures else "FAIL", "overlap":overlap, "failures":failures}


def build_integrity_receipt(records: Iterable[RunRecord], provenance: dict[str, RunProvenance]) -> dict[str, Any]:
    rows=list(records)
    integrity=verify_record_integrity(rows,provenance)
    leakage=detect_split_leakage(provenance,rows)
    body={
        "format":"awa-v2.18-evidence-integrity-v1",
        "integrity":integrity,
        "split_leakage":leakage,
        "records_sha256":canonical_sha256([r.to_dict() for r in rows]),
        "provenance_sha256":canonical_sha256({k:v.to_dict() for k,v in sorted(provenance.items())}),
    }
    body["receipt_sha256"]=canonical_sha256(body)
    body["status"]="PASS" if integrity["status"]==leakage["status"]=="PASS" else "FAIL"
    return body
