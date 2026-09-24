from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from contextlib import contextmanager
import hashlib,json,time,traceback


def _canonical_hash(obj)->str:
    raw=json.dumps(obj,sort_keys=True,separators=(",",":"),default=str).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclass
class PhaseRecord:
    status:str="pending"
    started_at:float|None=None
    finished_at:float|None=None
    result:dict|None=None
    error:str|None=None
    def to_dict(self): return asdict(self)


class ExperimentLedger:
    """Atomic, resumable experiment-state ledger bound to an immutable config hash."""
    def __init__(self,root:str|Path,config:dict,*,experiment_id:str="experiment"):
        self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/"experiment_state.json"; self.config=dict(config); self.config_hash=_canonical_hash(self.config)
        if self.path.exists():
            self.state=json.loads(self.path.read_text())
            if self.state.get("config_hash")!=self.config_hash:
                raise ValueError("experiment config changed; use a new output directory or restore the original config")
        else:
            self.state={"format":"awa-v2.12-experiment-ledger-v1","experiment_id":str(experiment_id),"config_hash":self.config_hash,"config":self.config,"created_at":time.time(),"phases":{}}
            self._save()

    def _save(self):
        tmp=self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state,indent=2,sort_keys=True)+"\n",encoding="utf-8")
        tmp.replace(self.path)

    def phase(self,name:str):
        raw=self.state["phases"].get(name,{})
        return PhaseRecord(**{k:raw.get(k) for k in PhaseRecord.__dataclass_fields__})

    def completed(self,name:str)->bool:
        return self.phase(name).status=="completed"

    def run(self,name:str,fn):
        if self.completed(name): return self.phase(name).result
        rec=PhaseRecord("running",time.time(),None,None,None)
        self.state["phases"][name]=rec.to_dict(); self._save()
        try:
            result=fn() or {}
            if not isinstance(result,dict): result={"value":result}
            rec=PhaseRecord("completed",rec.started_at,time.time(),result,None)
            self.state["phases"][name]=rec.to_dict(); self._save(); return result
        except Exception as exc:
            rec=PhaseRecord("failed",rec.started_at,time.time(),None,"".join(traceback.format_exception_only(type(exc),exc)).strip())
            self.state["phases"][name]=rec.to_dict(); self._save(); raise

    def summary(self): return json.loads(json.dumps(self.state))
