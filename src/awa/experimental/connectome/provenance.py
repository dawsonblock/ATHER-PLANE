from __future__ import annotations
from pathlib import Path
import hashlib, json, time


def sha256_file(path: str | Path, chunk: int=1024*1024) -> str:
    h=hashlib.sha256()
    with open(path,"rb") as f:
        while True:
            b=f.read(chunk)
            if not b:break
            h.update(b)
    return h.hexdigest()


def build_provenance_manifest(files, dataset: str="MaleCNS", version: str="unknown", metadata: dict | None=None):
    rows=[]
    for p in files:
        path=Path(p); rows.append({"path":str(path),"bytes":path.stat().st_size,"sha256":sha256_file(path)})
    return {"dataset":dataset,"version":version,"created_unix":int(time.time()),"files":rows,"metadata":dict(metadata or {})}


def write_manifest(path, manifest):
    Path(path).write_text(json.dumps(manifest,indent=2,sort_keys=True),encoding="utf-8")
