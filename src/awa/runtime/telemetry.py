from __future__ import annotations
from pathlib import Path
import json
import time


class JsonlTelemetry:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, **fields):
        fields = {"time": time.time(), **fields}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(fields, sort_keys=True) + "\n")
