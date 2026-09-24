from __future__ import annotations

import json
from pathlib import Path

from awa.v2.system_split import split_manifest, validate_split_packages


def split_check_main() -> None:
    src_root = Path(__file__).resolve().parents[2]
    violations = validate_split_packages(src_root)
    payload = split_manifest()
    payload.update({
        "status": "ok" if not violations else "blocked",
        "violations": [v.to_dict() for v in violations],
    })
    print(json.dumps(payload, indent=2, sort_keys=True))
    if violations:
        raise SystemExit(2)
