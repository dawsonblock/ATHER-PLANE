from __future__ import annotations
from pathlib import Path
import json


def compare_runs(results: dict[str, dict]) -> dict:
    if not results: return {"ranking": [], "best": None}
    ranking=sorted(results, key=lambda k:(results[k].get("success_rate",0.0), results[k].get("mean_reward",0.0)), reverse=True)
    return {"ranking": ranking, "best": ranking[0], "results": results}


def write_markdown_report(path: str | Path, title: str, results: dict[str, dict]):
    report=compare_runs(results)
    lines=[f"# {title}", "", "| Variant | Success | Mean reward |", "|---|---:|---:|"]
    for name in report["ranking"]:
        r=results[name]
        lines.append(f"| {name} | {r.get('success_rate',0):.3f} | {r.get('mean_reward',0):.3f} |")
    lines += ["", f"Best by declared ordering: `{report['best']}`" if report['best'] else "No results."]
    Path(path).write_text("\n".join(lines)+"\n", encoding="utf-8")
    return report


def load_result_files(paths):
    out={}
    for p in paths:
        path=Path(p); out[path.stem]=json.loads(path.read_text(encoding="utf-8"))
    return out
