from __future__ import annotations


def percent_change(new: float, old: float) -> float:
    denom = max(abs(old), 1e-9)
    return 100.0 * (new - old) / denom


def acceptance_report(baseline: dict, candidate: dict, rules: dict):
    success = percent_change(candidate["success_rate"], baseline["success_rate"])
    compute = percent_change(baseline.get("compute", 1.0), candidate.get("compute", 1.0))
    passed = success >= rules["success_improvement_pct"] or compute >= rules["compute_reduction_pct"]
    return {"passed": passed, "success_improvement_pct": success, "compute_reduction_pct": compute}
