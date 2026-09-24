from __future__ import annotations
from dataclasses import dataclass, asdict


@dataclass
class PromotionDecision:
    promoted: bool
    average_horizon_improvement_pct: float
    worst_horizon_regression_pct: float
    nll_regression_pct: float
    reasons: list[str]

    def to_dict(self):
        return asdict(self)


def _pct_improvement(new: float, old: float) -> float:
    if old == 0:
        return 0.0 if new == 0 else float("-inf")
    return 100.0 * (old - new) / abs(old)


def assess_world_model_promotion(
    candidate: dict,
    baseline: dict,
    *,
    min_average_horizon_improvement_pct: float = 0.0,
    max_horizon_regression_pct: float = 3.0,
    max_nll_regression_pct: float = 3.0,
) -> PromotionDecision:
    common = sorted(set(candidate["horizon_rmse"]) & set(baseline["horizon_rmse"]), key=lambda x: int(x))
    if not common:
        raise ValueError("candidate and baseline have no common horizon metrics")
    improvements = [
        _pct_improvement(float(candidate["horizon_rmse"][h]), float(baseline["horizon_rmse"][h]))
        for h in common
    ]
    avg = sum(improvements) / len(improvements)
    worst_regression = max(0.0, -min(improvements))
    nll_improvement = _pct_improvement(float(candidate["one_step_nll"]), float(baseline["one_step_nll"]))
    nll_regression = max(0.0, -nll_improvement)
    reasons = []
    if avg < min_average_horizon_improvement_pct:
        reasons.append(f"average horizon improvement {avg:.3f}% below required {min_average_horizon_improvement_pct:.3f}%")
    if worst_regression > max_horizon_regression_pct:
        reasons.append(f"worst horizon regression {worst_regression:.3f}% exceeds {max_horizon_regression_pct:.3f}%")
    if nll_regression > max_nll_regression_pct:
        reasons.append(f"NLL regression {nll_regression:.3f}% exceeds {max_nll_regression_pct:.3f}%")
    return PromotionDecision(not reasons, avg, worst_regression, nll_regression, reasons)
