from __future__ import annotations

from dataclasses import dataclass
import math
import torch


class UnsafeActionError(RuntimeError):
    pass


@dataclass
class Decision:
    action: torch.Tensor
    source: str
    budget: int
    expected_voc: float
    horizon_uncertainty: float
    risk: float
    metadata: dict


class SelectiveDecisionController:
    """Actor first, optional search second, mandatory risk rejection last."""

    def __init__(
        self,
        actor,
        world,
        voc,
        planners: dict[str, object],
        uncertainty_limit: float = 10.0,
        risk_limit: float = 0.8,
        guard=None,
    ):
        self.actor = actor
        self.world = world
        self.voc = voc
        self.planners = planners
        self.uncertainty_limit = float(uncertainty_limit)
        self.risk_limit = float(risk_limit)
        self.guard = guard

    @torch.no_grad()
    def act(self, belief, decision_features, costs=None):
        actor_action = self.actor.deterministic_action(belief)
        if isinstance(actor_action, tuple):
            actor_action = actor_action[0]
        choices, _ = self.voc.choose(decision_features, costs)
        choice = choices[0]
        horizon = max(1, choice.budget)
        u = float(self.world.uncertainty.at_horizon(belief, horizon).mean().item())
        actor_risk = float(self.world.risk.aggregate_risk(belief, actor_action).mean().item())
        if not math.isfinite(u):
            u = float("inf")
        if not math.isfinite(actor_risk):
            actor_risk = float("inf")
        action = actor_action
        source = "actor"
        budget = 0
        metadata = {}

        if (
            choice.planner != "actor"
            and choice.voc > 0
            and u <= self.uncertainty_limit
            and actor_risk <= self.risk_limit
        ):
            planner = self.planners.get(choice.planner)
            if planner is not None:
                result = planner.plan(belief, actor=self.actor, budget=choice.budget)
                action = result.action
                source = result.planner
                budget = choice.budget
                metadata = {
                    "planner_score": result.score,
                    "world_model_calls": result.world_model_calls,
                }
            else:
                metadata = {"reason": "planner_unavailable"}
        else:
            metadata = {"reason": "no_positive_reliable_search_value"}

        risk = float(self.world.risk.aggregate_risk(belief, action).mean().item())
        if not math.isfinite(risk):
            risk = float("inf")
        if self.guard is not None:
            guarded = self.guard.validate(belief, action)
            action = guarded.action
            risk = guarded.risk
            if not guarded.accepted:
                if not guarded.recovery_safe:
                    raise UnsafeActionError(
                        "SafeActionGuard could not produce a recovery action below the risk limit"
                    )
                rejected_source = source
                source = "recovery"
                budget = 0
                metadata = {
                    **metadata,
                    "guard_reason": guarded.reason,
                    "rejected_source": rejected_source,
                    "proposed_risk": guarded.proposed_risk,
                }
        elif risk > self.risk_limit:
            # A risky planner result may fall back to a known-safe actor proposal.
            if source != "actor" and actor_risk <= self.risk_limit:
                return Decision(
                    actor_action,
                    "actor",
                    0,
                    choice.voc,
                    u,
                    actor_risk,
                    {**metadata, "warning": "planner_risk_rejected_without_guard"},
                )
            # If even the actor is unsafe and no recovery guard exists, fail closed.
            raise UnsafeActionError(
                "risk limit exceeded and no SafeActionGuard/recovery action is configured"
            )
        return Decision(action, source, budget, choice.voc, u, risk, metadata)


class InformationValue:
    @staticmethod
    def utility(task_value, information_gain, risk, compute_cost, beta: float = 0.25):
        return task_value + beta * information_gain - risk - compute_cost
