import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.meta_exploration import (
    DeclarativeExplorationPolicy,
    DeclarativePolicyMutator,
    DiscoveryNode,
    DiscoveryTree,
    EvidenceClass,
    EvolvingExplorationController,
    ExplorationPolicyPromotionGate,
    MetaPolicyOptimizer,
    OnlinePolicyResult,
    ReplayObjective,
    ReplaySimulatorPool,
    ReplayWorld,
)
from awa.v2.reasoning import (
    AdaptiveReasoningEffortController,
    EffortOutcome,
    EffortOutcomeLedger,
    EffortSignals,
)


def node(node_id, parent, idx, *, quality=0.0, novelty=0.0, transfer=0.0, info=0.0,
         evidence=EvidenceClass.OBSERVED, cost=0.0):
    return DiscoveryNode(
        node_id=node_id,
        parent_id=parent,
        iteration=idx + 1,
        generation_index=idx,
        evidence=evidence,
        metrics={
            "quality": quality,
            "novelty": novelty,
            "transfer": transfer,
            "information_gain": info,
            "uncertainty_reduction": info / 2,
            "constraint_violations": 0.0,
        },
        compute_cost=cost,
        provenance_sha256=(f"{idx + 1:064x}"[-64:]),
    )


def sample_tree():
    t = DiscoveryTree("world-a")
    t.add(node("a", "root", 0, quality=.1, novelty=.8, info=.2))
    t.add(node("b", "root", 1, quality=.4, novelty=.1, info=.1))
    t.add(node("a1", "a", 0, quality=.9, transfer=.7, info=.3))
    t.add(node("b1", "b", 0, quality=.45, transfer=.1, info=.1))
    return t


def test_version_is_closed_across_packages():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_discovery_tree_rejects_missing_parent_and_roundtrips():
    t = sample_tree()
    with pytest.raises(ValueError):
        t.add(node("bad", "missing", 0))
    restored = DiscoveryTree.from_dict(t.to_dict())
    assert [n.node_id for n in restored.nodes] == [n.node_id for n in t.nodes]


def test_replay_hides_simulated_nodes_from_empirical_world():
    t = sample_tree()
    t.add(node("sim", "root", 2, quality=99, evidence=EvidenceClass.MODEL_SIMULATED))
    p = DeclarativeExplorationPolicy("p", new_world_bias=1.0, stop_threshold=-1.0, max_parallel=1)
    r = ReplayWorld(t).run(p, max_rounds=10)
    assert "sim" not in r.revealed
    assert all(t.get(x).evidence.grounded for x in r.revealed)


def test_replay_can_compare_branching_and_continuation_policies():
    t = sample_tree()
    continue_policy = DeclarativeExplorationPolicy(
        "continue", quality_weight=2.0, novelty_weight=0.0, new_world_bias=.15,
        stop_threshold=-1.0, max_parallel=1,
    )
    open_policy = DeclarativeExplorationPolicy(
        "open", quality_weight=0.0, novelty_weight=0.0, new_world_bias=2.0,
        stop_threshold=-1.0, max_parallel=1,
    )
    world = ReplayWorld(t, ReplayObjective(compute_cost_weight=0.0))
    a = world.run(continue_policy, max_rounds=3)
    b = world.run(open_policy, max_rounds=3)
    assert a.revealed != b.revealed
    assert a.score >= b.score


def test_mutator_preserves_exploration_reserve_floors():
    base = DeclarativeExplorationPolicy("base", new_world_fraction=.01, adversarial_fraction=.01)
    mut = DeclarativePolicyMutator(.1, min_new_world_fraction=.15, min_adversarial_fraction=.10)
    rows = mut.neighbors(base)[1:]
    assert all(p.new_world_fraction >= .15 for p in rows)
    assert all(p.adversarial_fraction >= .10 for p in rows)


def test_meta_optimizer_proposes_replay_scored_candidate():
    pool = ReplaySimulatorPool([ReplayWorld(sample_tree())])
    opt = MetaPolicyOptimizer(pool, DeclarativePolicyMutator(step=.5))
    base = DeclarativeExplorationPolicy("base", quality_weight=0.0, new_world_bias=.2, stop_threshold=-1)
    proposal = opt.propose(base, max_rounds=3)
    assert proposal.replay_gain >= 0
    assert proposal.candidate.policy_id


def test_promotion_requires_grounded_paired_online_evidence():
    pool = ReplaySimulatorPool([ReplayWorld(sample_tree())])
    c = EvolvingExplorationController(
        DeclarativeExplorationPolicy("base", quality_weight=0, novelty_weight=-1.0, new_world_bias=.2, stop_threshold=-1, max_parallel=1),
        simulator_pool=pool,
        optimizer=MetaPolicyOptimizer(pool, DeclarativePolicyMutator(step=1.5)),
        promotion_gate=ExplorationPolicyPromotionGate(minimum_seeds=5, minimum_mean_gain=0.0, max_seed_regression=.2),
    )
    p = c.dream(max_rounds=3)
    assert p.candidate.policy_id != "base"
    rows=[]
    for seed in range(5):
        rows.append(OnlinePolicyResult("base", seed, 1.0, EvidenceClass.VALIDATED, f"{seed+100:064x}"))
        rows.append(OnlinePolicyResult(p.candidate.policy_id, seed, 1.1, EvidenceClass.VALIDATED, f"{seed+200:064x}"))
    receipt=c.validate_and_promote(rows)
    assert receipt.status == "PASS"
    assert c.active_policy.policy_id == p.candidate.policy_id


def test_promotion_rejects_simulated_evidence_even_if_score_is_high():
    gate=ExplorationPolicyPromotionGate(minimum_seeds=2)
    from awa.v2.meta_exploration import PolicyProposal
    base=DeclarativeExplorationPolicy("base")
    cand=DeclarativeExplorationPolicy("cand", quality_weight=1.1)
    proposal=PolicyProposal("base",cand,2,1,1)
    rows=[]
    for seed in range(2):
        rows.append(OnlinePolicyResult("base",seed,0,EvidenceClass.OBSERVED,f"{seed+300:064x}"))
        rows.append(OnlinePolicyResult("cand",seed,10,EvidenceClass.MODEL_SIMULATED))
    assert gate.evaluate(proposal,rows).status == "FAIL"


def test_adaptive_effort_masks_planning_when_model_is_unreliable():
    ctl=AdaptiveReasoningEffortController(minimum_model_reliability=.5)
    s=EffortSignals(.5,.8,.5,.2,.1,.7,model_reliability=.1)
    d=ctl.choose(s,[0,.5,.8,1.0,1.2])
    assert d.level.planner == "actor"
    assert set(d.masked_levels) == {"shallow","medium","deep","strategic"}


def test_adaptive_effort_trades_gain_against_resource_pressure():
    ctl=AdaptiveReasoningEffortController(compute_lambda=.5,latency_lambda=0,risk_lambda=0,risk_deliberation_bonus=0)
    low=EffortSignals(.1,.1,.1,.9,.1,.2,resource_pressure=0)
    high=EffortSignals(.1,.1,.1,.9,.1,.2,resource_pressure=1)
    gains=[0,.2,.35,.45,.5]
    d1=ctl.choose(low,gains)
    d2=ctl.choose(high,gains)
    assert d2.level.normalized_compute <= d1.level.normalized_compute


def test_effort_ledger_reports_planner_dependence_and_training_arrays():
    ledger=EffortOutcomeLedger()
    ledger.add(EffortOutcome("reflex","actor",0,0,0,0,1,.1,.2,True))
    ledger.add(EffortOutcome("medium","policy_mppi",128,.3,.4,128,4,.2,.8,True))
    s=ledger.summary()
    assert s["planner_fraction"] == .5
    x,y=ledger.training_arrays()
    assert x.shape == (2,4) and y.shape == (2,)

def test_grounded_online_result_requires_provenance_digest():
    with pytest.raises(ValueError):
        OnlinePolicyResult("p",0,1.0,EvidenceClass.OBSERVED)


def test_promotion_gate_rejects_duplicate_policy_seed_rows():
    from awa.v2.meta_exploration import PolicyProposal
    base=DeclarativeExplorationPolicy("base")
    cand=DeclarativeExplorationPolicy("cand",quality_weight=1.1)
    proposal=PolicyProposal("base",cand,2,1,1)
    rows=[
        OnlinePolicyResult("base",0,1.0,EvidenceClass.VALIDATED,"1"*64),
        OnlinePolicyResult("base",0,1.0,EvidenceClass.VALIDATED,"2"*64),
        OnlinePolicyResult("cand",0,1.2,EvidenceClass.VALIDATED,"3"*64),
        OnlinePolicyResult("base",1,1.0,EvidenceClass.VALIDATED,"4"*64),
        OnlinePolicyResult("cand",1,1.2,EvidenceClass.VALIDATED,"5"*64),
    ]
    receipt=ExplorationPolicyPromotionGate(minimum_seeds=2).evaluate(proposal,rows)
    assert receipt.status == "FAIL"
    assert any("duplicate" in x for x in receipt.failures)
