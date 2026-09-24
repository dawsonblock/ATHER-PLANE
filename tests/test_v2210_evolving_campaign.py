from pathlib import Path

import pytest

from awa import __version__ as ROOT_VERSION
from awa.v2 import __version__ as V2_VERSION
from awa.v2.empirical_closure import RunRecord
from awa.v2.evidence_integrity import RunProvenance
from awa.v2.evolving_campaign import (
    CampaignCell,
    CampaignEvidence,
    CampaignEvidenceStore,
    DiscoveryTreeBuilder,
    EvolvingCampaignOrchestrator,
    MetricProjection,
    PairedCampaignPlan,
    ValidationMatrix,
)
from awa.v2.meta_exploration import DeclarativeExplorationPolicy, EvidenceClass

H1 = "1" * 64
H2 = "2" * 64
H3 = "3" * 64
H4 = "4" * 64


def make_evidence(system, seed, task, split, transitions, score, *, novelty=0.0, information=0.0):
    metrics = {
        "success_rate": float(score),
        "episode_return": float(score),
        "constraint_violations": 0.0,
        "planner_calls_per_episode": 1.0,
        "world_model_calls_per_episode": 4.0,
        "inference_latency_ms": 2.0,
        "wall_clock_seconds": 10.0,
        "novelty": float(novelty),
        "information_gain": float(information),
        "uncertainty_reduction": float(information) / 2.0,
        "normalized_compute_cost": 0.01,
    }
    r = RunRecord(system, seed, task, split, transitions, metrics, H1, H2)
    rid = "|".join(map(str, (system, seed, task, split, transitions)))
    p = RunProvenance(rid, H1, H2, H3, H4, (f"scenario-{split}",))
    return CampaignEvidence(r, p, EvidenceClass.VALIDATED)


def history_rows():
    return [
        make_evidence("branch-a", 7, "task-a", "heldout", 10, .10, novelty=.90, information=.20),
        make_evidence("branch-a", 7, "task-a", "heldout", 20, 1.00, novelty=.30, information=.30),
        make_evidence("branch-b", 7, "task-b", "heldout", 10, .50, novelty=.10, information=.10),
        make_evidence("branch-b", 7, "task-b", "heldout", 20, .60, novelty=.10, information=.10),
    ]


def test_version_is_closed_across_packages():
    assert ROOT_VERSION == V2_VERSION == "2.38.6"


def test_campaign_cell_id_is_stable_and_keyed_by_full_cell():
    a = CampaignCell("p", 1, "doom", "heldout", 25_000)
    b = CampaignCell("p", 1, "doom", "heldout", 25_000)
    c = CampaignCell("p", 2, "doom", "heldout", 25_000)
    assert a.cell_id == b.cell_id
    assert a.cell_id != c.cell_id


def test_evidence_store_is_idempotent_and_rejects_conflict(tmp_path):
    store = CampaignEvidenceStore(tmp_path)
    row = make_evidence("p", 1, "doom", "heldout", 100, .5)
    store.put(row)
    store.put(row)
    changed = make_evidence("p", 1, "doom", "heldout", 100, .8)
    with pytest.raises(ValueError, match="evidence conflict"):
        store.put(changed)


def test_history_builder_preserves_experiment_branching():
    trees = DiscoveryTreeBuilder(MetricProjection()).build(history_rows())
    assert len(trees) == 1
    tree = trees[0]
    roots = tree.children("root", grounded_only=True)
    assert len(roots) == 2
    assert {n.metadata["system"] for n in roots} == {"branch-a", "branch-b"}
    assert all(len(tree.children(n.node_id, grounded_only=True)) == 1 for n in roots)


def test_paired_plan_exactly_expands_matrix():
    p = PairedCampaignPlan(
        "active", "candidate", (1, 2, 3, 4, 5), ("doom",),
        ("heldout", "transfer"), (25_000, 100_000), minimum_seeds=5,
    )
    assert len(p.cells) == 2 * 5 * 1 * 2 * 2
    assert p.protocol.cells() == {c.key for c in p.cells}


class FakeRunner:
    def __init__(self):
        self.calls = 0

    def run(self, cell, policy, work_dir: Path):
        self.calls += 1
        score = .75 if cell.policy_id != "active" else .60
        return make_evidence(cell.policy_id, cell.seed, cell.task, cell.split, cell.transitions, score)


def make_orchestrator(tmp_path):
    active = DeclarativeExplorationPolicy(
        "active", quality_weight=0.0, novelty_weight=-1.0,
        new_world_bias=.2, stop_threshold=-1.0, max_parallel=1,
    )
    validation = ValidationMatrix(
        seeds=(1, 2, 3, 4, 5), tasks=("doom",), splits=("heldout",),
        milestones=(25_000,), primary_metric="success_rate", minimum_seeds=5,
    )
    o = EvolvingCampaignOrchestrator(
        tmp_path, active_policy=active, validation=validation,
        mutation_step=1.5, maximum_seed_regression=.2,
    )
    o.import_history(history_rows())
    return o


def test_plan_only_never_invokes_runner(tmp_path):
    o = make_orchestrator(tmp_path)
    report = o.plan_iteration(max_rounds=3)
    assert report["status"] == "PLANNED"
    assert report["proposal"]["candidate"]["policy_id"] != "active"
    assert not (tmp_path / "iterations" / "0000" / "runs").exists()


def test_closed_loop_executes_exact_paired_cells_and_promotes(tmp_path):
    o = make_orchestrator(tmp_path)
    plan = o.plan_iteration(max_rounds=3)
    candidate = plan["proposal"]["candidate"]["policy_id"]
    runner = FakeRunner()
    receipt = o.execute_planned_iteration(runner)
    assert runner.calls == 10
    assert receipt.protocol_status == "PASS"
    assert receipt.promotion_status == "PASS"
    assert receipt.active_policy_after == candidate
    assert o.active_policy.policy_id == candidate
    assert o.iteration == 1


def test_execution_resumes_from_committed_cell_evidence(tmp_path):
    o = make_orchestrator(tmp_path)
    report = o.plan_iteration(max_rounds=3)
    from awa.v2.evolving_campaign import EvolvingCampaignOrchestrator as O
    plan = O._plan_from_report(report)
    proposal = O._proposal_from_plan(report)
    first = plan.cells[0]
    score = .75 if first.policy_id == proposal.candidate.policy_id else .60
    o.evidence_store.put(make_evidence(first.policy_id, first.seed, first.task, first.split, first.transitions, score))
    runner = FakeRunner()
    receipt = o.execute_planned_iteration(runner)
    assert receipt.protocol_status == "PASS"
    assert runner.calls == len(plan.cells) - 1


def test_protocol_failure_blocks_promotion_if_runner_returns_wrong_cell(tmp_path):
    o = make_orchestrator(tmp_path)
    o.plan_iteration(max_rounds=3)

    class WrongRunner(FakeRunner):
        def run(self, cell, policy, work_dir):
            return make_evidence(cell.policy_id, cell.seed, "wrong-task", cell.split, cell.transitions, .9)

    with pytest.raises(ValueError, match="different planned cell"):
        o.execute_planned_iteration(WrongRunner())


def test_non_grounded_campaign_evidence_is_rejected():
    r = RunRecord("p", 1, "doom", "heldout", 1, {"success_rate": .5}, H1, H2)
    rid = "p|1|doom|heldout|1"
    p = RunProvenance(rid, H1, H2, H3, H4, ("heldout",))
    with pytest.raises(ValueError, match="must be OBSERVED or VALIDATED"):
        CampaignEvidence(r, p, EvidenceClass.MODEL_SIMULATED)
