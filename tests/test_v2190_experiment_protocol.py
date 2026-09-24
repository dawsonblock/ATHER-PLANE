from awa.v2.empirical_closure import RunRecord
from awa.v2.evidence_integrity import RunProvenance
from awa.v2.experiment_protocol import ExperimentProtocol, verify_protocol_completeness, verify_metric_semantics, build_protocol_receipt

H="a"*64

def rec(system="full",seed=1):
    return RunRecord(system,seed,"doom","heldout",100,{"success_rate":.5,"episode_return":1,"constraint_violations":0,"planner_calls_per_episode":1,"world_model_calls_per_episode":2,"inference_latency_ms":3,"wall_clock_seconds":4},H,H)
def prov(r):
    rid="|".join(map(str,(r.system,r.seed,r.task,r.split,r.transitions)))
    return rid,RunProvenance(rid,H,H,H,H,("heldout-01",))

def test_exact_matrix_passes():
    p=ExperimentProtocol("p",("full",),(1,2),("doom",),("heldout",),(100,),minimum_seeds=2)
    rows=[rec(seed=1),rec(seed=2)]
    assert verify_protocol_completeness(p,rows)["status"]=="PASS"

def test_missing_and_extra_fail_closed():
    p=ExperimentProtocol("p",("full",),(1,2),("doom",),("heldout",),(100,),minimum_seeds=2)
    assert verify_protocol_completeness(p,[rec(seed=1)])["status"]=="FAIL"
    x=RunRecord("other",1,"doom","heldout",100,rec().metrics,H,H)
    assert verify_protocol_completeness(p,[rec(seed=1),rec(seed=2),x])["status"]=="FAIL"

def test_metric_domain_checks():
    bad=RunRecord("full",1,"doom","heldout",100,{"success_rate":1.2},H,H)
    assert verify_metric_semantics([bad])["status"]=="FAIL"

def test_receipt_binds_protocol_and_integrity():
    p=ExperimentProtocol("p",("full",),(1,2),("doom",),("heldout",),(100,),minimum_seeds=2)
    rows=[rec(seed=1),rec(seed=2)]; ps=dict(prov(r) for r in rows)
    out=build_protocol_receipt(p,rows,ps)
    assert out["status"]=="PASS"
    assert len(out["protocol_sha256"])==64 and len(out["receipt_sha256"])==64
