from awa.v2.empirical_closure import RunRecord
from awa.v2.evidence_integrity import RunProvenance, build_integrity_receipt, canonical_sha256

H="a"*64

def record(split, scenario, seed=0):
    r=RunRecord("full",seed,"compose",split,25000,{"success_rate":.5},checkpoint_sha256=H,config_sha256=H)
    key="|".join(map(str,(r.system,r.seed,r.task,r.split,r.transitions)))
    p=RunProvenance(key,H,H,H,H,(scenario,))
    return r,key,p

def test_receipt_passes_and_is_deterministic():
    a,ka,pa=record("train","train-a",0); b,kb,pb=record("transfer","eval-a",1)
    x=build_integrity_receipt([a,b],{ka:pa,kb:pb}); y=build_integrity_receipt([a,b],{ka:pa,kb:pb})
    assert x["status"]=="PASS" and x["receipt_sha256"]==y["receipt_sha256"]

def test_split_leakage_fails_closed():
    a,ka,pa=record("train","same",0); b,kb,pb=record("heldout","same",1)
    out=build_integrity_receipt([a,b],{ka:pa,kb:pb})
    assert out["status"]=="FAIL" and out["split_leakage"]["overlap"]==["same"]

def test_duplicate_run_key_fails():
    a,ka,pa=record("transfer","eval-a",0)
    assert build_integrity_receipt([a,a],{ka:pa})["status"]=="FAIL"

def test_canonical_digest_ignores_dict_order():
    assert canonical_sha256({"a":1,"b":2})==canonical_sha256({"b":2,"a":1})
