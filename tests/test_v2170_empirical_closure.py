import json
from awa.v2.empirical_closure import RunRecord, summarize_runs, qualification_gates, planner_dependence, build_evidence_bundle, load_jsonl

def rows():
    out=[]
    for system,offset in [("full",.1),("actor_only",0.)]:
        for seed in range(5):
            for transitions in (25000,100000):
                out.append(RunRecord(system,seed,"compose","transfer",transitions,{
                    "success_rate":.5+offset+seed*.01+(transitions/1000000),
                    "episode_return":10+offset,"constraint_violations":0.,
                    "planner_calls_per_episode":max(0.,10-transitions/10000) if system=="full" else 0.,
                    "world_model_calls_per_episode":12.,"inference_latency_ms":2.,"wall_clock_seconds":5.,
                }))
    return out

def test_empirical_summary_is_paired_and_versioned():
    report=summarize_runs(rows())
    assert report["format"]=="awa-v2.17-empirical-closure-v1"
    assert report["summary"]["actor_only"]["paired_delta_vs_full"]["status"]=="ok"
    assert report["summary"]["actor_only"]["paired_delta_vs_full"]["mean"] < 0

def test_gates_require_five_seeds_and_heldout():
    assert qualification_gates(rows())["status"]=="PASS"
    assert qualification_gates(rows()[:2])["status"]=="FAIL"

def test_planner_dependence_curve():
    curve=planner_dependence(rows())
    assert curve[0]["transitions"]==25000
    assert curve[-1]["mean_planner_calls_per_episode"] <= curve[0]["mean_planner_calls_per_episode"]

def test_bundle_and_jsonl(tmp_path):
    rs=rows(); p=tmp_path/"runs.jsonl"
    p.write_text("\n".join(json.dumps(r.to_dict()) for r in rs))
    loaded=load_jsonl(p); bundle=build_evidence_bundle(loaded)
    assert len(loaded)==len(rs)
    assert bundle["qualification"]["status"]=="PASS"
