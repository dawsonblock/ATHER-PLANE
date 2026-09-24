from __future__ import annotations
from pathlib import Path
import copy, csv, json, math, statistics
from awa.utils import seed_everything, resolve_device
from awa.training.trainer import train, evaluate


def deep_update(base: dict, patch: dict) -> dict:
    out=copy.deepcopy(base)
    for k,v in patch.items():
        if isinstance(v,dict) and isinstance(out.get(k),dict): out[k]=deep_update(out[k],v)
        else: out[k]=copy.deepcopy(v)
    return out


def mean_ci95(values):
    vals=list(map(float,values)); m=statistics.mean(vals)
    if len(vals)<2:return m,0.0
    return m,1.96*statistics.stdev(vals)/math.sqrt(len(vals))


def run_matrix(base_cfg: dict, variants: dict[str,dict], seeds: list[int], out_dir: str, episodes: int=20):
    root=Path(out_dir); root.mkdir(parents=True,exist_ok=True); rows=[]
    for name,patch in variants.items():
        for seed in seeds:
            cfg=deep_update(base_cfg,patch); cfg["seed"]=int(seed); seed_everything(seed); device=resolve_device(cfg.get("device","auto"))
            c=train(cfg,device,root/name/f"seed_{seed}"); r=evaluate(cfg,c,device,episodes=episodes)
            rows.append({"variant":name,"seed":seed,**r})
    summary={}
    for name in variants:
        subset=[r for r in rows if r["variant"]==name]; ss=[r["success_rate"] for r in subset]; rr=[r["mean_reward"] for r in subset]
        sm,sci=mean_ci95(ss); rm,rci=mean_ci95(rr); item={"success_mean":sm,"success_ci95":sci,"reward_mean":rm,"reward_ci95":rci,"seeds":len(ss)}
        distances=[r["mean_final_distance"] for r in subset if "mean_final_distance" in r]
        if distances:
            dm,dci=mean_ci95(distances); item.update({"final_distance_mean":dm,"final_distance_ci95":dci})
        summary[name]=item
    (root/"matrix_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    fieldnames=[]
    for r in rows:
        for key in r:
            if key not in fieldnames: fieldnames.append(key)
    with (root/"matrix_runs.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader(); w.writerows(rows)
    lines=["# Experiment Matrix","","| Variant | Success mean ±95% CI | Reward mean ±95% CI | Seeds |","|---|---:|---:|---:|"]
    for n,r in summary.items(): lines.append(f"| {n} | {r['success_mean']:.3f} ± {r['success_ci95']:.3f} | {r['reward_mean']:.3f} ± {r['reward_ci95']:.3f} | {r['seeds']} |")
    (root/"matrix_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary
