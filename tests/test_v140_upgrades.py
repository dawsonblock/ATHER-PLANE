import csv
import json
import numpy as np
import torch

from awa.memory.vector_store import VectorMemory
from awa.training.target import TargetValueNetwork
from awa.planning.learned_gate import LearnedArbitrator
from awa.connectome.official import detect_connection_columns, compile_official_connections
from awa.evaluation.ablation import compare_runs, write_markdown_report
from awa.evaluation.suite import run_generalization_suite
from awa.training.trainer import build_components


def _cfg():
    return {
        "seed": 1, "device": "cpu",
        "env": {"horizon": 10, "cue_delay": 3, "obs_dim": 8, "action_dim": 3},
        "model": {"obs_dim": 8, "latent_dim": 8, "deterministic_dim": 16, "stochastic_dim": 4,
                  "hidden_dim": 16, "action_dim": 3, "dynamics": "gru", "slow_enabled": False,
                  "uncertainty_heads": 2, "experts_enabled": False},
        "planner": {"enabled": False, "horizon": 2, "candidates": 4, "elites": 2, "iterations": 1, "uncertainty_limit": 1.0},
        "training": {"steps": 10, "batch_size": 2, "sequence_length": 2, "replay_capacity": 100,
                     "discount": 0.99, "lambda": 0.95, "target_tau": 0.01},
    }


def test_vector_memory_numpy_fallback():
    mem=VectorMemory(3,use_faiss=False)
    mem.add([1,0,0],{"name":"x"}); mem.add([0,1,0],{"name":"y"})
    hits=mem.search([0.9,0.1,0],1)
    assert hits[0].metadata["name"]=="x"


def test_target_value_ema_updates():
    src=torch.nn.Linear(4,1); target=TargetValueNetwork(src)
    before=next(target.parameters()).detach().clone()
    with torch.no_grad(): next(src.parameters()).add_(1.0)
    target.update(src,tau=0.5)
    after=next(target.parameters()).detach()
    assert not torch.allclose(before,after)
    assert all(not p.requires_grad for p in target.parameters())


def test_learned_arbitrator_shape():
    gate=LearnedArbitrator(8)
    p=gate(torch.tensor([0.2,0.3]),torch.tensor([0.5,0.4]),torch.tensor([0.1,0.2]),torch.tensor([0.3,0.3]))
    assert p.shape==(2,)
    assert torch.all((p>=0)&(p<=1))


def test_official_connection_alias_detection(tmp_path):
    p=tmp_path/"conn.csv"
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["bodyId_pre","bodyId_post","synapse_count"]); w.writerow([10,20,3]); w.writerow([20,10,2])
    cols=detect_connection_columns(p)
    assert cols=={"source":"bodyId_pre","target":"bodyId_post","weight":"synapse_count"}
    g=compile_official_connections(p)
    assert g.nodes==2 and g.edges==2


def test_ablation_report(tmp_path):
    results={"a":{"success_rate":0.5,"mean_reward":0.1},"b":{"success_rate":0.8,"mean_reward":0.0}}
    r=compare_runs(results); assert r["best"]=="b"
    out=tmp_path/"report.md"; write_markdown_report(out,"Test",results)
    assert "| b |" in out.read_text()


def test_generalization_suite_runs():
    cfg=_cfg(); c=build_components(cfg,torch.device("cpu"))
    rows=run_generalization_suite(cfg,c,torch.device("cpu"),episodes=2)
    assert len(rows)>=5
    assert {r["name"] for r in rows}>={"baseline","long_delay","noise_dropout"}
