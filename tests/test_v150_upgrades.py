import json
import numpy as np
import torch
from awa.training.sequence import continuation_mask, overshooting_latent_loss
from awa.planning.adaptive import AdaptivePlanningPolicy
from awa.planning.calibrator import LogisticUncertaintyCalibrator
from awa.planning.learned_gate import LearnedArbitrator
from awa.memory.vector_store import VectorMemory
from awa.connectome.graph import CSRGraph
from awa.connectome.controls import permute_weights, group_preserving_rewire
from awa.connectome.motifs import in_degrees, out_degrees
from awa.connectome.circuit import FixedSparseCircuit
from awa.connectome.provenance import build_provenance_manifest
from awa.evaluation.matrix import deep_update, mean_ci95
from awa.model import WorldModel
import torch.nn.functional as F


def test_continuation_mask_after_terminal():
    d=torch.tensor([[[0.],[1.],[0.],[0.]]])
    m=continuation_mask(d)
    assert m.flatten().tolist()==[1.0,1.0,0.0,0.0]


def test_overshooting_loss_runs():
    wm=WorldModel(6,8,12,4,3,16,"gru")
    B,T=2,4; obs=torch.randn(B,T+1,6); acts=torch.randint(0,3,(B,T))
    b=wm.initial_belief(B,torch.device("cpu")); z=torch.zeros(B,3); posts=[]
    o=wm.observe(b,z,obs[:,0]); b=o.belief
    for t in range(T):
        a=F.one_hot(acts[:,t],3).float(); o=wm.observe(b,a,obs[:,t+1],step_index=t+1); b=o.belief; posts.append(b)
    loss=overshooting_latent_loss(wm,posts,obs,acts,3,horizon=3,burn_in=1)
    assert torch.isfinite(loss) and loss.ndim==0


def test_adaptive_planning_budget():
    p=AdaptivePlanningPolicy(20,200,low=.2,medium=.5,high=.8,min_horizon=3,min_candidates=20)
    assert p.choose(.1).horizon==20
    assert p.choose(.6).horizon==3
    assert not p.choose(.9).use_planner


def test_uncertainty_calibrator_learns_direction():
    c=LogisticUncertaintyCalibrator(); u=np.array([0.01,0.05,0.1,1.,2.,3.]); y=np.array([0,0,0,1,1,1])
    c.fit_numpy(u,y,steps=120,lr=.08)
    with torch.no_grad(): p=c(torch.tensor([0.05,2.0]))
    assert p[1]>p[0]


def test_learned_gate_five_features():
    g=LearnedArbitrator(8); p=g(.1,.2,.3,.4,.5); assert p.numel()==1 and 0<=float(p.item())<=1


def test_vector_memory_dedup_and_bound():
    m=VectorMemory(2,use_faiss=False,max_items=2,dedup_threshold=.999)
    i=m.add([1,0],{"name":"a","importance":1}); j=m.add([1,0],{"name":"a2","importance":2})
    assert i==j and len(m.metadata)==1
    m.add([0,1],{"name":"b","importance":0}); m.add([-1,0],{"name":"c","importance":3})
    assert len(m.metadata)==2
    st=m.state_dict(); m2=VectorMemory(2,use_faiss=False).load_state_dict(st); assert len(m2.search([-1,0],1))==1


def _graph():
    return CSRGraph(np.array([0,2,4,6,8],dtype=np.int64),np.array([1,2,0,3,0,3,1,2],dtype=np.int32),np.arange(1,9,dtype=np.float32),np.arange(4,dtype=np.int64))


def test_connectome_controls_preserve_degrees_and_weights():
    g=_graph(); r=group_preserving_rewire(g,[0,0,1,1],swaps=2,seed=4,allow_self=False)
    assert np.array_equal(in_degrees(g),in_degrees(r)); assert np.array_equal(out_degrees(g),out_degrees(r))
    p=permute_weights(g,seed=3); assert sorted(p.weight.tolist())==sorted(g.weight.tolist())


def test_fixed_sparse_circuit_gradient():
    c=FixedSparseCircuit(_graph()); x=torch.randn(3,4,requires_grad=True); y=c(x).sum(); y.backward()
    assert x.grad is not None and c.gain.grad is not None


def test_provenance_hash(tmp_path):
    p=tmp_path/'x.txt'; p.write_text('abc'); m=build_provenance_manifest([p],version='v')
    assert m['files'][0]['sha256']=='ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad'


def test_matrix_helpers():
    x=deep_update({'a':{'b':1},'c':2},{'a':{'b':3}}); assert x['a']['b']==3 and x['c']==2
    mean,ci=mean_ci95([1,2,3]); assert mean==2 and ci>0
