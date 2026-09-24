from __future__ import annotations
import numpy as np
import torch

from awa.v2.world import MultimodalWorldModel
from awa.v2.ensemble import WorldModelEnsemble
from awa.v2.planners import RiskAwarePolicySeededMPPI,_score_stochastic_rollout
from awa.v2.game.action_codec import HybridGameActionCodec
from awa.v2.replay import ShardedReplayStore
from awa.v2.evaluation_stats import bootstrap_ci,paired_bootstrap_ci
from awa.v2.game import IterativeDataAggregator,LogicalArenaTeacher,ReusableGameLoop,SeedMetric,summarize_seed_metrics
from awa.v2.curriculum import ProceduralTaskFactory


class Actor:
    def __init__(self, action_dim): self.action_dim=action_dim
    def deterministic_action(self, belief):
        return torch.zeros(belief.shape[0],self.action_dim,device=belief.device)


def test_hybrid_codec_roundtrip_and_enumeration():
    c=HybridGameActionCodec()
    a=c.encode([.4,-.3],attack=True,interact=False)
    h=c.decode(a)
    assert h.attack and not h.interact
    assert np.allclose(h.movement,[.4,-.3])
    assert c.enumerate_binary([0,0]).shape==(4,4)
    q=c.quantize([.1,.2,.2,.8])
    assert tuple(q[2:])==(-1.0,1.0)


def test_stochastic_rollout_and_risk_mppi_metadata():
    torch.manual_seed(211)
    w=MultimodalWorldModel(8,4,hidden=16,components=3,horizons=(1,2))
    actor=Actor(4); b=torch.zeros(1,8)
    acts=torch.zeros(5,3,4)
    score,calls,stats=_score_stochastic_rollout(w,b.expand(5,-1),acts,samples=3,cvar_alpha=.34)
    assert score.shape==(5,) and calls==5*3*3
    assert torch.all(stats['cvar']<=stats['mean']+1e-6)
    p=RiskAwarePolicySeededMPPI(w,actor,[-1]*4,[1]*4,horizon=3,candidates=5,samples=3)
    r=p.plan(b,budget=5)
    assert r.action.shape==(1,4)
    assert r.world_model_calls==5*3*3
    assert r.metadata['samples']==3 and 'cvar_score' in r.metadata


def test_world_model_ensemble_disagreement_and_prediction():
    torch.manual_seed(1); a=MultimodalWorldModel(6,2,hidden=12,components=2)
    torch.manual_seed(2); b=MultimodalWorldModel(6,2,hidden=12,components=2)
    e=WorldModelEnsemble([a,b]); s=torch.zeros(4,6); u=torch.zeros(4,2)
    d=e.disagreement(s,u)
    assert d.next_state.shape==(4,) and torch.all(d.next_state>=0)
    out=e.imagine_step(s,u,deterministic=True)
    assert out['belief'].shape==(4,6) and out['epistemic'].shape==(4,)


def test_sharded_replay_roundtrip_and_integrity(tmp_path):
    s=ShardedReplayStore(tmp_path/'replay',shard_size=3)
    arrays={'x':np.arange(8,dtype=np.float32)[:,None],'y':np.arange(8,dtype=np.int64)}
    shards=s.append(arrays)
    assert len(shards)==3 and len(s)==8 and s.verify()==[]
    rows=list(s.iter_shards())
    assert np.concatenate([r['x'] for r in rows]).shape==(8,1)
    reopened=ShardedReplayStore(tmp_path/'replay',shard_size=3)
    assert len(reopened)==8 and reopened.verify()==[]


def test_bootstrap_confidence_intervals_are_paired():
    ci=bootstrap_ci([1,2,3,4],resamples=200,seed=1)
    assert ci.n==4 and ci.low<=ci.mean<=ci.high
    d=paired_bootstrap_ci([2,3,4],[1,1,1],resamples=200,seed=1)
    assert d.mean==2.0


def test_iterative_aggregation_grows_one_replay_across_rounds():
    f=ProceduralTaskFactory(211); tasks=[f.make(1,.15,0,'agg')]
    loop=ReusableGameLoop(horizon=35)
    teacher=LogicalArenaTeacher()
    agg=IterativeDataAggregator(loop,seed=211)
    rows,_,_=agg.run(tasks,teacher,teacher,rounds=2,episodes_per_task=1,planner_schedule=lambda r:.5)
    assert len(rows)==2 and rows[1].replay_size>rows[0].replay_size>0


def test_multiseed_summary_reports_ci():
    rows=[SeedMetric(i,.5+.1*i,1+i,.3-.05*i,.2) for i in range(3)]
    rep=summarize_seed_metrics(rows,resamples=200,seed=211)
    assert rep.seeds==3 and rep.success_rate['low']<=rep.success_rate['mean']<=rep.success_rate['high']


def test_bootstrap_world_ensemble_trains_in_shared_belief_space(tmp_path):
    from awa.v2.curriculum import ProceduralTaskFactory
    from awa.v2.game import collect_game_dataset,LogicalArenaTeacher,GameBeliefEncoder,GameBeliefSequenceDataset,train_bootstrap_world_ensemble
    from awa.v2.datasets import OfflineTransitionDataset
    f=ProceduralTaskFactory(212); task=f.make(1,.1,0,'ens')
    path=tmp_path/'game.npz'
    collect_game_dataset([task],path,episodes_per_task=1,horizon=24,policy=LogicalArenaTeacher())
    ds=OfflineTransitionDataset(path); seq=GameBeliefSequenceDataset(ds,sequence_length=2)
    enc=GameBeliefEncoder(latent_dim=8,local_dim=8,global_dim=8,goal_latent_dim=4)
    base=MultimodalWorldModel(enc.belief_dim,4,hidden=16,components=2,horizons=(1,2))
    ens=train_bootstrap_world_ensemble(seq,enc,base,members=2,epochs=1,batch_size=8,seed=212)
    b=torch.zeros(1,enc.belief_dim); a=torch.zeros(1,4)
    assert len(ens.members)==2 and ens.disagreement(b,a).next_state.shape==(1,)
