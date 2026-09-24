import numpy as np
import torch

from awa.environments.continuous_point import ContinuousPointEnv
from awa.v2.curriculum import TaskSpec, TaskOutcome, ProceduralTaskFactory, LearningFrontierScheduler
from awa.v2.experience import ExperienceAnalyzer, RunningNovelty, branch_actions
from awa.v2.replay import StructuralPriority, PrioritizedExperienceBuffer
from awa.v2.reasoning import EffortLevel, ReasoningEffortController
from awa.v2.distill import TeacherPool, TeacherCandidate, DistillationBuffer, PolicyDistiller
from awa.v2.skills import SkillRegistry, SkillSpec, SkillDiscovery, SkillComposer
from awa.v2.engram import EngramLite
from awa.v2.training import RolloutEnvelope, StalenessPolicy, ExperienceQueue
from awa.v2.context_replay import BoundedContextReplay
from awa.planning.actor import TanhGaussianActor


def test_procedural_factory_randomizes_rules_and_is_deterministic():
    f=ProceduralTaskFactory(7)
    a=f.make(10,.6,3); b=f.make(10,.6,3); c=f.make(10,.6,4)
    assert a==b and a.task_id!=c.task_id
    assert a.environment['gravity_scale'] != 1.0 or a.environment['friction_scale'] != 1.0
    assert 'adaptation' in a.concepts


def test_learning_frontier_prefers_learnable_middle_over_mastered():
    f=ProceduralTaskFactory(1); mid=f.make(5,.5,0); mastered=f.make(1,.1,0)
    s=LearningFrontierScheduler(seed=1)
    for i in range(20):
        s.record(TaskOutcome(mid.task_id, i>=8,1.0, prediction_error=.3, novelty=.3))
        s.record(TaskOutcome(mastered.task_id, True,1.0, prediction_error=.01, novelty=.01))
    assert s.priority(mid) > s.priority(mastered)


def test_experience_analyzer_separates_repeated_surprise_and_priority():
    a=ExperienceAnalyzer(.2,repeat_window=3); rows=[]
    for _ in range(3):
        rows.append(a.make(state=[0],goal=[1],action=[1],predicted_next=[0],actual_next=[1],reward=0,success=False,
                           novelty=.7,uncertainty=.2,risk=.1,td_error=.5,task_importance=.4,information_value=.3))
    assert rows[-1].repeated_surprise
    p=StructuralPriority()(rows[-1]); assert 0 < p <= 5
    buf=PrioritizedExperienceBuffer(capacity=8,seed=0); [buf.add(r) for r in rows]
    items,idx,w=buf.sample(2); assert len(items)==2 and w.shape==(2,)


def test_running_novelty_drops_for_repeat():
    n=RunningNovelty(); x=np.array([1.,0.,0.],np.float32)
    first=n.observe(x); second=n.observe(x)
    assert first > second


def test_counterfactual_branching_restores_exact_env():
    env=ContinuousPointEnv(seed=4); env.reset(); before=env.state_dict()
    batch=branch_actions(env,[[-1,0],[1,0],[0,1]],state_id='s0')
    after=env.state_dict()
    assert len(batch.branches)==3 and batch.consequence_spread()>0
    assert before['t']==after['t'] and np.allclose(before['position'],after['position']) and np.allclose(before['target'],after['target'])


def test_reasoning_effort_chooses_gain_minus_cost():
    levels=[EffortLevel('actor','actor',0,0),EffortLevel('small','mppi',32,.2),EffortLevel('big','mppi',128,.8)]
    c=ReasoningEffortController(levels,cost_lambda=.5)
    d=c.choose([0],predicted_gains=[0,.3,.45])
    assert d.level.name=='small' and d.voc>0


def test_teacher_pool_selects_utility_and_distills():
    pool=TeacherPool()
    c=[TeacherCandidate('mppi',np.array([.2,-.1],np.float32),value=1.0,risk=.1,compute_cost=.1),
       TeacherCandidate('icem',np.array([.8,.8],np.float32),value=.9,risk=.5,compute_cost=.1)]
    ex=pool.example(np.zeros(4,np.float32),np.zeros(2,np.float32),c,student_value=.2)
    assert ex.teacher=='mppi'
    db=DistillationBuffer(); [db.add(ex) for _ in range(8)]
    actor=TanhGaussianActor(4,2,32,[-1,-1],[1,1]); dist=PolicyDistiller(actor,lr=1e-2)
    report=dist.fit(db,epochs=2,batch_size=4); assert report['steps']>0 and np.isfinite(report['final_loss'])


def test_skill_discovery_registration_and_composition():
    d=SkillDiscovery(round_decimals=1,min_examples=3)
    states=np.array([[0,0],[.5,0],[1,0]],np.float32); actions=np.array([[1,0],[1,0]],np.float32)
    for _ in range(4): d.observe(states,actions,True)
    candidates=d.candidates(); assert candidates
    reg=SkillRegistry(); s=candidates[0]; s.preconditions=frozenset({'start'}); s.effects=frozenset({'finish'})
    assert reg.register(s,min_success=.6,min_confidence=.5)
    path,cost=SkillComposer(reg).plan({'start'},{'finish'}); assert path==[s.name] and np.isfinite(cost)


def test_engram_lite_sparse_pattern_recall():
    m=EngramLite(capacity=4,buckets=1)
    m.put('physics_regime',[1,0,0],{'name':'ice'},score=1)
    m.put('tactic',[0,1,0],{'name':'cover'},score=.7)
    hit=m.query([1,.01,0],k=1,pattern='physics_regime')
    assert hit and hit[0][1].payload['name']=='ice'


def test_staleness_policy_downweights_and_drops_old_rollouts():
    p=StalenessPolicy(max_lag=3,half_life=1)
    fresh=RolloutEnvelope('x',10); stale=RolloutEnvelope('y',5)
    assert p.weight(10,fresh)==1.0 and p.weight(10,stale)==0.0
    q=ExperienceQueue(staleness=p); q.push(stale); q.push(fresh)
    xs,w=q.pop_batch(2,10); assert [x.payload for x in xs]==['x'] and w==[1.0]


def test_bounded_context_replay_reconstructs_from_periodic_checkpoint():
    r=BoundedContextReplay(checkpoint_interval=4,max_transitions=32)
    state=0
    for step in range(9):
        r.record(step,state,transition=1)
        state += 1
    assert r.reconstruct(7,lambda s,t:s+t)==7
