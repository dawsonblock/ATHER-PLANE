import torch
from awa.planning.learned_gate import LearnedArbitrator
from awa.planning.arbitrator_train import train_arbitrator_step
from awa.skills.compiler import SkillDataset, distill_discrete_skill


def test_arbitrator_training_step():
    gate=LearnedArbitrator(8); opt=torch.optim.Adam(gate.parameters(),lr=.05)
    x=torch.tensor([[.1,.1,.8,.1,.1],[.9,.7,.1,.8,1.0]],dtype=torch.float32)
    out=train_arbitrator_step(gate,opt,x,torch.tensor([2.,0.]),torch.tensor([0.,1.]),planning_cost=.1)
    assert out['loss']>=0 and 0<=out['accuracy']<=1


def test_skill_distillation_memorizes_easy_mapping():
    states=torch.tensor([[1.,0.],[.9,.1],[0.,1.],[.1,.9]]).repeat(16,1)
    actions=torch.tensor([0,0,1,1]).repeat(16)
    model,meta=distill_discrete_skill(SkillDataset(states,actions),2,hidden_dim=16,epochs=40,lr=.02,batch_size=16)
    assert meta['train_accuracy']>.95
