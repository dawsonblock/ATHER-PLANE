from __future__ import annotations
import copy
import numpy as np
import torch
from torch.utils.data import DataLoader,Subset
import torch.nn.functional as F
from awa.v2.ensemble import WorldModelEnsemble
from .belief import GameBeliefSequenceDataset,GameBeliefWorldTrainer


def train_bootstrap_world_ensemble(sequence_dataset: GameBeliefSequenceDataset, encoder, base_world, *,
                                   members=3, epochs=1, batch_size=64, lr=5e-4, seed=0):
    """Train dynamics heads on bootstrap resamples in one shared belief space.

    The temporal encoder is frozen so ensemble disagreement measures uncertainty in
    dynamics, not arbitrary disagreement between unrelated latent coordinate systems.
    """
    if int(members)<2: raise ValueError('members must be >= 2')
    device=next(encoder.parameters()).device
    was_training=encoder.training; requires=[p.requires_grad for p in encoder.parameters()]
    encoder.eval()
    for p in encoder.parameters(): p.requires_grad=False
    out=[]
    try:
        for mi in range(int(members)):
            world=copy.deepcopy(base_world).to(device); world.train()
            opt=torch.optim.AdamW(world.parameters(),lr=float(lr))
            helper=GameBeliefWorldTrainer(encoder,world,lr=lr)
            rng=np.random.default_rng(int(seed)+mi*1009)
            ids=rng.integers(0,len(sequence_dataset),size=len(sequence_dataset)).tolist()
            loader=DataLoader(Subset(sequence_dataset,ids),batch_size=min(int(batch_size),len(ids)),shuffle=True)
            for _ in range(int(epochs)):
                for batch in loader:
                    with torch.no_grad(): b,nb=helper.encode_sequence(batch)
                    actions=batch['actions'].to(device); rewards=batch['rewards'].to(device); dones=batch['dones'].to(device)
                    B,T,D=b.shape; fb=b.reshape(B*T,D); fa=actions.reshape(B*T,-1); fn=nb.reshape(B*T,D)
                    nll=world.nll_loss(fb,fa,fn,reduction='mean')
                    _,tr=world.distribution(fb,fa)
                    rp=world.reward(tr).reshape(B,T); cp=torch.sigmoid(world.cont(tr)).reshape(B,T)
                    reward_loss=F.mse_loss(rp,rewards); cont_loss=F.binary_cross_entropy(cp,1.0-dones)
                    loss=nll+reward_loss+.25*cont_loss
                    if 'constraints' in batch:
                        labels=batch['constraints'].to(device).reshape(B*T,-1)
                        loss=loss+.25*F.binary_cross_entropy_with_logits(world.risk.logits(fb,fa),labels)
                    opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(world.parameters(),50.0); opt.step()
            out.append(world.eval())
    finally:
        for p,req in zip(encoder.parameters(),requires): p.requires_grad=req
        encoder.train(was_training)
    return WorldModelEnsemble(out)
