from __future__ import annotations
from dataclasses import dataclass
import torch
from torch import nn

@dataclass
class TemporalState:
    local: torch.Tensor
    global_ctx: torch.Tensor
    event_buffer: torch.Tensor

    @property
    def vector(self):
        return torch.cat([self.local, self.global_ctx], dim=-1)

class LocalGlobalTemporalCore(nn.Module):
    """Fast recurrent local state plus lower-rate attention-based global context."""
    def __init__(self, latent_dim: int, action_dim: int, local_dim: int = 256,
                 global_dim: int = 256, event_slots: int = 8, global_stride: int = 4,
                 heads: int = 4):
        super().__init__()
        self.local_dim=local_dim; self.global_dim=global_dim; self.event_slots=event_slots
        self.global_stride=max(1,int(global_stride))
        self.local_cell=nn.GRUCell(latent_dim+action_dim, local_dim)
        self.event_proj=nn.Linear(local_dim, global_dim)
        layer=nn.TransformerEncoderLayer(d_model=global_dim,nhead=heads,dim_feedforward=global_dim*4,batch_first=True,activation='gelu')
        self.global_encoder=nn.TransformerEncoder(layer,num_layers=1)
        self.global_gate=nn.Sequential(nn.Linear(global_dim*2,global_dim),nn.Sigmoid())

    def initial(self,batch:int,device,dtype=torch.float32):
        return TemporalState(
            torch.zeros(batch,self.local_dim,device=device,dtype=dtype),
            torch.zeros(batch,self.global_dim,device=device,dtype=dtype),
            torch.zeros(batch,self.event_slots,self.global_dim,device=device,dtype=dtype),
        )

    def forward(self,state:TemporalState,latent:torch.Tensor,action:torch.Tensor,step:int):
        local=self.local_cell(torch.cat([latent,action],-1),state.local)
        event=self.event_proj(local)
        buf=torch.cat([state.event_buffer[:,1:],event.unsqueeze(1)],1)
        global_ctx=state.global_ctx
        if step % self.global_stride == 0:
            enc=self.global_encoder(buf)
            proposal=enc[:,-1]
            gate=self.global_gate(torch.cat([global_ctx,proposal],-1))
            global_ctx=gate*proposal+(1-gate)*global_ctx
        return TemporalState(local,global_ctx,buf)
