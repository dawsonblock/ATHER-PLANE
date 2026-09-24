from __future__ import annotations
import numpy as np
import torch
from torch import nn
from awa.experimental.connectome.graph import CSRGraph


class FixedSparseCircuit(nn.Module):
    """Differentiable fixed-topology graph block exported from a connectome subgraph."""
    def __init__(self, graph: CSRGraph, learnable_gain: bool=True, activation: str="tanh"):
        super().__init__(); self.nodes=graph.nodes
        src=np.repeat(np.arange(graph.nodes,dtype=np.int64),np.diff(graph.row_ptr))
        self.register_buffer("src",torch.as_tensor(src,dtype=torch.long)); self.register_buffer("dst",torch.as_tensor(graph.dst,dtype=torch.long))
        base=torch.as_tensor(graph.weight,dtype=torch.float32)
        self.register_buffer("base_weight",base)
        self.gain=nn.Parameter(torch.ones_like(base)) if learnable_gain else None
        self.activation=activation

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1]!=self.nodes: raise ValueError(f"expected last dim {self.nodes}")
        w=self.base_weight if self.gain is None else self.base_weight*self.gain
        msgs=x[...,self.src]*w
        out=torch.zeros_like(x)
        # flatten leading dimensions for index_add on final axis
        flat=out.reshape(-1,self.nodes); mflat=msgs.reshape(-1,msgs.shape[-1])
        for i in range(flat.shape[0]): flat[i].index_add_(0,self.dst,mflat[i])
        if self.activation=="tanh": return torch.tanh(out)
        if self.activation=="relu": return torch.relu(out)
        return out
