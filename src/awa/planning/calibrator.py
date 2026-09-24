from __future__ import annotations
import numpy as np
import torch
from torch import nn


class LogisticUncertaintyCalibrator(nn.Module):
    """Calibrates raw non-negative disagreement into P(prediction failure).

    The model is intentionally tiny: sigmoid(a * log1p(u) + b) with a constrained
    positive slope. `fit_numpy` is dependency-free apart from PyTorch.
    """
    def __init__(self):
        super().__init__()
        self.raw_slope = nn.Parameter(torch.tensor(0.0))
        self.bias = nn.Parameter(torch.tensor(-1.0))

    def forward(self, uncertainty: torch.Tensor) -> torch.Tensor:
        slope = torch.nn.functional.softplus(self.raw_slope) + 1e-4
        return torch.sigmoid(slope * torch.log1p(uncertainty.clamp_min(0)) + self.bias)

    def fit_numpy(self, uncertainty, failures, steps: int = 250, lr: float = 0.05):
        x=torch.as_tensor(np.asarray(uncertainty),dtype=torch.float32).reshape(-1)
        y=torch.as_tensor(np.asarray(failures),dtype=torch.float32).reshape(-1)
        if x.numel()!=y.numel() or x.numel()==0: raise ValueError("non-empty equal-length arrays required")
        opt=torch.optim.Adam(self.parameters(),lr=lr)
        for _ in range(int(steps)):
            p=self(x); loss=torch.nn.functional.binary_cross_entropy(p,y)
            opt.zero_grad(); loss.backward(); opt.step()
        return float(loss.item())
