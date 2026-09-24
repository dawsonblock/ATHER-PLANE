from __future__ import annotations
import torch


def train_arbitrator_step(arbitrator, optimizer, diagnostics: torch.Tensor,
                           planner_return: torch.Tensor, actor_return: torch.Tensor,
                           planning_cost: torch.Tensor | float = 0.0, margin: float = 0.0):
    """Supervise the learned gate from measured planner benefit.

    `diagnostics` is [B,5] in the same order used by LearnedArbitrator.features.
    A planner is labeled beneficial when its measured return exceeds actor return
    by planning cost + margin.
    """
    pc=torch.as_tensor(planning_cost,dtype=planner_return.dtype,device=planner_return.device)
    beneficial=(planner_return > actor_return + pc + float(margin)).float()
    p=torch.sigmoid(arbitrator.net(diagnostics)).squeeze(-1)
    loss=torch.nn.functional.binary_cross_entropy(p,beneficial)
    optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
    with torch.no_grad(): acc=((p>=0.5)==(beneficial>=0.5)).float().mean()
    return {"loss":float(loss.item()),"accuracy":float(acc.item()),"positive_rate":float(beneficial.mean().item())}
