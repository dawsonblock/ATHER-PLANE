from __future__ import annotations

import torch
import pytest

from awa.v2.rng_state import capture_rng_state, restore_rng_state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA host")
def test_checkpoint_rng_restores_after_gpu_map_location(tmp_path):
    torch.manual_seed(2385)
    torch.cuda.manual_seed_all(2385)
    checkpoint=tmp_path/"rng.pt"
    torch.save(capture_rng_state(),checkpoint)
    expected_cpu=torch.rand(4)
    expected_cuda=torch.rand(4,device="cuda")
    loaded=torch.load(checkpoint,map_location="cuda",weights_only=False)
    assert loaded["torch_cpu"].is_cuda
    torch.rand(4)
    torch.rand(4,device="cuda")
    assert restore_rng_state(loaded)
    assert torch.equal(torch.rand(4),expected_cpu)
    assert torch.equal(torch.rand(4,device="cuda"),expected_cuda)
