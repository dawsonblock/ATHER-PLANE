# v2.38.5 GPU RNG execution repair

The v2.38.4 P2 attempt stopped during checkpoint resume, before model training. `torch.load(..., map_location="cuda")` moved the stored CPU RNG ByteTensor to the GPU. The RNG restore helper passed that CUDA tensor to the CPU generator, which raises `TypeError: RNG state must be a torch.ByteTensor`.

The helper now places CPU and CUDA generator state tensors in CPU memory before restoring them; the model and optimizer checkpoint mapping is unchanged. A CUDA test saves both generators, reloads with GPU mapping, and verifies the next random values match. All evidence from v2.38.4 remains attached to its old build. Start a new `runs/v2_38_5` qualification at P0.
