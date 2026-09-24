from __future__ import annotations
import torch
from torch import nn


class HuggingFaceVisionAdapter(nn.Module):
    """Optional adapter for locally available Hugging Face vision encoders.

    Requires the optional `hf` dependency group. `model_name_or_path` can be a
    local path; callers control whether Transformers is allowed to download.
    """
    def __init__(self, model_name_or_path: str, output_dim: int, local_files_only: bool = True):
        super().__init__()
        try:
            from transformers import AutoModel, AutoProcessor
        except ImportError as e:
            raise ImportError("Install with `pip install -e '.[hf]'` to use this adapter") from e
        self.processor = AutoProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model.eval()
        for p in self.model.parameters(): p.requires_grad = False
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            raise ValueError("Could not infer hidden_size from model config")
        self.projection = nn.Linear(hidden, output_dim)

    def forward(self, images):
        with torch.no_grad():
            batch = self.processor(images=images, return_tensors="pt")
            device = next(self.model.parameters()).device
            batch = {k:v.to(device) for k,v in batch.items()}
            out = self.model(**batch)
            if hasattr(out, "pooler_output") and out.pooler_output is not None:
                z = out.pooler_output
            else:
                z = out.last_hidden_state.mean(dim=1)
        return self.projection(z)
