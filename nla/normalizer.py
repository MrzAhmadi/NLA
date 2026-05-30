import math

import torch

_SCALED_EMBED_TYPES = frozenset({"gemma", "gemma2", "gemma3", "gemma3_text", "t5"})


def resolve_embed_scale(model_type: str, hidden_size: int) -> float:
    if model_type in _SCALED_EMBED_TYPES:
        return math.sqrt(hidden_size)
    return 1.0


def normalize_activation(v: torch.Tensor, target_scale: float) -> torch.Tensor:
    norm_fp32 = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return v / (norm_fp32 / target_scale).to(v.dtype)
