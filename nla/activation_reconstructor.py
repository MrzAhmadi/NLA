import math
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .normalizer import normalize_activation

AR_PROMPT_TEMPLATE = (
    "Describe the concept this activation vector encodes: [PLACEHOLDER]\n"
    "<explanation>{explanation}</explanation>"
)

_FINAL_LN_ATTRS = ("norm", "final_layernorm", "ln_f")


class ActivationReconstructor(nn.Module):
    def __init__(self, model_name: str, num_critic_layers: Optional[int] = None):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype  = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        hf_cfg       = AutoConfig.from_pretrained(model_name)
        text_cfg     = getattr(hf_cfg, "text_config", hf_cfg)
        self.d_model = text_cfg.hidden_size
        total_layers = text_cfg.num_hidden_layers

        if num_critic_layers is None:
            num_critic_layers = total_layers // 2
        self.num_critic_layers = num_critic_layers

        backbone         = AutoModelForCausalLM.from_pretrained(model_name, dtype=self.dtype)
        inner            = None
        layers_attr_found = None

        for container in ("model", "transformer"):
            cand = getattr(backbone, container, None)
            if cand is None:
                continue
            for layers_attr in ("layers", "h", "blocks"):
                if hasattr(cand, layers_attr):
                    inner             = cand
                    layers_attr_found = layers_attr
                    break
            if inner is not None:
                break

        assert inner is not None, f"Cannot find decoder layers in {type(backbone).__name__}."

        all_layers = getattr(inner, layers_attr_found)
        setattr(inner, layers_attr_found, nn.ModuleList(list(all_layers)[:num_critic_layers]))

        for attr in _FINAL_LN_ATTRS:
            if hasattr(inner, attr):
                setattr(inner, attr, nn.Identity())
                break

        backbone.lm_head = nn.Identity()
        self.backbone    = backbone

        self.value_head  = nn.Linear(self.d_model, self.d_model, bias=False, dtype=self.dtype)
        self.mse_scale   = math.sqrt(self.d_model)

        self.to(self.device)
        self.eval()

        print(f"Reconstructor: {model_name} ({num_critic_layers}/{total_layers} layers, d={self.d_model})")

    @torch.inference_mode()
    def reconstruct(self, explanation: str) -> torch.Tensor:
        prompt = AR_PROMPT_TEMPLATE.format(explanation=explanation)
        ids    = self.tokenizer(
            prompt, return_tensors="pt", add_special_tokens=True,
            truncation=True, max_length=512,
        )["input_ids"].to(self.device)
        h    = self.backbone.model(ids, use_cache=False).last_hidden_state
        last = h[0, -1, :]
        return self.value_head(last).float().cpu()

    def score(self, explanation: str, original: torch.Tensor) -> dict[str, float]:
        pred   = self.reconstruct(explanation).float()
        gold   = original.float().cpu()
        pred_n = normalize_activation(pred, self.mse_scale)
        gold_n = normalize_activation(gold, self.mse_scale)
        mse    = float(((pred_n - gold_n) ** 2).mean())
        cos    = float(nn.functional.cosine_similarity(pred_n.unsqueeze(0), gold_n.unsqueeze(0)))
        return {"mse": mse, "cosine_similarity": cos}
