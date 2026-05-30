import math

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .injection_utils import (
    find_injection_token,
    find_injection_neighbors,
    inject_at_marked_positions,
)
from .normalizer import normalize_activation, resolve_embed_scale

AV_PROMPT_TEMPLATE = (
    "Describe the concept this activation vector encodes: {injection_char}\n"
    "<explanation>"
)


class ActivationVerbalizer:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype      = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self.dtype, device_map="auto"
        )
        self.model.eval()

        hf_cfg               = AutoConfig.from_pretrained(model_name)
        text_cfg             = getattr(hf_cfg, "text_config", hf_cfg)
        self.d_model: int    = text_cfg.hidden_size
        model_type: str      = getattr(text_cfg, "model_type", "")
        self.embed_scale     = resolve_embed_scale(model_type, self.d_model)
        self.injection_scale = math.sqrt(self.d_model)

        self.injection_char, self.injection_token_id = find_injection_token(self.tokenizer)

        self._prompt    = AV_PROMPT_TEMPLATE.format(injection_char=self.injection_char)
        self._input_ids = self.tokenizer.encode(
            self._prompt, return_tensors="pt", add_special_tokens=True
        )
        self.left_neighbor_id, self.right_neighbor_id = find_injection_neighbors(
            self._input_ids, self.injection_token_id
        )

        print(f"Verbalizer: {model_name} (d={self.d_model}, char={self.injection_char!r})")

    def verbalize(self, activation: torch.Tensor, max_new_tokens: int = 80) -> str:
        input_ids   = self._input_ids.to(self.device)
        embed_layer = self.model.get_input_embeddings()

        with torch.no_grad():
            embeds = embed_layer(input_ids).to(self.dtype)
        embeds = embeds * self.embed_scale

        v        = activation.to(self.device).to(torch.float32)
        v_scaled = normalize_activation(v.unsqueeze(0), self.injection_scale)

        embeds = inject_at_marked_positions(
            input_ids, embeds, v_scaled,
            self.injection_token_id,
            self.left_neighbor_id,
            self.right_neighbor_id,
        )

        attention_mask = torch.ones(1, embeds.shape[1], dtype=torch.long).to(self.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=embeds.to(self.dtype),
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def verbalize_batch(self, activations: list[torch.Tensor], max_new_tokens: int = 80) -> list[str]:
        if not activations:
            return []
        input_ids   = self._input_ids.to(self.device)
        embed_layer = self.model.get_input_embeddings()

        with torch.no_grad():
            base_embeds = embed_layer(input_ids).to(self.dtype) * self.embed_scale

        batch_embeds = []
        for activation in activations:
            embeds   = base_embeds.clone()
            v_scaled = normalize_activation(
                activation.to(self.device).to(torch.float32).unsqueeze(0), self.injection_scale
            )
            embeds = inject_at_marked_positions(
                input_ids, embeds, v_scaled,
                self.injection_token_id, self.left_neighbor_id, self.right_neighbor_id,
            )
            batch_embeds.append(embeds)

        batch_embeds   = torch.cat(batch_embeds, dim=0)
        attention_mask = torch.ones(len(activations), batch_embeds.shape[1], dtype=torch.long).to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=batch_embeds.to(self.dtype),
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        return [self.tokenizer.decode(out, skip_special_tokens=True).strip() for out in output_ids]
