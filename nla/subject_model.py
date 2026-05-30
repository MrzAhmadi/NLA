import math
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .normalizer import normalize_activation


@dataclass
class GeneratedToken:
    token_id:   int
    text:       str
    activation: torch.Tensor
    position:   int


class SubjectModel:
    def __init__(self, model_name: str):
        print(f"Loading {model_name}")
        self.model_name = model_name
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype      = torch.bfloat16 if self.device.type == "cuda" else torch.float32
        print(f"Running on {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=self.dtype, device_map="auto",
        )
        self.model.eval()

        hf_cfg             = AutoConfig.from_pretrained(model_name)
        text_cfg           = getattr(hf_cfg, "text_config", hf_cfg)
        self.d_model       = text_cfg.hidden_size
        self.num_layers    = text_cfg.num_hidden_layers
        self.injection_scale = math.sqrt(self.d_model)
        self._captured: Optional[torch.Tensor] = None

    def _get_decoder_layers(self) -> torch.nn.ModuleList:
        for container_attr in ("model", "transformer"):
            inner = getattr(self.model, container_attr, None)
            if inner is not None:
                for layers_attr in ("layers", "h", "blocks"):
                    if hasattr(inner, layers_attr):
                        return getattr(inner, layers_attr)
        raise AttributeError(
            f"Cannot find decoder layers in {type(self.model).__name__}."
        )

    def _register_hook(self, layer_index: int) -> torch.utils.hooks.RemovableHandle:
        layers = self._get_decoder_layers()
        assert 0 <= layer_index < len(layers), (
            f"layer_index={layer_index} out of range; model has {len(layers)} layers."
        )

        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured = hidden.detach().clone()

        return layers[layer_index].register_forward_hook(hook)

    def steer(self, text: str, layer_index: int, position: int,
              new_activation: torch.Tensor, k: int = 8) -> list[dict]:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        new_act   = new_activation.to(self.device).to(self.dtype)

        def hook(_m, _in, out):
            hidden = out[0] if isinstance(out, tuple) else out
            hidden = hidden.clone()
            hidden[:, position, :] = new_act
            return (hidden,) + out[1:] if isinstance(out, tuple) else hidden

        handle = self._get_decoder_layers()[layer_index].register_forward_hook(hook)
        try:
            with torch.no_grad():
                logits = self.model(input_ids, use_cache=False).logits
        finally:
            handle.remove()

        probs = torch.softmax(logits[0, -1].float(), dim=-1)
        top   = probs.topk(k)
        return [
            {"token": self.tokenizer.decode([i.item()]), "prob": round(p.item(), 4)}
            for i, p in zip(top.indices, top.values)
        ]

    def logit_lens(self, text: str, position: int, k: int = 10) -> list[dict]:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            logits = self.model(input_ids, use_cache=False).logits[0, position].float()
        probs = torch.softmax(logits, dim=-1)
        top   = probs.topk(k)
        return [
            {"token": self.tokenizer.decode([i.item()]), "prob": round(p.item(), 4)}
            for i, p in zip(top.indices, top.values)
        ]

    def logit_lens_sweep(self, text: str, position: int, k: int = 3) -> list[dict]:
        input_ids   = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        layers      = self._get_decoder_layers()
        captured: dict[int, torch.Tensor] = {}
        handles = []

        for i, layer in enumerate(layers):
            def _hook(_m, _in, out, idx=i):
                h = out[0] if isinstance(out, tuple) else out
                captured[idx] = h.detach()
            handles.append(layer.register_forward_hook(_hook))

        try:
            with torch.no_grad():
                self.model(input_ids, use_cache=False)
        finally:
            for h in handles:
                h.remove()

        results = []
        for layer_idx in range(len(layers)):
            h = captured[layer_idx][0, position].to(self.dtype).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = self.model.lm_head(self.model.model.norm(h))[0, 0].float()
            probs = torch.softmax(logits, dim=-1)
            top   = probs.topk(k)
            results.append({
                "layer": layer_idx,
                "top": [
                    {"token": self.tokenizer.decode([i.item()]), "prob": round(p.item(), 4)}
                    for i, p in zip(top.indices, top.values)
                ],
            })
        return results

    def extract(self, text: str, layer_index: int) -> list[GeneratedToken]:
        input_ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)
        handle    = self._register_hook(layer_index)
        try:
            with torch.no_grad():
                self.model(input_ids=input_ids, use_cache=False)
        finally:
            handle.remove()

        all_hidden: torch.Tensor = self._captured[0]
        all_ids:    list[int]    = input_ids[0].tolist()

        return [
            GeneratedToken(
                token_id=token_id,
                text=self.tokenizer.decode([token_id]),
                activation=normalize_activation(all_hidden[pos], self.injection_scale).cpu(),
                position=pos,
            )
            for pos, token_id in enumerate(all_ids)
        ]
