import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from nla.activation_reconstructor import AR_PROMPT_TEMPLATE, ActivationReconstructor
from nla.activation_verbalizer import ActivationVerbalizer
from nla.checkpoint_utils import save_chunked
from nla.injection_utils import inject_at_marked_positions
from nla.normalizer import normalize_activation

MODEL_NAME        = "Qwen/Qwen2.5-0.5B-Instruct"
EXPLANATIONS_PATH = Path("artifacts/data/explanations.pt")
ACTIVATIONS_PATH  = Path("artifacts/data/activations.pt")
SAVE_DIR          = Path("artifacts/checkpoints")

BATCH_SIZE   = 4
AR_SFT_STEPS = 100
AV_SFT_STEPS = 100
RL_STEPS     = 300
LR_AV = 5e-6
LR_AR = 1e-4


def ar_mse_loss(ar, description: str, activation: torch.Tensor) -> torch.Tensor:
    ids = ar.tokenizer(
        AR_PROMPT_TEMPLATE.format(explanation=description),
        return_tensors="pt", add_special_tokens=True, truncation=True, max_length=512,
    )["input_ids"].to(ar.device)
    h    = ar.backbone.model(ids, use_cache=False).last_hidden_state
    pred = ar.value_head(h[0, -1]).float()
    gold = normalize_activation(activation.float().to(ar.device), ar.mse_scale)
    return F.mse_loss(normalize_activation(pred, ar.mse_scale), gold)


def av_sft_loss(av, activation: torch.Tensor, explanation: str) -> torch.Tensor:
    input_ids   = av._input_ids.to(av.device)
    embed_layer = av.model.get_input_embeddings()

    with torch.no_grad():
        prompt_embeds = embed_layer(input_ids).to(av.dtype) * av.embed_scale

    v_scaled = normalize_activation(
        activation.to(av.device).float().unsqueeze(0), av.injection_scale
    )
    prompt_embeds = inject_at_marked_positions(
        input_ids, prompt_embeds, v_scaled,
        av.injection_token_id, av.left_neighbor_id, av.right_neighbor_id,
    )

    exp_ids = av.tokenizer(
        explanation + av.tokenizer.eos_token,
        return_tensors="pt", add_special_tokens=False,
    )["input_ids"].to(av.device)

    with torch.no_grad():
        exp_embeds = embed_layer(exp_ids).to(av.dtype) * av.embed_scale

    full_embeds  = torch.cat([prompt_embeds, exp_embeds], dim=1)
    logits       = av.model(inputs_embeds=full_embeds, use_cache=False).logits
    prompt_len   = prompt_embeds.shape[1]
    n_exp        = exp_ids.shape[1]
    pred_logits  = logits[0, prompt_len - 1 : prompt_len - 1 + n_exp]
    return F.cross_entropy(pred_logits, exp_ids[0])


def compute_log_prob(av, activation: torch.Tensor, desc_ids: torch.Tensor, model=None) -> torch.Tensor:
    if model is None:
        model = av.model
    input_ids   = av._input_ids.to(av.device)
    embed_layer = model.get_input_embeddings()

    with torch.no_grad():
        prompt_embeds = embed_layer(input_ids).to(av.dtype) * av.embed_scale

    v_scaled = normalize_activation(
        activation.to(av.device).float().unsqueeze(0), av.injection_scale
    )
    prompt_embeds = inject_at_marked_positions(
        input_ids, prompt_embeds, v_scaled,
        av.injection_token_id, av.left_neighbor_id, av.right_neighbor_id,
    )

    with torch.no_grad():
        desc_embeds = embed_layer(desc_ids.to(av.device)).to(av.dtype) * av.embed_scale

    full_embeds = torch.cat([prompt_embeds, desc_embeds], dim=1)
    logits      = model(inputs_embeds=full_embeds, use_cache=False).logits
    prompt_len  = prompt_embeds.shape[1]
    n_desc      = desc_ids.shape[1]
    gen_logits  = logits[0, prompt_len - 1 : prompt_len - 1 + n_desc]
    log_probs   = F.log_softmax(gen_logits, dim=-1)
    return log_probs[torch.arange(n_desc), desc_ids[0].to(av.device)].sum()


def main() -> None:
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading models")
    av = ActivationVerbalizer(MODEL_NAME)
    ar = ActivationReconstructor(MODEL_NAME, num_critic_layers=12)

    nn.init.eye_(ar.value_head.weight)
    for p in ar.backbone.parameters():
        p.requires_grad_(False)

    opt_av = torch.optim.AdamW(av.model.parameters(),      lr=LR_AV)
    opt_ar = torch.optim.AdamW(ar.value_head.parameters(), lr=LR_AR)

    if EXPLANATIONS_PATH.exists():
        dataset          = torch.load(EXPLANATIONS_PATH)
        has_explanations = all("explanation" in ex and ex["explanation"] for ex in dataset)
        print(f"{len(dataset)} examples loaded")
    else:
        dataset          = torch.load(ACTIVATIONS_PATH)
        has_explanations = False
        print(f"{len(dataset)} examples loaded (no explanations, skipping SFT)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = [{**ex, "activation": ex["activation"].to(device)} for ex in dataset]
    print(f"Activations pre-loaded to {device}")

    if has_explanations:
        for step in tqdm(range(1, AR_SFT_STEPS + 1), desc="Stage 1: training AR"):
            batch = random.sample(dataset, BATCH_SIZE)
            loss  = sum(ar_mse_loss(ar, ex["explanation"], ex["activation"]) for ex in batch) / BATCH_SIZE
            opt_ar.zero_grad()
            loss.backward()
            opt_ar.step()
    else:
        print("Stage 1: skipped (no explanations)")

    if has_explanations:
        for step in tqdm(range(1, AV_SFT_STEPS + 1), desc="Stage 2: training AV"):
            batch = random.sample(dataset, BATCH_SIZE)
            loss  = sum(av_sft_loss(av, ex["activation"], ex["explanation"]) for ex in batch) / BATCH_SIZE
            opt_av.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(av.model.parameters(), 1.0)
            opt_av.step()
    else:
        print("Stage 2: skipped (no explanations)")

    if not has_explanations:
        for step in tqdm(range(1, 101), desc="AR warmup"):
            batch = random.sample(dataset, BATCH_SIZE)
            with torch.no_grad():
                descs = [av.verbalize(ex["activation"]) for ex in batch]
            loss = sum(ar_mse_loss(ar, d, ex["activation"]) for d, ex in zip(descs, batch))
            (loss / BATCH_SIZE).backward()
            opt_ar.step()
            opt_ar.zero_grad()

    baseline    = 0.0
    mean_reward = 0.0

    pbar = tqdm(range(1, RL_STEPS + 1), desc="Stage 3: RL")
    for step in pbar:
        batch = random.sample(dataset, BATCH_SIZE)

        with torch.no_grad():
            descriptions  = av.verbalize_batch([ex["activation"] for ex in batch])
            rewards       = [math.log(1 + ar.score(d, ex["activation"])["cosine_similarity"])
                             for d, ex in zip(descriptions, batch)]
            desc_ids_list = [av.tokenizer(d, return_tensors="pt")["input_ids"] for d in descriptions]

        mean_reward = sum(rewards) / len(rewards)
        baseline    = 0.9 * baseline + 0.1 * mean_reward

        av_loss = sum(
            -(r - baseline) * compute_log_prob(av, ex["activation"], ids)
            for ex, r, ids in zip(batch, rewards, desc_ids_list)
        ) / BATCH_SIZE
        opt_av.zero_grad()
        av_loss.backward()
        torch.nn.utils.clip_grad_norm_(av.model.parameters(), 1.0)
        opt_av.step()

        ar_loss = sum(
            ar_mse_loss(ar, d, ex["activation"]) for d, ex in zip(descriptions, batch)
        ) / BATCH_SIZE
        opt_ar.zero_grad()
        ar_loss.backward()
        opt_ar.step()

        pbar.set_postfix(cos=f"{math.exp(mean_reward)-1:.3f}", av=f"{av_loss.item():.4f}", ar=f"{ar_loss.item():.4f}")

    save_chunked(av.model.state_dict(),    SAVE_DIR / "av_model")
    torch.save(ar.value_head.state_dict(), SAVE_DIR / "ar_value_head.pt")
    print(f"\nSaved to {SAVE_DIR}/")
    print(f"Final cos_sim: {mean_reward:.3f}")


if __name__ == "__main__":
    main()
