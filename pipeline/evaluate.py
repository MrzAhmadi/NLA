import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import random

import numpy as np
import torch
from tqdm import tqdm

from nla.activation_reconstructor import ActivationReconstructor
from nla.activation_verbalizer import ActivationVerbalizer

MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
DATA_PATH   = Path("artifacts/data/explanations.pt")
AV_CKPT     = Path("artifacts/checkpoints/av_model.pt")
AR_CKPT     = Path("artifacts/checkpoints/ar_value_head.pt")
OUT_PATH    = Path("artifacts/data/eval_results.pt")
N_EVAL       = 200
EVAL_BATCH   = 8
RANDOM_SEED = 42

_FUNCTION = {
    "the", "a", "an", "and", "or", "but", "of", "in", "on", "at", "to",
    "for", "with", "by", "is", "are", "was", "were", "be", "been", "has",
    "have", "had", "do", "does", "did", "will", "would", "could", "should",
    "may", "might", "not", "no", "it", "its", "this", "that", "which",
    "who", "what", ",", ".", ":", ";", "(", ")", '"', "'", "-", "\n", " ",
    "as", "from", "also", "more", "their",
}


def token_type(text: str) -> str:
    return "function" if text.strip().lower() in _FUNCTION or len(text.strip()) <= 1 else "content"


def load_models() -> tuple[ActivationVerbalizer, ActivationReconstructor]:
    av = ActivationVerbalizer(MODEL_NAME)
    ar = ActivationReconstructor(MODEL_NAME, num_critic_layers=12)
    if AV_CKPT.exists():
        av.model.load_state_dict(torch.load(AV_CKPT, map_location=av.device))
        print("  verbalizer checkpoint loaded")
    else:
        print("  no verbalizer checkpoint, using untrained model")
    if AR_CKPT.exists():
        ar.value_head.load_state_dict(torch.load(AR_CKPT, map_location=ar.device))
        print("  reconstructor checkpoint loaded")
    else:
        print("  no reconstructor checkpoint, value head is random")
    return av, ar


def print_distribution(values: list[float], label: str) -> None:
    buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\n  {label}:")
    for lo, hi in buckets:
        n   = sum(1 for v in values if lo <= v < hi)
        bar = "#" * int(n / max(len(values), 1) * 30)
        print(f"    {lo:.1f}-{hi:.1f}  {bar:<30s} {n} ({100 * n / len(values):.0f}%)")


def main() -> None:
    print("Loading models")
    av, ar = load_models()

    dataset = torch.load(DATA_PATH)
    print(f"\n{len(dataset)} examples in dataset")

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = [{**ex, "activation": ex["activation"].to(device)} for ex in dataset]

    random.seed(RANDOM_SEED)
    sample = random.sample(dataset, min(N_EVAL or len(dataset), len(dataset)))
    print(f"Evaluating {len(sample)} samples\n")

    results = []
    pbar = tqdm(range(0, len(sample), EVAL_BATCH), desc="Evaluating")
    for i in pbar:
        batch        = sample[i : i + EVAL_BATCH]
        descriptions = av.verbalize_batch([ex["activation"] for ex in batch])

        for ex, description in zip(batch, descriptions):
            metrics = ar.score(description, ex["activation"])
            cos     = metrics["cosine_similarity"]
            results.append({
                "token_text":  ex["token_text"],
                "position":    ex["position"],
                "text":        ex["text"][:80],
                "cos_sim":     cos,
                "fve":         2 * cos - 1,
                "token_type":  token_type(ex["token_text"]),
                "description": description,
            })

        pbar.set_postfix(cos=f"{np.mean([r['cos_sim'] for r in results]):.3f}")

    all_cos  = [r["cos_sim"] for r in results]
    all_fve  = [r["fve"]     for r in results]
    content  = [r for r in results if r["token_type"] == "content"]
    function = [r for r in results if r["token_type"] == "function"]

    print("\n--- Results ---")

    print(f"\nAll tokens ({len(results)})")
    print(f"  cos_sim: mean={np.mean(all_cos):.3f}, median={np.median(all_cos):.3f}, std={np.std(all_cos):.3f}")
    print(f"  FVE:     mean={np.mean(all_fve):.3f}, median={np.median(all_fve):.3f}")
    print_distribution(all_cos, "cos_sim distribution")

    if content:
        c_cos = [r["cos_sim"] for r in content]
        c_fve = [r["fve"]     for r in content]
        print(f"\nContent tokens ({len(content)})")
        print(f"  cos_sim: mean={np.mean(c_cos):.3f}, median={np.median(c_cos):.3f}, std={np.std(c_cos):.3f}")
        print(f"  FVE:     mean={np.mean(c_fve):.3f}, median={np.median(c_fve):.3f}")
        print_distribution(c_cos, "cos_sim distribution")

    if function:
        f_cos = [r["cos_sim"] for r in function]
        f_fve = [r["fve"]     for r in function]
        print(f"\nFunction tokens ({len(function)})")
        print(f"  cos_sim: mean={np.mean(f_cos):.3f}, median={np.median(f_cos):.3f}, std={np.std(f_cos):.3f}")
        print(f"  FVE:     mean={np.mean(f_fve):.3f}, median={np.median(f_fve):.3f}")
        print_distribution(f_cos, "cos_sim distribution")

    by_cos = sorted(results, key=lambda r: r["cos_sim"])

    print("\nBest 5:")
    for r in by_cos[-5:][::-1]:
        print(f"  {r['cos_sim']:.3f}  {r['token_text']!r:12s}  {r['description'][:70]}")

    print("\nWorst 5:")
    for r in by_cos[:5]:
        print(f"  {r['cos_sim']:.3f}  {r['token_text']!r:12s}  {r['description'][:70]}")

    print(f"\n  FVE all:     {np.mean(all_fve):.3f}")
    print(f"  FVE content: {np.mean([r['fve'] for r in content]):.3f}")
    print(f"  cos_sim all: {np.mean(all_cos):.3f}")
    print(f"  cos_sim cont:{np.mean([r['cos_sim'] for r in content]):.3f}")

    torch.save(results, OUT_PATH)
    print(f"\nResults saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
