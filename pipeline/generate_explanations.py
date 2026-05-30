import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA_PATH      = Path("artifacts/data/activations.pt")
OUTPUT_PATH    = Path("artifacts/data/explanations.pt")
MODEL_NAME     = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_NEW_TOKENS = 100
BATCH_SIZE     = 32


def build_prompt(text: str, token_text: str, position: int) -> str:
    return (
        f"<|im_start|>system\n"
        f"You describe what a language model is encoding at a specific token position.<|im_end|>\n"
        f"<|im_start|>user\n"
        f"A language model processes this text:\n\"{text}\"\n\n"
        f"At position {position}, it has just read the token '{token_text.strip()}'.\n"
        f"In 2-3 sentences, describe what concepts or relationships this model is most "
        f"likely encoding at this position, based only on the left context it has seen.<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


def main() -> None:
    print(f"Loading {MODEL_NAME}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.padding_side = "left"
    model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=dtype, device_map="auto")
    model.generation_config.temperature = None
    model.generation_config.top_p       = None
    model.generation_config.top_k       = None
    model.eval()

    dataset = torch.load(DATA_PATH)
    print(f"{len(dataset)} examples\n")

    results = []
    for i in tqdm(range(0, len(dataset), BATCH_SIZE), desc="Generating explanations"):
        batch   = dataset[i : i + BATCH_SIZE]
        prompts = [build_prompt(ex["text"], ex["token_text"], ex["position"]) for ex in batch]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        for ex, out in zip(batch, output_ids):
            new_ids     = out[prompt_len:]
            explanation = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
            results.append({**ex, "explanation": explanation})

    torch.save(results, OUTPUT_PATH)
    print(f"Done. {len(results)} examples saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
