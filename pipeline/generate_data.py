import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datasets
import torch
from datasets import load_dataset
from tqdm import tqdm

datasets.logging.set_verbosity_error()

from nla.subject_model import SubjectModel

MODEL_NAME       = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_INDEX      = 16
OUTPUT_PATH      = Path("artifacts/data/activations.pt")
N_WIKIPEDIA      = 300
N_ALPACA         = 200
MIN_LEFT_TOKENS  = 10
MAX_POS_PER_TEXT = 10


def load_texts() -> list[str]:
    texts: list[str] = []

    print("Loading Wikipedia")
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True,
                        trust_remote_code=True)
    for article in tqdm(wiki, total=N_WIKIPEDIA, desc="Wikipedia"):
        para = article["text"].split("\n\n")[0].strip()
        if len(para) > 150:
            texts.append(para)
        if len(texts) >= N_WIKIPEDIA:
            break

    print("Loading Alpaca")
    alpaca = load_dataset("tatsu-lab/alpaca", split="train", keep_in_memory=True)
    alpaca_texts = []
    for ex in tqdm(alpaca, total=N_ALPACA, desc="Alpaca"):
        instruction = ex.get("instruction", "").strip()
        output      = ex.get("output", "").strip()
        if instruction and output and len(output) > 80:
            alpaca_texts.append(f"{instruction}\n{output}")
        if len(alpaca_texts) >= N_ALPACA:
            break
    texts.extend(alpaca_texts)

    return texts


def sample_positions(token_ids: list[int], max_pos: int, min_left: int) -> list[int]:
    eligible = [i for i in range(len(token_ids)) if i >= min_left]
    if not eligible:
        return []
    step = max(1, len(eligible) // max_pos)
    return eligible[::step][:max_pos]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    texts = load_texts()
    print(f"{len(texts)} texts loaded")

    print("Loading subject model")
    model = SubjectModel(MODEL_NAME)

    dataset = []
    for text in tqdm(texts, desc="Extracting activations"):
        tokens    = model.extract(text, LAYER_INDEX)
        token_ids = [t.token_id for t in tokens]
        positions = sample_positions(token_ids, MAX_POS_PER_TEXT, MIN_LEFT_TOKENS)

        for pos in positions:
            tok = tokens[pos]
            dataset.append({
                "text":       text,
                "position":   tok.position,
                "token_text": tok.text,
                "activation": tok.activation.clone(),
            })

    torch.save(dataset, OUTPUT_PATH)
    print(f"Done. {len(dataset)} activations saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
