import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

DATA_PATH   = Path("artifacts/data/eval_results.pt")
OUTPUT_DIR  = Path("docs/figures")


def load_results() -> list[dict]:
    raw = torch.load(DATA_PATH, map_location="cpu")
    return raw["results"] if isinstance(raw, dict) else raw


def plot_cos_sim_distribution(results: list[dict], out_path: Path) -> None:
    buckets     = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    labels      = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    all_cos     = [r["cos_sim"] for r in results]
    counts      = [sum(1 for v in all_cos if lo <= v < hi) for lo, hi in buckets]
    pcts        = [100 * c / len(all_cos) for c in counts]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, pcts, color="#4C72B0", edgecolor="white", linewidth=0.8)

    for bar, pct, cnt in zip(bars, pcts, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                f"{cnt}\n({pct:.0f}%)", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.set_ylabel("Percentage of tokens (%)", fontsize=11)
    ax.set_title("cos_sim distribution — 200 eval tokens (seed 42)", fontsize=12)
    ax.set_ylim(0, max(pcts) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_content_vs_function(results: list[dict], out_path: Path) -> None:
    buckets  = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]
    labels   = ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"]
    content  = [r["cos_sim"] for r in results if r["token_type"] == "content"]
    function = [r["cos_sim"] for r in results if r["token_type"] == "function"]

    def bucket_pcts(values):
        n = len(values)
        return [100 * sum(1 for v in values if lo <= v < hi) / n for lo, hi in buckets]

    c_pcts = bucket_pcts(content)
    f_pcts = bucket_pcts(function)

    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars_c = ax.bar(x - width / 2, c_pcts, width, label=f"Content ({len(content)})",
                    color="#4C72B0", edgecolor="white", linewidth=0.8)
    bars_f = ax.bar(x + width / 2, f_pcts, width, label=f"Function ({len(function)})",
                    color="#DD8452", edgecolor="white", linewidth=0.8)

    for bar, pct in zip(list(bars_c) + list(bars_f), c_pcts + f_pcts):
        if pct > 1:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{pct:.0f}%", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Cosine similarity", fontsize=11)
    ax.set_ylabel("Percentage of tokens (%)", fontsize=11)
    ax.set_title("cos_sim by token type — content vs function words", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(c_pcts + f_pcts) * 1.25)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    print(f"Loaded {len(results)} results")

    plot_cos_sim_distribution(results, OUTPUT_DIR / "cos_sim_distribution.png")
    plot_content_vs_function(results,  OUTPUT_DIR / "content_vs_function.png")


if __name__ == "__main__":
    main()
