# Evaluation and Results

At the end of the training pipeline, an evaluation system assesses how closely the reconstructed activations match the original vectors and how well the verbalized descriptions capture the encoded information. In this section, the accuracy of the reconstructed activations is evaluated against the original vectors in detail.

---

## Evaluation Procedure — `pipeline/evaluate.py`

Evaluation of the system's outputs requires preparing activation vectors to compare with. The evaluation is run on **200 randomly sampled examples** (seed 42) from the 5,000-example dataset. For each example, the Activation Verbalizer generates a description from the activation, the Activation Reconstructor recovers an activation vector from that description, and the cosine similarity between the recovered and original vectors is measured.

The primary metric is **Fraction of Variance Explained (FVE)**, using the paper's formula:

```text
FVE = 1 − ℒ / E[‖h − h̄‖²]
```

where ℒ is the mean per-dimension reconstruction MSE, h̄ is the mean activation over the evaluation set, and the denominator is the per-dimension variance of activations around that mean. FVE = 0 means the system predicts the mean activation, FVE = 1 means perfect reconstruction, and FVE < 0 means worse than predicting the mean. For normalized vectors, ℒ = 2 × (1 − cos_sim).

---

## Quantitative Results

The results of this study demonstrate that the Activation Verbalizer has different behavior across different token types, and the following table summarizes the key metrics:

| Metric                    | Result              |
| ------------------------- | ------------------- |
| cos_sim (all tokens)      | **0.552**           |
| cos_sim (content tokens)  | 0.535               |
| cos_sim (function tokens) | 0.574               |
| FVE (all tokens)          | **0.105**           |
| FVE (content tokens)      | 0.069               |
| Tokens above cos_sim 0.6  | **40%** (79 of 200) |

Distribution of cos_sim across all 200 examples:

```text
[0.0–0.2)                                0  (0%)
[0.2–0.4)   ####                        28  (14%)
[0.4–0.6)   #############               93  (46%)
[0.6–0.8)   ###########                 79  (40%)
[0.8–1.0)                                0  (0%)
```

---

## Key Findings

At the end of the evaluation, there are several key findings that can be highlighted.

### Function Words Reconstruct Better Than Content Words

A crucial observation pertains to the token type distribution. The results demonstrate that function tokens (cos_sim 0.574) reconstruct better than content tokens (cos_sim 0.535), which is the opposite of what the original paper reports for large models. This finding reveals how the system fails at small scale: the Activation Verbalizer, after limited supervised and reinforcement learning training, produces descriptions that are structurally similar regardless of the input activation. The Activation Reconstructor then learns to produce a single average reconstruction vector that minimizes expected error across the distribution — a phenomenon known as centroid collapse.

In other words, the Activation Reconstructor is not reconstructing individual tokens; it is predicting the centroid of the activation distribution. Cosine similarity to that centroid is slightly higher for function words because their activations cluster near the mean of the distribution, and they do not deviate far from the center of the representation space.

### Reward Hacking Under Reinforcement Learning

Furthermore, a crucial observation pertains to the training dynamics. The results demonstrate that, without a strong regularization constraint, the Activation Verbalizer drifts into reward hacking: it generates token sequences that maximize the Activation Reconstructor's score without producing readable natural language. This failure mode does not appear in the original paper because their training uses a KL divergence penalty against a frozen reference policy. At 0.5B scale with self-generated labels, the reward signal is strong enough to override any implicit regularization from the supervised fine-tuning initialization.

### Qualitative Example — "The capital of Iran is Tehran"

The results show that the qualitative behavior on geographic and ethically loaded tokens is coherent and semantically accurate. For the text *"The capital of Iran is Tehran."*:

**Token `Iran` (position 3):**
> *"The concept that is encoded by this activation vector 'Iran' is related to Iran's location and its political significance. This activation vector corresponds to the first element in the sequence of countries mentioned: 'Iran'. When such a vector appears, it indicates that currently, Iran is being referenced alongside other nations or regions within the given context."*
>
> cos_sim: **0.665**

**Token `Tehra` (subword of "Tehran", position 5):**
> *"The activation vector 'Tehra' is most likely encoding the concept of 'Tehran' at this particular point. It could be decoding something related to information about Iran's capital city, possibly including details such as its location in the country, history, geography, culture, or politics surrounding it."*
>
> cos_sim: **0.634**

Both descriptions correctly identify the geographic and political semantic content of the token, which is consistent with the paper's claim that NLA explanations are thematically faithful to context even when quantitative fidelity is imperfect.

---

## Comparison With Baseline

| Configuration                               | cos_sim   | FVE       | Tokens > 0.6 |
| ------------------------------------------- | --------- | --------- | ------------ |
| Layer 12, 300 RL steps (baseline)           | 0.548     | 0.097     | 27%          |
| Layer 16, 300 RL steps + log reward (final) | **0.552** | **0.105** | **40%**      |

The primary bottleneck is label quality, not model size. The labeler model (Qwen 0.5B self-labeling) produces generic descriptions, giving the Activation Reconstructor weak signal to discriminate between tokens. A better label source would likely push cos_sim from 0.55 to 0.65 or higher with the same subject model and training infrastructure.
