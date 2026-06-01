# Training

The main task of the training procedure is optimizing the round-trip reconstruction fidelity, which is done through three sequential stages. Understanding these stages is essential before examining the architecture of the proposed pipeline.

First, the Activation Reconstructor is trained on supervised pairs. Then, the Activation Verbalizer is fine-tuned to reproduce reference descriptions. Finally, the system is trained end-to-end using reinforcement learning.

---

## Architecture

### Activation Verbalizer (AV)

The Activation Verbalizer converts a 896-dimensional activation vector into a natural language description. The most important part of verbalization is the injection mechanism, which places the activation directly into the model's embedding space at a fixed position.

The injection procedure uses a rare CJK character `㈎` (token id 149705) as a placeholder in the prompt. Before the forward pass, the embedding of this token is replaced with the L2-normalized activation vector, scaled to match the average embedding norm (`injection_scale = sqrt(896) ≈ 29.93`). The model then generates a description autoregressively from `inputs_embeds`. The full Qwen 0.5B model serves as the Activation Verbalizer, and all transformer weights are updated during training.

### Activation Reconstructor (AR)

The Activation Reconstructor converts a description back into an activation vector, enabling the round-trip that defines the autoencoder. The architecture consists of the **first 12 layers** of Qwen (frozen), followed by a single `Linear(896, 896, bias=False)` value head.

**Identity initialization** is applied to the value head: `nn.init.eye_(ar.value_head.weight)`. This initializes the value head as an identity mapping, so the Activation Reconstructor starts by returning the last hidden state unchanged. In practice, this drops the initial reconstruction error from 1.94 to 1.61, giving the optimizer a better starting point.

---

## Three-Stage Training — `pipeline/train.py`

The main script controls the entire training pipeline and runs the three stages sequentially. It is not just limited to running one optimization loop; furthermore, there are additional stages that need to be completed before reinforcement learning begins.

### Stage 1 — Reconstructor Supervised Fine-Tuning (100 steps, LR = 1e-4)

The value head is trained on (description, activation) pairs from the labeled dataset. The backbone is frozen throughout all training. The loss is MSE between the predicted and target activation vectors, both L2-normalized.

This stage teaches the Activation Reconstructor the mapping from natural language descriptions to activation vectors. Without it, the reinforcement learning signal is too noisy to learn from.

### Stage 2 — Verbalizer Supervised Fine-Tuning (100 steps, LR = 5e-6)

The Activation Verbalizer is trained to reproduce the reference descriptions from the labeled dataset, conditioned on the corresponding activation vectors via injection. The loss is masked cross-entropy over description tokens only.

This is the imitation phase: the Activation Verbalizer learns the style and structure of valid descriptions before reinforcement learning nudges it toward reconstruction-optimized descriptions.

### Stage 3 — Reinforcement Learning (300 steps, LR = 5e-6)

The reward is `cos_sim(AR(AV(activation)), activation)` — the round-trip fidelity. A **log reward transformation** is applied: `reward = log(1 + cos_sim)`, which compresses high rewards and amplifies the gradient signal when reconstruction quality is low, and it is consistent with the reward shaping described in the original paper.

The algorithm is **REINFORCE with an exponential-moving-average baseline** (α = 0.9). Each step: the Activation Verbalizer samples a batch of descriptions (no gradient), the Activation Reconstructor scores them (no gradient), the cosine similarity is computed, and the Activation Verbalizer is updated using the policy gradient `-(reward - baseline) × log_prob(description | activation)`. The Activation Reconstructor is simultaneously updated with MSE loss on the same batch.

---

## Training Summary

| Hyperparameter | Value |
|---------------|-------|
| AR SFT steps | 100 |
| AV SFT steps | 100 |
| RL steps | 300 |
| Batch size | 4 |
| LR (AV) | 5e-6 |
| LR (AR) | 1e-4 |
| RL algorithm | REINFORCE + EMA baseline |
| Reward | log(1 + cos_sim) |
| Checkpoints | `artifacts/checkpoints/` |
