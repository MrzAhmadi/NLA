# Usage and Interface

This section discusses how to use the interactive interface to inspect any sentence and check what the model encodes at each token position. Understanding these principles is essential before examining the interface of the proposed system.

First, the interface is launched. Then, a text is entered and extracted. Finally, each step of the analysis pipeline is explored.

---

## Hardware Requirements

The most characteristic property of the host machine is having sufficient GPU resources to run the experiment, since the procedure of any single forward pass runs on the GPU, and the host must allocate sufficient VRAM to it. In general, the heaviest part of the pipeline is the language model used to run verbalization and activation extraction.

In this project, all training and inference steps were executed on an **NVIDIA GeForce RTX 3070 Laptop GPU (8 GB VRAM)**. In conclusion, the environment should be prepared to run the pipeline with the necessary software and sufficient hardware resources; specifically, a CUDA-compatible GPU and the corresponding CUDA toolkit are required.

Running the pipeline on CPU is technically possible, but it is time-consuming and resource-intensive, and for this reason, it is strongly recommended to use a CUDA-capable GPU. More GPU resources make it possible to run larger batch sizes, and larger batch sizes shorten the overall training and inference duration significantly.

To verify that CUDA is available before launching:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

---

## Launching the Interface

The interactive interface is implemented using Streamlit. To run it, the following command should be executed from the project root:

```bash
python -m streamlit run app.py
```

The interface is then accessible at `http://localhost:8501`. It is not just limited to running the verbalizer; furthermore, there are more analysis steps that are available within the interface, such as the logit lens, the Activation Verbalizer, the Activation Reconstructor, and causal steering.

---

## Step 1 — Extract Token Activations

The main objective of the extraction step is to take an input text and run the subject model forward, capturing the hidden state at each token position at layer 16.

Enter any sentence in the text area and click **Extract**. The system tokenizes the text, runs the forward pass, and stores the activation vectors. By the end of the extraction process, the tokenized sequence is displayed, and each token is selectable for further analysis.

Use the **Token position** slider to select a specific token. The activation L2-norm is shown below the slider. It should be approximately `sqrt(896) ≈ 29.93`, confirming the normalization is applied correctly.

---

## Step 1.5 — Logit Lens: What the Model Is Thinking

Due to limitations in direct inspection of activation vectors, the logit lens approach is more accurate for understanding what the model predicts, but it requires a full forward pass through the model at all 24 layers. The logit lens projects the hidden state at every layer through the final layer norm and the unembedding matrix, and it is the most direct method to observe what the model computes.

This step displays two panels:

- **Final prediction** — the model's actual next-token distribution at this position, projected through the full 24-layer output. This is not an approximation; it is the model's actual output distribution.
- **Layer-by-layer sweep** — for each layer from 0 to 23, the top-3 predicted tokens are shown. By expanding this panel, it is discoverable at which layer the model's prediction stabilizes and forms its final answer.

---

## Step 2 — NLA Verbalizer (Approximate)

The Activation Verbalizer injects the activation vector into the model's embedding space at the position of the injection character and generates a natural language description autoregressively.

Click **Run Verbalizer** to generate a description from the selected token's activation. At this scale (0.5B model, self-labeled training data), the output is often generic or thematically approximate. Step 1.5 above gives the exact readout; this step shows the learned approximation that the autoencoder trains toward.

The **Max description tokens** slider in the sidebar controls the maximum length of the generated description.

---

## Step 3 — Reconstructor: Round-Trip Fidelity

The Activation Reconstructor converts the generated description back into an activation vector and measures how closely it matches the original.

Click **Run Reconstructor** to score the round-trip. Two metrics are displayed:

- **Cosine similarity** — 1.0 is a perfect round-trip, 0.0 is random, -1.0 is antipodal
- **MSE (normalized)** — 0 is perfect, 4 is the maximum; related by `MSE = 2 × (1 - cos_sim)`

---

## Step 4 — Causal Steering

In this step, there is a possibility to edit the NLA description, reconstruct a new activation from it, and inject the modified activation into the model at this layer and position to observe how the output prediction changes.

After running the Verbalizer in Step 2, a text area appears with the generated description pre-filled. To perform a causal steering experiment:

1. Edit the description — for example, replace a country name with another
2. Click **Steer**
3. The system reconstructs a new activation vector from the edited description using the Activation Reconstructor, injects it into the model at layer 16 via a forward hook, and shows the top-5 next-token predictions **before** and **after** the steering side by side

This demonstrates whether the Activation Reconstructor's learned mapping causally influences the model's behavior, which is consistent with the causal intervention validation described in the original paper.
