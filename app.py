from pathlib import Path

import numpy as np
import streamlit as st
import torch

from nla.activation_reconstructor import ActivationReconstructor
from nla.activation_verbalizer import ActivationVerbalizer
from nla.checkpoint_utils import load_chunked
from nla.subject_model import GeneratedToken, SubjectModel

AV_CHECKPOINT = Path("artifacts/checkpoints/av_model")   # directory of part_XXXX.pt chunks
AR_CHECKPOINT = Path("artifacts/checkpoints/ar_value_head.pt")

MODEL_NAME  = "Qwen/Qwen2.5-0.5B-Instruct"
LAYER_INDEX = 16
AR_LAYERS   = 12
AV_MAX_TOKENS = 80

st.set_page_config(page_title="NLA — Read Model's Mind", layout="wide")

with st.sidebar:
    st.title("Settings")

    if torch.cuda.is_available():
        st.success(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        st.info("No GPU — running on CPU (slow)")

    st.caption(f"Model: `{MODEL_NAME}`  |  Layer: `{LAYER_INDEX}/24`")

    st.divider()

    if AV_CHECKPOINT.exists() and AR_CHECKPOINT.exists():
        st.success("Trained checkpoints found")
    else:
        st.warning("No checkpoints — run pipeline/train.py first")

    av_max_tokens = st.slider("Max description tokens", 20, 200, AV_MAX_TOKENS)


@st.cache_resource(show_spinner="Loading subject model")
def get_subject_model(name: str) -> SubjectModel:
    return SubjectModel(name)


@st.cache_resource(show_spinner="Loading verbalizer")
def get_av(name: str) -> ActivationVerbalizer:
    av = ActivationVerbalizer(name)
    if AV_CHECKPOINT.exists():
        av.model.load_state_dict(load_chunked(AV_CHECKPOINT, map_location=av.device))
    return av


@st.cache_resource(show_spinner="Loading reconstructor")
def get_ar(name: str, layers: int) -> ActivationReconstructor:
    ar = ActivationReconstructor(name, num_critic_layers=layers)
    if AR_CHECKPOINT.exists():
        ar.value_head.load_state_dict(torch.load(AR_CHECKPOINT, map_location=ar.device))
    return ar


st.title("NLA — Read Model's Mind")
st.markdown(
    "The **subject model** extracts token activations. "
    "The **verbalizer** injects the activation as an embedding and generates a description. "
    "The **reconstructor** converts that description back into an activation vector and scores the round-trip."
)

text = st.text_area("Text to analyze", "The capital of Iran is Tehran.", height=80)

if text != st.session_state.get("last_text"):
    for key in ["tokens", "av_description", "av_activation", "last_text"]:
        st.session_state.pop(key, None)

if st.button("Extract", type="primary"):
    subject = get_subject_model(MODEL_NAME)
    if LAYER_INDEX >= subject.num_layers:
        st.error(
            f"{MODEL_NAME} has {subject.num_layers} layers "
            f"(0 to {subject.num_layers - 1}). Lower the layer index."
        )
        st.stop()
    with st.spinner("Extracting activations"):
        tokens = subject.extract(text, int(LAYER_INDEX))
    st.session_state["tokens"]    = tokens
    st.session_state["last_text"] = text

if "tokens" not in st.session_state:
    st.stop()

tokens: list[GeneratedToken] = st.session_state["tokens"]
st.info("".join(t.text for t in tokens))

st.markdown("---")
st.markdown("### Step 1 — Select a token to inspect")

options = [f"{t.position}: {t.text!r}" for t in tokens]
choice  = st.select_slider("Token position", options=options)
idx     = int(choice.split(":")[0])
tok     = tokens[idx]

norm_val = tok.activation.float().norm().item()
st.markdown(
    f"**Token** `{tok.text!r}` at position `{tok.position}` — "
    f"activation L2-norm: `{norm_val:.2f}`  "
    f"(target: `{norm_val:.2f}` = sqrt({tok.activation.shape[0]}) = "
    f"`{tok.activation.shape[0] ** 0.5:.2f}`)"
)

st.markdown("---")
st.markdown("### Step 1.5 — What the model is thinking")
st.caption(
    "Exact next-token predictions from the full forward pass. "
    "This is not an approximation — it is the model's actual output distribution at this position."
)

subject = get_subject_model(MODEL_NAME)
context  = "".join(t.text for t in tokens[: tok.position + 1]).strip()
lens     = subject.logit_lens(text, tok.position, k=8)
top_prob = lens[0]["prob"]

st.markdown(f"After reading: **`{context}`**")
st.markdown(f"The model predicts the next token will be:")

for entry in lens[:5]:
    bar = int(entry["prob"] / top_prob * 28)
    st.markdown(
        f"&nbsp;&nbsp;`{entry['token']!r}` &nbsp; {'█' * bar}{'░' * (28 - bar)} &nbsp; "
        f"**{entry['prob']*100:.1f}%**",
        unsafe_allow_html=True,
    )

with st.expander("How this prediction forms layer by layer"):
    with st.spinner("Running layer sweep"):
        sweep = subject.logit_lens_sweep(text, tok.position, k=3)

    top_final = lens[0]["token"]
    stabilizes_at = next(
        (s["layer"] for s in sweep if s["top"][0]["token"] == top_final), None
    )
    if stabilizes_at is not None:
        st.caption(
            f"Top prediction `{top_final!r}` first appears at layer {stabilizes_at} "
            f"(of {len(sweep)}) and stays there."
        )

    rows = []
    for s in sweep:
        top3 = "  /  ".join(f"{e['token']!r} {e['prob']*100:.0f}%" for e in s["top"])
        rows.append({"layer": s["layer"], "top-3 predictions": top3})
    st.dataframe(rows, use_container_width=True, hide_index=True, height=320)

st.markdown("---")
st.markdown("### Step 2 — NLA Verbalizer (research component, approximate)")
st.caption(
    "The NLA verbalizer injects the activation vector into the model's embedding space and asks it "
    "to describe what it encodes. At this scale (0.5B, self-labeled training data) the output is "
    "often generic or off-topic. Step 1.5 above gives the exact readout; this step shows the "
    "learned approximation from training."
)

if st.button("Run Verbalizer"):
    av = get_av(MODEL_NAME)
    if av.d_model != tok.activation.shape[0]:
        st.error(
            f"Dimension mismatch: subject has d_model={tok.activation.shape[0]}, "
            f"verbalizer has d_model={av.d_model}. Use a model with the same hidden size."
        )
        st.stop()
    with st.spinner("Generating description from activation"):
        description = av.verbalize(tok.activation, max_new_tokens=av_max_tokens)
    st.session_state["av_description"] = description
    st.session_state["av_activation"]  = tok.activation

if "av_description" in st.session_state:
    st.success(st.session_state["av_description"] or "(empty — model generated nothing)")

if "av_description" not in st.session_state:
    st.stop()

st.markdown("---")
st.markdown("### Step 3 — Reconstruct: recover the activation from the description")
st.caption(
    "Without training, the reconstructor value head is random (cos_sim near 0). "
    "After training on activation / description round-trips it improves."
)

if st.button("Run Reconstructor"):
    ar = get_ar(MODEL_NAME, int(AR_LAYERS))
    if ar.d_model != st.session_state["av_activation"].shape[0]:
        st.error(
            f"Dimension mismatch: activation has d={st.session_state['av_activation'].shape[0]}, "
            f"reconstructor has d_model={ar.d_model}."
        )
        st.stop()
    with st.spinner("Reconstructing activation from description"):
        metrics = ar.score(
            st.session_state["av_description"],
            st.session_state["av_activation"],
        )

    col1, col2 = st.columns(2)
    col1.metric(
        "Cosine similarity",
        f"{metrics['cosine_similarity']:.3f}",
        help="1.0 = perfect round-trip  |  0.0 = random  |  -1.0 = antipodal",
    )
    col2.metric(
        "MSE (normalized)",
        f"{metrics['mse']:.3f}",
        help="MSE = 2 * (1 - cos_sim)  |  range [0, 4]  |  0 = perfect",
    )

    with st.expander("Raw activation stats (top-16 dimensions)"):
        act     = st.session_state["av_activation"].float().cpu().numpy()
        top_idx = np.argsort(np.abs(act))[-16:][::-1]
        st.dataframe(
            [{"dim": int(i), "value": float(act[i])} for i in top_idx],
            use_container_width=True,
        )

st.markdown("---")
st.markdown("### Step 4 — Causal Steering")
st.caption(
    "Edit the NLA description, reconstruct a new activation from it, "
    "then inject it into the model at this layer and position to see how the output changes."
)

if "av_description" not in st.session_state:
    st.info("Run the verbalizer first (Step 2).")
    st.stop()

edited = st.text_area(
    "Edit the description to steer the model",
    value=st.session_state.get("av_description", ""),
    height=100,
    key="steer_description",
)

if st.button("Steer"):
    ar      = get_ar(MODEL_NAME, int(AR_LAYERS))
    subject = get_subject_model(MODEL_NAME)

    with st.spinner("Reconstructing steered activation"):
        steered_activation = ar.reconstruct(edited)

    col_before, col_after = st.columns(2)

    with col_before:
        st.markdown("**Before steering** (original model output)")
        original_preds = subject.logit_lens(text, tok.position, k=5)
        for e in original_preds:
            st.markdown(f"`{e['token']!r}` — {e['prob']*100:.1f}%")

    with col_after:
        st.markdown("**After steering** (with edited description injected)")
        steered_preds = subject.steer(text, LAYER_INDEX, tok.position, steered_activation, k=5)
        for e in steered_preds:
            st.markdown(f"`{e['token']!r}` — {e['prob']*100:.1f}%")
