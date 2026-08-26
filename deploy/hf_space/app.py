"""Hugging Face Space entry point for the GPU-CorruptNet demo.

Loads model.pt + calibration.json from this folder if present (drop them in before
pushing the Space). Falls back to generator + drift mode if no model is provided.
"""

from __future__ import annotations

import json
import os

import numpy as np
import streamlit as st
from PIL import Image

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.serve import CorruptionInspector

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "model.pt")
CAL_PATH = os.path.join(HERE, "calibration.json")

st.set_page_config(page_title="GPU-CorruptNet", layout="wide")
st.title("GPU-CorruptNet — GPU rendering-corruption detection")
st.caption(
    "Detects & classifies 10 classes of GPU rendering corruption (shader glitches, screen "
    "tearing, texture/block artifacts, discoloration, stuck-memory patterns) with calibrated, "
    "conformal confidence."
)


@st.cache_resource
def load_inspector() -> CorruptionInspector:
    temperature, tau = 1.0, None
    if os.path.exists(CAL_PATH):
        cal = json.load(open(CAL_PATH))
        temperature = float(cal.get("temperature", 1.0))
        tau = cal.get("splits", {}).get("seen_test", {}).get("conformal", {}).get("tau")
    model = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    return CorruptionInspector(model, temperature=temperature, conformal_tau=tau)


inspector = load_inspector()

st.sidebar.header("Input frame")
source = st.sidebar.radio("Source", ["Procedural frame", "Upload"])
if source == "Procedural frame":
    seed = st.sidebar.slider("seed", 0, 500, 7)
    frame = make_demo_frame(rng=seed)
else:
    up = st.sidebar.file_uploader("Image", type=["png", "jpg", "jpeg"])
    frame = np.asarray(Image.open(up).convert("RGB")) if up else make_demo_frame(rng=7)

st.sidebar.header("Inject corruption")
name = st.sidebar.selectbox("Type", ["(none)"] + available())
severity = st.sidebar.slider("severity", 1, 5, 4)
shown = frame if name == "(none)" else apply(name, frame, severity=severity, rng=0)

left, right = st.columns(2)
with left:
    st.subheader("Frame")
    st.image(shown, use_container_width=True)

with right:
    st.subheader("Inspector")
    res = inspector.predict(shown)
    c1, c2 = st.columns(2)
    ood_label = "out-of-distribution" if res["ood_flag"] else "in-distribution"
    c1.metric("OOD score", f"{res['ood_score']:.2f}", ood_label)
    if res["has_model"]:
        c2.metric("latency", f"{res['latency_ms']:.1f} ms")
        st.metric("corrupted confidence", f"{res['corrupted_prob'] * 100:.1f}%")
        st.markdown("**Predicted artifacts:** " + (", ".join(res["predicted"]) or "_clean_"))
        if "conformal_set" in res:
            st.markdown("**Conformal set (90%):** " + (", ".join(res["conformal_set"]) or "_∅_"))
        st.bar_chart(dict(sorted(res["probs"].items(), key=lambda kv: -kv[1])))
    else:
        st.info("No model bundled — generator + drift only. Add model.pt to enable detection.")
