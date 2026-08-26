"""Streamlit Community Cloud entry point (share.streamlit.io).

Downloads the trained model + calibration from the repo's GitHub Release at startup
(the weights are too big for git), then serves the interactive demo. Deploy: point
Streamlit Cloud at this repo with main file `streamlit_app.py`.
"""

from __future__ import annotations

import json
import os
import urllib.request

import numpy as np
import streamlit as st
from PIL import Image

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.serve import CorruptionInspector

RELEASE = "https://github.com/tanaymihani/gpu-corruptnet/releases/download/demo-model"
CACHE = "/tmp/corruptnet"

st.set_page_config(page_title="GPU-CorruptNet", layout="wide")
st.title("GPU-CorruptNet — GPU rendering-corruption detection")
st.caption(
    "Detects & classifies 10 classes of GPU rendering corruption with calibrated, conformal "
    "confidence. Source: github.com/tanaymihani/gpu-corruptnet"
)


@st.cache_resource
def load_inspector() -> CorruptionInspector:
    os.makedirs(CACHE, exist_ok=True)
    model_path = os.path.join(CACHE, "model.pt")
    cal_path = os.path.join(CACHE, "calibration.json")
    if not os.path.exists(model_path):
        urllib.request.urlretrieve(f"{RELEASE}/model.pt", model_path)
    if not os.path.exists(cal_path):
        urllib.request.urlretrieve(f"{RELEASE}/calibration.json", cal_path)
    cal = json.load(open(cal_path))
    temperature = float(cal.get("temperature", 1.0))
    tau = cal.get("splits", {}).get("seen_test", {}).get("conformal", {}).get("tau")
    return CorruptionInspector(model_path, temperature=temperature, conformal_tau=tau)


with st.spinner("Loading model…"):
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
    c2.metric("latency", f"{res['latency_ms']:.1f} ms")
    st.metric("corrupted confidence", f"{res['corrupted_prob'] * 100:.1f}%")
    st.markdown("**Predicted artifacts:** " + (", ".join(res["predicted"]) or "_clean_"))
    if "conformal_set" in res:
        st.markdown("**Conformal set (90%):** " + (", ".join(res["conformal_set"]) or "_∅_"))
    st.bar_chart(dict(sorted(res["probs"].items(), key=lambda kv: -kv[1])))
