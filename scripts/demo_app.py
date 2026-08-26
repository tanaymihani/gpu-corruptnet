"""GPU-CorruptNet interactive demo (M10).

    streamlit run scripts/demo_app.py

Pick or upload a frame, optionally inject a corruption, and see the detected artifact
types, calibrated confidence, conformal set, OOD score, and inference latency. Works
without a model (generator + drift only); drop a trained checkpoint in runs/ for full
detection.
"""

from __future__ import annotations

import glob
import json

import numpy as np
import streamlit as st
from PIL import Image

from gpu_corruptnet.corruptions import apply, available
from gpu_corruptnet.data import make_demo_frame
from gpu_corruptnet.serve import CorruptionInspector

st.set_page_config(page_title="GPU-CorruptNet", layout="wide")
st.title("GPU-CorruptNet — corruption detection demo")


def _latest(pattern: str) -> str | None:
    hits = sorted(glob.glob(pattern))
    return hits[-1] if hits else None


@st.cache_resource
def get_inspector(model_path: str | None, temperature: float, tau: float | None):
    return CorruptionInspector(model_path or None, temperature=temperature, conformal_tau=tau)


# --- Sidebar: model + calibration -------------------------------------------------
st.sidebar.header("Model")
default_model = _latest("runs/model_*.pt") or ""
model_path = st.sidebar.text_input("Checkpoint path (runs/model_*.pt)", value=default_model)

temperature, tau = 1.0, None
cal_json = _latest("runs/calibration_*.json")
if cal_json:
    cal = json.load(open(cal_json))
    temperature = float(cal.get("temperature", 1.0))
    tau = cal.get("splits", {}).get("seen_test", {}).get("conformal", {}).get("tau")
    st.sidebar.caption(f"Calibration: T={temperature:.2f}, τ={tau}")

inspector = get_inspector(model_path, temperature, tau)
_status = "Model loaded" if inspector.net is not None else "No model — generator + drift only"
st.sidebar.success(_status)

# --- Input frame ------------------------------------------------------------------
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

# --- Layout -----------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Frame")
    st.image(shown, use_container_width=True)

with right:
    st.subheader("Inspector")
    res = inspector.predict(shown)

    c1, c2 = st.columns(2)
    ood_label = "out-of-distribution" if res["ood_flag"] else "in-dist"
    c1.metric("OOD score", f"{res['ood_score']:.2f}", ood_label)
    if res["has_model"]:
        c2.metric("latency", f"{res['latency_ms']:.1f} ms")
        st.metric("corrupted confidence", f"{res['corrupted_prob'] * 100:.1f}%")

        st.markdown("**Predicted artifacts:** " + (", ".join(res["predicted"]) or "_clean_"))
        if "conformal_set" in res:
            st.markdown("**Conformal set (90%):** " + (", ".join(res["conformal_set"]) or "_∅_"))

        probs = dict(sorted(res["probs"].items(), key=lambda kv: -kv[1]))
        st.bar_chart(probs)
    else:
        st.info("Add a checkpoint path in the sidebar to enable detection + calibrated confidence.")
