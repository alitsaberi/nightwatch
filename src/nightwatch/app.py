"""Streamlit UI entry point.

Run with::

    streamlit run nightwatch.app
"""

from __future__ import annotations

import streamlit as st

from nightwatch import __version__

st.set_page_config(page_title="Nightwatch", layout="wide")
st.title("Nightwatch")
st.caption(f"v{__version__} — sleep recording QC powered by somnio")

st.sidebar.header("Settings")
st.sidebar.text_input("Recording path", key="recording_path")
st.sidebar.selectbox("Format", options=["zmax"], key="format")
st.sidebar.text_input("Sleep model path (.onnx)", key="model_path")
st.sidebar.number_input("Edge minutes", min_value=1.0, value=30.0, key="edge_minutes")
st.sidebar.selectbox(
    "Usability model",
    options=["default", "lite", "binary", "lite_binary"],
    key="usability_model",
)

if st.sidebar.button("Run analysis"):
    st.info("Analysis pipeline is not implemented yet.")

st.markdown("Configure settings in the sidebar and run analysis to view metrics and plots.")
