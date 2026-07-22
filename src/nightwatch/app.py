"""Streamlit UI entry point.

Run with::

    streamlit run nightwatch.app
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import streamlit as st

from nightwatch import __version__
from nightwatch.config import AnalysisConfig, DEFAULT_EYE_MOVEMENT_PATTERN
from nightwatch.metrics import compute_metrics
from nightwatch.pipeline import AnalysisResult, run_analysis
from nightwatch.plots import build_plots
from nightwatch.report import plot_display_order, plot_title, render

UsabilityModel = Literal["default", "lite", "binary", "lite_binary"]


def _pct(value: object, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}%"


def _fmt(value: object, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}"


def _build_config(
    recording_path: str,
    format_choice: str,
    model_path: str,
    edge_minutes: float,
    usability_model: UsabilityModel,
    eye_movement_pattern: str,
) -> AnalysisConfig:
    return AnalysisConfig(
        recording_path=Path(recording_path),
        format=format_choice,  # type: ignore[arg-type]
        model_path=Path(model_path),
        edge_minutes=edge_minutes,
        usability_model=usability_model,
        eye_movement_pattern=eye_movement_pattern,
    )


def _validate_config(config: AnalysisConfig) -> str | None:
    if not config.recording_path.exists():
        return f"Recording path does not exist: {config.recording_path}"
    if not config.model_path.is_file():
        return f"Sleep-scoring model not found: {config.model_path}"
    return None


def _render_recording_section(rec: dict[str, Any]) -> None:
    st.subheader("Recording")
    col1, col2, col3 = st.columns(3)
    col1.metric("Duration", rec["duration_hms"])
    col2.metric("Sample rate", f"{_fmt(rec['sample_rate_hz'])} Hz")
    col3.metric("Format", rec["format"])
    st.caption(f"Start: {rec['start']} · End: {rec['end']}")
    st.markdown(f"**Channels:** `{', '.join(rec['channels'])}`")


def _render_sleep_section(sleep: dict[str, Any]) -> None:
    st.subheader("Sleep")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("TRT", f"{_fmt(sleep['trt_minutes'])} min")
    col2.metric("TST", f"{_fmt(sleep['tst_minutes'])} min")
    col3.metric("Sleep efficiency", _pct(sleep["sleep_efficiency_pct"]))
    col4.metric("SOL", f"{_fmt(sleep['sol_minutes'])} min")
    col5.metric("WASO", f"{_fmt(sleep['waso_minutes'])} min")
    col6.metric("Unusable", f"{_fmt(sleep.get('unusable_minutes', 0.0))} min")

    stage_minutes: dict[str, float] = sleep.get("stage_minutes", {})
    if stage_minutes:
        stage_pct: dict[str, float] = sleep.get("stage_pct", {})
        rows = [
            {
                "Stage": stage,
                "Minutes": round(minutes, 1),
                "Percent": _pct(stage_pct.get(stage, 0.0)),
            }
            for stage, minutes in sorted(stage_minutes.items())
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)


def _render_artifacts_section(artifacts: dict[str, Any]) -> None:
    st.subheader("Artifacts (EEG usability)")
    col1, col2, col3 = st.columns(3)
    col1.metric("Usable epochs (left)", _pct(artifacts["usable_epoch_pct_left"]))
    col2.metric("Usable epochs (right)", _pct(artifacts["usable_epoch_pct_right"]))
    col3.metric(
        "Samples to keep",
        f"{artifacts['samples_to_keep']:,} / {artifacts['samples_total']:,}",
        delta=_pct(artifacts["samples_to_keep_pct"]),
    )

    left_col, right_col = st.columns(2)
    with left_col:
        st.markdown("**Left electrode**")
        st.dataframe(
            [{"Label": label, "Percent": _pct(pct)} for label, pct in sorted(artifacts["left"].items())],
            use_container_width=True,
            hide_index=True,
        )
    with right_col:
        st.markdown("**Right electrode**")
        st.dataframe(
            [{"Label": label, "Percent": _pct(pct)} for label, pct in sorted(artifacts["right"].items())],
            use_container_width=True,
            hide_index=True,
        )


def _render_edge_table(edge: dict[str, Any], title: str) -> None:
    if not edge.get("has_matches"):
        return

    st.markdown(f"**{title}**")
    col1, col2 = st.columns(2)
    col1.metric("Window duration", edge["duration_hms"])
    col2.metric("Matched sequences", str(edge["sequence_count"]))

    seq_hist: dict[str, int] = edge.get("sequence_label_histogram", {})
    if seq_hist:
        st.caption("Sequence labels")
        st.dataframe(
            [{"Label": label, "Count": count} for label, count in seq_hist.items()],
            use_container_width=True,
            hide_index=True,
        )


def _render_eye_movement_section(em: dict[str, Any]) -> None:
    st.subheader("Eye movements (edge windows)")
    st.caption(f"Edge window length: {_fmt(em['edge_minutes'])} min (start and end)")
    st.caption(f"Sequence pattern: `{em['pattern']}`")
    if not em.get("has_matches"):
        st.info("No matching eye-movement sequences in either edge window.")
        return
    _render_edge_table(em["start"], "First edge window")
    _render_edge_table(em["end"], "Last edge window")


def _render_metrics(metrics: dict[str, Any]) -> None:
    _render_recording_section(metrics["recording"])
    st.divider()
    _render_sleep_section(metrics["sleep"])
    st.divider()
    _render_artifacts_section(metrics["artifacts"])
    st.divider()
    _render_eye_movement_section(metrics["eye_movement"])


def _render_plots(plots: dict[str, plt.Figure]) -> None:
    st.subheader("Plots")
    for key in plot_display_order(plots):
        fig = plots[key]
        st.markdown(f"**{plot_title(key)}**")
        st.pyplot(fig, use_container_width=True)


def _run_and_store(config: AnalysisConfig) -> None:
    with st.spinner("Running analysis…"):
        result = run_analysis(config)
        metrics = compute_metrics(result)
        report_plots = build_plots(result)
        html = render(metrics, report_plots)

    st.session_state["analysis_result"] = result
    st.session_state["analysis_metrics"] = metrics
    st.session_state["analysis_html"] = html


st.set_page_config(page_title="Nightwatch", layout="wide")
st.title("Nightwatch")
st.caption(f"v{__version__} — sleep recording QC powered by somnio")

st.sidebar.header("Settings")
recording_path = st.sidebar.text_input("Recording path")
format_choice = st.sidebar.selectbox("Format", options=["zmax"])
model_path = st.sidebar.text_input("Sleep model path (.onnx)")
edge_minutes = st.sidebar.number_input("Edge minutes", min_value=1.0, value=30.0, step=1.0)
usability_model = st.sidebar.selectbox(
    "Usability model",
    options=["lite", "default", "binary", "lite_binary"],
)
eye_movement_pattern = st.sidebar.text_input(
    "Eye-movement sequence pattern",
    value=DEFAULT_EYE_MOVEMENT_PATTERN,
)

run_clicked = st.sidebar.button("Run analysis", type="primary")

if run_clicked:
    if not recording_path.strip():
        st.error("Enter a recording path.")
    elif not model_path.strip():
        st.error("Enter a sleep-scoring model path.")
    elif not eye_movement_pattern.strip():
        st.error("Enter an eye-movement sequence pattern.")
    else:
        config = _build_config(
            recording_path.strip(),
            format_choice,
            model_path.strip(),
            float(edge_minutes),
            usability_model,  # type: ignore[arg-type]
            eye_movement_pattern.strip(),
        )
        validation_error = _validate_config(config)
        if validation_error:
            st.error(validation_error)
        else:
            try:
                _run_and_store(config)
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

if "analysis_metrics" in st.session_state:
    result: AnalysisResult = st.session_state["analysis_result"]
    metrics = st.session_state["analysis_metrics"]
    html_report: str = st.session_state["analysis_html"]
    config = result.config

    st.markdown(f"**Recording:** `{config.recording_path}`")

    download_name = f"nightwatch_{config.recording_path.name}.html"
    st.download_button(
        label="Download HTML report",
        data=html_report,
        file_name=download_name,
        mime="text/html",
    )

    _render_metrics(metrics)
    st.divider()
    _render_plots(build_plots(result))
else:
    st.info("Configure settings in the sidebar and click **Run analysis** to view metrics and plots.")
