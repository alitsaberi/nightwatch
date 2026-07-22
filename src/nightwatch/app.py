"""Streamlit UI entry point.

Run with::

    streamlit run nightwatch.app
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Literal

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


def _escape_applescript(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', '\\"')


def _run_osascript(script: str) -> str | None:
    import subprocess

    completed = subprocess.run(
        ["osascript", "-e", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    selected = completed.stdout.strip()
    return selected or None


def _pick_directory_macos(*, title: str) -> str | None:
    prompt = _escape_applescript(title)
    return _run_osascript(f'POSIX path of (choose folder with prompt "{prompt}")')


def _pick_onnx_file_macos(*, title: str) -> str | None:
    prompt = _escape_applescript(title)
    # Do not use `of type {"onnx"}`: macOS often has no UTI for .onnx, so the
    # dialog filters out valid models. Enforce the extension in Python instead.
    return _run_osascript(f'POSIX path of (choose file with prompt "{prompt}")')


def _pick_path_via_tkinter_subprocess(
    *,
    kind: Literal["dir", "file"],
    title: str,
    initial: str | None = None,
) -> str | None:
    """Run Tk in a subprocess so the dialog owns the process main thread.

    Calling Tk from Streamlit's script thread crashes on macOS
    (``NSWindow should only be instantiated on the main thread``).
    """
    import subprocess
    import sys

    initial_repr = repr(initial)
    title_repr = repr(title)
    if kind == "dir":
        picker = (
            "selected = filedialog.askdirectory("
            f"title={title_repr}, initialdir={initial_repr} or None)"
        )
    else:
        # Keep "*.onnx" preferred; include *.* so macOS/Tk still lists files when
        # the extension filter is flaky, then validate in Python.
        picker = (
            "selected = filedialog.askopenfilename("
            f"title={title_repr}, initialdir={initial_repr} or None, "
            'filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")], '
            'defaultextension=".onnx")'
        )

    script = "\n".join(
        [
            "import tkinter as tk",
            "from tkinter import filedialog",
            "root = tk.Tk()",
            "root.withdraw()",
            "try:",
            '    root.attributes("-topmost", True)',
            "except tk.TclError:",
            "    pass",
            picker,
            "root.destroy()",
            "print(selected or '')",
        ]
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    selected = completed.stdout.strip()
    return selected or None


def _pick_directory(*, title: str = "Select recording folder") -> str | None:
    """Open a native folder dialog."""
    import sys

    if sys.platform == "darwin":
        return _pick_directory_macos(title=title)
    initial = st.session_state.get("recording_path") or None
    return _pick_path_via_tkinter_subprocess(kind="dir", title=title, initial=initial)


def _pick_onnx_file(*, title: str = "Select sleep-scoring model") -> str | None:
    """Open a native file dialog and accept only ``.onnx`` paths."""
    import sys

    st.session_state.pop("model_path_error", None)
    if sys.platform == "darwin":
        selected = _pick_onnx_file_macos(title=title)
    else:
        current = st.session_state.get("model_path") or ""
        initial_dir = str(Path(current).expanduser().parent) if current else None
        selected = _pick_path_via_tkinter_subprocess(
            kind="file",
            title=title,
            initial=initial_dir,
        )
    if selected is None:
        return None
    if not selected.lower().endswith(".onnx"):
        st.session_state["model_path_error"] = "Please select a .onnx model file."
        return None
    return selected


def _path_input_with_browse(
    *,
    label: str,
    state_key: str,
    browse_label: str,
    pick: Callable[[], str | None],
    help_text: str,
) -> str:
    """Text path field with a side Browse button; the input shrinks first."""

    def _on_browse() -> None:
        # Callback runs before widgets are instantiated on the rerun, so we can
        # safely assign the widget's session-state key here.
        selected = pick()
        if selected:
            st.session_state[state_key] = selected

    try:
        cols = st.sidebar.columns([1, 0.42], gap="small", vertical_alignment="bottom")
    except TypeError:
        cols = st.sidebar.columns([1, 0.42])
    with cols[0]:
        # Marker so CSS can target only these path rows (not other sidebar controls).
        st.markdown('<div class="nw-path-pair"></div>', unsafe_allow_html=True)
        value = st.text_input(label, key=state_key)
    with cols[1]:
        st.button(
            browse_label,
            key=f"browse_{state_key}",
            help=help_text,
            on_click=_on_browse,
            width='stretch',
        )
    return value


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
        st.dataframe(rows, width='stretch', hide_index=True)


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
            width='stretch',
            hide_index=True,
        )
    with right_col:
        st.markdown("**Right electrode**")
        st.dataframe(
            [{"Label": label, "Percent": _pct(pct)} for label, pct in sorted(artifacts["right"].items())],
            width='stretch',
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
            width='stretch',
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
        st.pyplot(fig, width='stretch')


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
st.markdown(
    """
    <style>
    /* Hide layout markers used to find path+browse rows. */
    [data-testid="stSidebar"] .nw-path-pair {
      display: none;
    }

    /* Path rows only: fixed Browse width; text input absorbs shrink. */
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair) {
      display: flex !important;
      flex-wrap: nowrap !important;
      gap: 0.4rem !important;
      width: 100% !important;
      max-width: 100% !important;
      overflow: hidden !important;
      align-items: flex-end !important;
      box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      > [data-testid="stColumn"]:first-child,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      > div:first-child {
      flex: 1 1 0% !important;
      min-width: 0 !important;
      width: auto !important;
      max-width: calc(100% - 5.75rem) !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      > [data-testid="stColumn"]:last-child,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      > div:last-child {
      flex: 0 0 5.5rem !important;
      width: 5.5rem !important;
      min-width: 5.5rem !important;
      max-width: 5.5rem !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      [data-testid="stTextInput"],
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      [data-testid="stTextInput"] > div,
    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair)
      [data-testid="stTextInput"] input {
      min-width: 0 !important;
      max-width: 100% !important;
      width: 100% !important;
      box-sizing: border-box !important;
    }

    [data-testid="stSidebar"] [data-testid="stHorizontalBlock"]:has(.nw-path-pair) button {
      white-space: nowrap !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("Nightwatch")
st.caption(f"v{__version__} — sleep recording QC powered by somnio")

st.sidebar.header("Settings")
st.session_state.setdefault("recording_path", "")
st.session_state.setdefault("model_path", "")
recording_path = _path_input_with_browse(
    label="Recording path",
    state_key="recording_path",
    browse_label="Browse",
    pick=_pick_directory,
    help_text="Choose a recording folder",
)
format_choice = st.sidebar.selectbox("Format", options=["zmax"])
model_path = _path_input_with_browse(
    label="Sleep model path (.onnx)",
    state_key="model_path",
    browse_label="Browse",
    pick=_pick_onnx_file,
    help_text="Choose an ONNX sleep-scoring model",
)
if st.session_state.get("model_path_error"):
    st.sidebar.error(st.session_state["model_path_error"])
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
