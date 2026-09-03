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
from nightwatch.config import (
    DEFAULT_EYE_MOVEMENT_PATTERN,
    ZMAX_DEFAULT_EEG_LEFT,
    ZMAX_DEFAULT_EEG_RIGHT,
    ZMAX_DEFAULT_MOVEMENT,
    AnalysisConfig,
    apply_zmax_view_defaults,
)
from nightwatch.load import LoadedRecording, load_recording
from nightwatch.metrics import compute_metrics
from nightwatch.pipeline import AnalysisResult, run_analysis
from nightwatch.plots import build_plots
from nightwatch.report import plot_display_order, plot_title, render

UsabilityModel = Literal["lite", "lite_binary"]
FormatChoice = Literal["zmax", "edf"]
SKIP_OPTION = "(skip)"


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


def _pick_file_macos(*, title: str) -> str | None:
    prompt = _escape_applescript(title)
    return _run_osascript(f'POSIX path of (choose file with prompt "{prompt}")')


def _pick_path_via_tkinter_subprocess(
    *,
    kind: Literal["dir", "file"],
    title: str,
    initial: str | None = None,
    filetypes: list[tuple[str, str]] | None = None,
) -> str | None:
    """Run Tk in a subprocess so the dialog owns the process main thread."""
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
        types = filetypes or [("All files", "*.*")]
        types_repr = repr(types)
        picker = (
            "selected = filedialog.askopenfilename("
            f"title={title_repr}, initialdir={initial_repr} or None, "
            f"filetypes={types_repr})"
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
    import sys

    if sys.platform == "darwin":
        return _pick_directory_macos(title=title)
    initial = st.session_state.get("recording_path") or None
    return _pick_path_via_tkinter_subprocess(kind="dir", title=title, initial=initial)


def _pick_edf_file(*, title: str = "Select EDF file") -> str | None:
    import sys

    st.session_state.pop("recording_path_error", None)
    if sys.platform == "darwin":
        selected = _pick_file_macos(title=title)
    else:
        current = st.session_state.get("recording_path") or ""
        initial_dir = str(Path(current).expanduser().parent) if current else None
        selected = _pick_path_via_tkinter_subprocess(
            kind="file",
            title=title,
            initial=initial_dir,
            filetypes=[("EDF", "*.edf"), ("All files", "*.*")],
        )
    if selected is None:
        return None
    if not selected.lower().endswith(".edf"):
        st.session_state["recording_path_error"] = "Please select a .edf file."
        return None
    return selected


def _pick_onnx_file(*, title: str = "Select sleep-scoring model") -> str | None:
    import sys

    st.session_state.pop("model_path_error", None)
    if sys.platform == "darwin":
        selected = _pick_file_macos(title=title)
    else:
        current = st.session_state.get("model_path") or ""
        initial_dir = str(Path(current).expanduser().parent) if current else None
        selected = _pick_path_via_tkinter_subprocess(
            kind="file",
            title=title,
            initial=initial_dir,
            filetypes=[("ONNX model", "*.onnx"), ("All files", "*.*")],
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
        selected = pick()
        if selected:
            st.session_state[state_key] = selected

    try:
        cols = st.sidebar.columns([1, 0.42], gap="small", vertical_alignment="bottom")
    except TypeError:
        cols = st.sidebar.columns([1, 0.42])
    with cols[0]:
        st.markdown('<div class="nw-path-pair"></div>', unsafe_allow_html=True)
        value = st.text_input(label, key=state_key)
    with cols[1]:
        st.button(
            browse_label,
            key=f"browse_{state_key}",
            help=help_text,
            on_click=_on_browse,
            width="stretch",
        )
    return value


def _optional_select(label: str, options: list[str], *, key: str, default: str | None) -> str | None:
    choices = [SKIP_OPTION, *options]
    if key not in st.session_state:
        if default is not None and default in options:
            st.session_state[key] = default
        else:
            st.session_state[key] = SKIP_OPTION
    selected = st.sidebar.selectbox(label, options=choices, key=key)
    return None if selected == SKIP_OPTION else selected


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
        st.dataframe(rows, width="stretch", hide_index=True)


def _render_artifacts_section(artifacts: dict[str, Any]) -> None:
    st.subheader("Artifacts (EEG usability)")
    col1, col2 = st.columns(2)
    col1.metric(
        "Samples to keep",
        f"{artifacts['samples_to_keep']:,} / {artifacts['samples_total']:,}",
        delta=_pct(artifacts["samples_to_keep_pct"]),
    )
    col2.metric("Epoch length", f"{_fmt(artifacts['epoch_length_seconds'])} s")

    usable_pct: dict[str, float] = artifacts.get("usable_epoch_pct", {})
    channels: dict[str, dict[str, float]] = artifacts.get("channels", {})
    for channel, labels in channels.items():
        st.markdown(f"**{channel}** — usable {_pct(usable_pct.get(channel, 0.0))}")
        st.dataframe(
            [{"Label": label, "Percent": _pct(pct)} for label, pct in sorted(labels.items())],
            width="stretch",
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
            width="stretch",
            hide_index=True,
        )


def _render_eye_movement_section(em: dict[str, Any]) -> None:
    st.subheader("Eye movements (edge windows)")
    st.caption(f"Edge window length: {_fmt(em['edge_minutes'])} min (start and end)")
    st.caption(f"Channels: `{em.get('left')}` / `{em.get('right')}`")
    st.caption(f"Sequence pattern: `{em['pattern']}`")
    if not em.get("has_matches"):
        st.info("No matching eye-movement sequences in either edge window.")
        return
    _render_edge_table(em["start"], "First edge window")
    _render_edge_table(em["end"], "Last edge window")


def _render_metrics(metrics: dict[str, Any]) -> None:
    _render_recording_section(metrics["recording"])
    if "sleep" in metrics:
        st.divider()
        _render_sleep_section(metrics["sleep"])
    if "artifacts" in metrics:
        st.divider()
        _render_artifacts_section(metrics["artifacts"])
    if "eye_movement" in metrics:
        st.divider()
        _render_eye_movement_section(metrics["eye_movement"])


def _render_plots(plots: dict[str, plt.Figure]) -> None:
    if not plots:
        return
    st.subheader("Plots")
    for key in plot_display_order(plots):
        fig = plots[key]
        st.markdown(f"**{plot_title(key)}**")
        st.pyplot(fig, width="stretch")


def _recording_metrics_from_loaded(
    loaded: LoadedRecording,
    *,
    path: Path,
    format_choice: str,
) -> dict[str, Any]:
    """Build a recording-only metrics dict without running analysis views."""
    from nightwatch.config import AnalysisConfig
    from nightwatch.pipeline import AnalysisResult

    config = AnalysisConfig(recording_path=path, format=format_choice)  # type: ignore[arg-type]
    result = AnalysisResult(
        config=config,
        recording=loaded.timeseries,
        raw_channel_names=loaded.raw_channel_names,
        sleep_channels=(),
        hypnodensity=None,
        hypnogram=None,
        usability_scores=None,
        usability_samples_to_keep=None,
        usability_epoch_length=None,
        edge_start=None,
        edge_end=None,
    )
    return compute_metrics(result)


def _seed_view_defaults(loaded: LoadedRecording, format_choice: str) -> None:
    """Pre-fill ZMax view widgets; leave EDF empty."""
    names = list(loaded.timeseries.channel_names)
    if format_choice != "zmax":
        st.session_state["raw_channels"] = []
        st.session_state["spectrogram_channels"] = []
        st.session_state["sleep_channels"] = []
        st.session_state["artifact_eeg_channels"] = []
        st.session_state["movement_channel"] = SKIP_OPTION
        st.session_state["eye_left"] = SKIP_OPTION
        st.session_state["eye_right"] = SKIP_OPTION
        return

    seed = apply_zmax_view_defaults(
        AnalysisConfig(recording_path=Path("."), format="zmax"),
        names,
    )
    st.session_state["raw_channels"] = list(seed.raw_channels)
    st.session_state["spectrogram_channels"] = list(seed.spectrogram_channels)
    st.session_state["sleep_channels"] = list(seed.sleep_channels)
    st.session_state["artifact_eeg_channels"] = list(seed.artifact_eeg_channels)
    st.session_state["movement_channel"] = seed.movement_channel or SKIP_OPTION
    st.session_state["eye_left"] = seed.eye_left or SKIP_OPTION
    st.session_state["eye_right"] = seed.eye_right or SKIP_OPTION


st.set_page_config(page_title="Nightwatch", layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] .nw-path-pair {
      display: none;
    }

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

st.sidebar.header("Recording")
st.session_state.setdefault("recording_path", "")
st.session_state.setdefault("model_path", "")

format_choice: FormatChoice = st.sidebar.selectbox(
    "Format",
    options=["zmax", "edf"],
)

if format_choice == "zmax":
    recording_path = _path_input_with_browse(
        label="Recording path",
        state_key="recording_path",
        browse_label="Browse",
        pick=_pick_directory,
        help_text="Choose a ZMax recording folder",
    )
else:
    recording_path = _path_input_with_browse(
        label="EDF file path",
        state_key="recording_path",
        browse_label="Browse",
        pick=_pick_edf_file,
        help_text="Choose an EDF file",
    )
    if st.session_state.get("recording_path_error"):
        st.sidebar.error(st.session_state["recording_path_error"])

load_clicked = st.sidebar.button("Load recording", type="secondary")

if load_clicked:
    if not recording_path.strip():
        st.error("Enter a recording path.")
    else:
        path = Path(recording_path.strip())
        load_config = AnalysisConfig(recording_path=path, format=format_choice)
        try:
            loaded = load_recording(load_config)
        except (FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError) as exc:
            st.error(str(exc))
        else:
            st.session_state["loaded_recording"] = loaded
            st.session_state["loaded_path"] = str(path)
            st.session_state["loaded_format"] = format_choice
            st.session_state.pop("analysis_result", None)
            st.session_state.pop("analysis_metrics", None)
            st.session_state.pop("analysis_html", None)
            _seed_view_defaults(loaded, format_choice)

loaded: LoadedRecording | None = st.session_state.get("loaded_recording")
channel_names: list[str] = list(loaded.timeseries.channel_names) if loaded else []

if loaded is not None:
    st.sidebar.header("Views")
    st.sidebar.caption("Leave empty / skip to omit a view.")

    raw_channels = st.sidebar.multiselect(
        "Raw traces",
        options=channel_names,
        key="raw_channels",
    )
    spectrogram_channels = st.sidebar.multiselect(
        "Spectrogram",
        options=channel_names,
        key="spectrogram_channels",
    )
    sleep_channels = st.sidebar.multiselect(
        "Sleep scoring",
        options=channel_names,
        key="sleep_channels",
    )
    if sleep_channels:
        model_path = _path_input_with_browse(
            label="Sleep model path (.onnx)",
            state_key="model_path",
            browse_label="Browse",
            pick=_pick_onnx_file,
            help_text="Choose an ONNX sleep-scoring model",
        )
        if st.session_state.get("model_path_error"):
            st.sidebar.error(st.session_state["model_path_error"])
    else:
        model_path = st.session_state.get("model_path", "")

    artifact_eeg_channels = st.sidebar.multiselect(
        "Artifact EEG",
        options=channel_names,
        key="artifact_eeg_channels",
    )
    movement_channel = _optional_select(
        "Movement channel",
        channel_names,
        key="movement_channel",
        default=ZMAX_DEFAULT_MOVEMENT if ZMAX_DEFAULT_MOVEMENT in channel_names else None,
    )
    eye_left = _optional_select(
        "Eye left",
        channel_names,
        key="eye_left",
        default=ZMAX_DEFAULT_EEG_LEFT if ZMAX_DEFAULT_EEG_LEFT in channel_names else None,
    )
    eye_right = _optional_select(
        "Eye right",
        channel_names,
        key="eye_right",
        default=ZMAX_DEFAULT_EEG_RIGHT if ZMAX_DEFAULT_EEG_RIGHT in channel_names else None,
    )

    edge_minutes = st.sidebar.number_input("Edge minutes", min_value=1.0, value=30.0, step=1.0)
    usability_model = st.sidebar.selectbox(
        "Usability model",
        options=["lite", "lite_binary"],
    )
    eye_movement_pattern = st.sidebar.text_input(
        "Eye-movement sequence pattern",
        value=DEFAULT_EYE_MOVEMENT_PATTERN,
    )

    run_clicked = st.sidebar.button("Run analysis", type="primary")

    if run_clicked:
        if sleep_channels and not str(model_path).strip():
            st.error("Enter a sleep-scoring model path when sleep channels are selected.")
        elif not eye_movement_pattern.strip():
            st.error("Enter an eye-movement sequence pattern.")
        else:
            config = AnalysisConfig(
                recording_path=Path(st.session_state["loaded_path"]),
                format=st.session_state["loaded_format"],
                model_path=Path(str(model_path).strip()) if sleep_channels else None,
                edge_minutes=float(edge_minutes),
                usability_model=usability_model,  # type: ignore[arg-type]
                eye_movement_pattern=eye_movement_pattern.strip(),
                raw_channels=list(raw_channels),
                spectrogram_channels=list(spectrogram_channels),
                sleep_channels=list(sleep_channels),
                artifact_eeg_channels=list(artifact_eeg_channels),
                movement_channel=movement_channel,
                eye_left=eye_left,
                eye_right=eye_right,
            )
            try:
                with st.spinner("Running analysis…"):
                    result = run_analysis(config)
                    metrics = compute_metrics(result)
                    report_plots = build_plots(result)
                    html = render(metrics, report_plots)
                st.session_state["analysis_result"] = result
                st.session_state["analysis_metrics"] = metrics
                st.session_state["analysis_html"] = html
            except (FileNotFoundError, ValueError) as exc:
                st.error(str(exc))

# Main pane
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
elif loaded is not None:
    path = Path(st.session_state["loaded_path"])
    format_loaded = st.session_state["loaded_format"]
    st.markdown(f"**Recording:** `{path}`")
    rec_metrics = _recording_metrics_from_loaded(loaded, path=path, format_choice=format_loaded)
    _render_recording_section(rec_metrics["recording"])
    st.info("Select views in the sidebar and click **Run analysis**.")
else:
    st.info("Choose a format and recording path, then click **Load recording**.")
