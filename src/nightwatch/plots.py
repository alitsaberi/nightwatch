"""Matplotlib figures for reports and Streamlit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.signal import spectrogram

from nightwatch.metrics import USABILITY_LABELS_BINARY, USABILITY_LABELS_MULTICLASS
from nightwatch.pipeline import USABILITY_EPOCH_NS, available_eeg_channels

if TYPE_CHECKING:
    from somnio.data.annotations import Event

    from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult

DEFAULT_STAGE_ORDER = ("W", "N1", "N2", "N3", "REM", "R", "Unusable")
NS_PER_HOUR = 3.6e12
MAX_SIGNAL_PLOT_POINTS = 60_000
ARTIFACT_SPAN_COLOR = "#424242"


def _figure(**kwargs: object) -> Figure:
    fig, _ = plt.subplots(**kwargs)  # type: ignore[arg-type]
    return fig


def _time_axis_hours(timestamps_ns: np.ndarray, t0_ns: float | None = None) -> np.ndarray:
    """Convert absolute nanosecond timestamps to hours relative to ``t0_ns``."""
    if timestamps_ns.size == 0:
        return np.array([], dtype=np.float64)
    t0 = float(timestamps_ns[0] if t0_ns is None else t0_ns)
    return (timestamps_ns.astype(np.float64) - t0) / NS_PER_HOUR


def _recording_time_span(recording: object) -> tuple[int, float | None]:
    """Return ``(t0_ns, duration_hours)`` for aligning full-recording plots."""
    if getattr(recording, "n_samples", 0) == 0:
        return 0, None

    t0_ns = int(recording.timestamps[0])
    sample_rate = getattr(recording, "sample_rate", None)
    if sample_rate is not None and float(sample_rate) > 0:
        duration_h = float(recording.n_samples) / float(sample_rate) / 3600.0
    else:
        duration_h = (float(recording.timestamps[-1]) - float(t0_ns)) / NS_PER_HOUR
    return t0_ns, duration_h


def _apply_shared_xlim(ax: plt.Axes, duration_h: float | None) -> None:
    if duration_h is not None and duration_h > 0:
        ax.set_xlim(0.0, duration_h)


def _downsample_series(
    times_h: np.ndarray,
    values: np.ndarray,
    *,
    max_points: int = MAX_SIGNAL_PLOT_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    if times_h.size <= max_points:
        return times_h, values
    step = int(np.ceil(times_h.size / max_points))
    return times_h[::step], values[::step]


def _stage_y_map(labels: tuple[str, ...] | list[str]) -> dict[str, int]:
    ordered = [stage for stage in DEFAULT_STAGE_ORDER if stage in labels]
    ordered.extend(stage for stage in labels if stage not in ordered)
    return {stage: idx for idx, stage in enumerate(ordered)}


def _usability_label_map(model_key: str) -> dict[int, str]:
    if model_key in {"binary", "lite_binary"}:
        return USABILITY_LABELS_BINARY
    return USABILITY_LABELS_MULTICLASS


def _usability_channel_index(usability_scores: object, eeg_channel: str, config: object) -> int:
    if eeg_channel == getattr(config, "eeg_left", None):
        return 0
    if eeg_channel == getattr(config, "eeg_right", None):
        return 1
    names = list(getattr(usability_scores, "channel_names", ()))
    for index, name in enumerate(names):
        if eeg_channel.replace("EEG_", "").lower() in str(name).lower():
            return index
    raise ValueError(f"No usability channel mapping for EEG channel {eeg_channel!r}")


def _artifact_any_channel_mask(usability_scores: object) -> np.ndarray:
    values = np.asarray(usability_scores.values)
    if values.size == 0:
        return np.array([], dtype=bool)
    return np.any(values != 0, axis=1)


def _overlay_usability_spans(
    ax: plt.Axes,
    usability_scores: object,
    *,
    t0_ns: float,
    mask: np.ndarray,
    color: str = ARTIFACT_SPAN_COLOR,
    alpha: float = 0.9,
    label: str | None = None,
) -> None:
    if mask.size == 0 or not np.any(mask):
        return

    mid_h = _time_axis_hours(usability_scores.timestamps, t0_ns=t0_ns)
    half_h = (USABILITY_EPOCH_NS / NS_PER_HOUR) / 2.0
    labeled = False
    for midpoint, is_marked in zip(mid_h, mask, strict=True):
        if not is_marked:
            continue
        ax.axvspan(
            float(midpoint) - half_h,
            float(midpoint) + half_h,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label if not labeled else None,
            zorder=0,
        )
        labeled = True


def _draw_hypnodensity(
    ax: plt.Axes,
    hypnodensity: object,
    *,
    t0_ns: float | None,
    duration_h: float | None,
    usability_scores: object | None,
) -> None:
    times_h = _time_axis_hours(hypnodensity.timestamps, t0_ns=t0_ns)
    probs = hypnodensity.values
    stage_names = list(hypnodensity.channel_names)

    if times_h.size == 0 or probs.size == 0:
        ax.set_title("Hypnodensity")
        ax.text(0.5, 0.5, "No hypnodensity data", ha="center", va="center", transform=ax.transAxes)
        return

    origin = float(hypnodensity.timestamps[0] if t0_ns is None else t0_ns)
    if usability_scores is not None:
        _overlay_usability_spans(
            ax,
            usability_scores,
            t0_ns=origin,
            mask=_artifact_any_channel_mask(usability_scores),
            label="Unusable",
        )

    ax.stackplot(times_h, probs.T, labels=stage_names, alpha=0.85, zorder=2)
    ax.set_ylabel("Probability")
    ax.set_title("Hypnodensity", pad=28)
    ax.set_ylim(0, 1)
    _apply_shared_xlim(ax, duration_h)
    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(
            handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            borderaxespad=0.0,
            fontsize="small",
            frameon=False,
            ncol=len(legend_labels),
        )


def _draw_hypnogram(
    ax: plt.Axes,
    hypnogram: object,
    *,
    t0_ns: float | None,
    duration_h: float | None,
) -> None:
    labels = [str(label) for label in hypnogram.labels]
    if not labels:
        ax.set_title("Hypnogram")
        ax.text(0.5, 0.5, "No hypnogram data", ha="center", va="center", transform=ax.transAxes)
        return

    y_map = _stage_y_map(labels)
    y_values = [y_map[label] for label in labels]
    epoch_h = hypnogram.period_length / NS_PER_HOUR
    origin_ns = float(hypnogram.onset if t0_ns is None else t0_ns)
    onset_h = (float(hypnogram.onset) - origin_ns) / NS_PER_HOUR
    times_h = onset_h + np.arange(len(labels), dtype=np.float64) * epoch_h

    ax.step(times_h, y_values, where="post", color="black", linewidth=1.2)
    if duration_h is not None and duration_h > 0:
        _apply_shared_xlim(ax, duration_h)
    else:
        ax.set_xlim(times_h[0], times_h[-1] + epoch_h)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_title("Hypnogram")
    ax.grid(True, axis="x", alpha=0.3)


def plot_hypnodensity(
    hypnodensity: object,
    *,
    t0_ns: float | None = None,
    duration_h: float | None = None,
    usability_scores: object | None = None,
) -> Figure:
    """Stacked area plot of fused sleep-stage probabilities over time."""
    fig = _figure(figsize=(12, 4))
    ax = fig.gca()
    _draw_hypnodensity(
        ax,
        hypnodensity,
        t0_ns=t0_ns,
        duration_h=duration_h,
        usability_scores=usability_scores,
    )
    ax.set_xlabel("Time (h)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    return fig


def plot_hypnogram(
    hypnogram: object,
    *,
    t0_ns: float | None = None,
    duration_h: float | None = None,
) -> Figure:
    """Step plot of discrete sleep stages over time."""
    fig = _figure(figsize=(12, 3))
    ax = fig.gca()
    _draw_hypnogram(ax, hypnogram, t0_ns=t0_ns, duration_h=duration_h)
    ax.set_xlabel("Time (h)")
    fig.tight_layout()
    return fig


def plot_sleep_scoring(
    hypnodensity: object,
    hypnogram: object,
    *,
    t0_ns: float | None = None,
    duration_h: float | None = None,
    usability_scores: object | None = None,
) -> Figure:
    """Packed, time-aligned hypnodensity and hypnogram for sleep scoring."""
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        sharex=True,
        gridspec_kw={"height_ratios": [1.4, 1.0]},
    )
    _draw_hypnodensity(
        axes[0],
        hypnodensity,
        t0_ns=t0_ns,
        duration_h=duration_h,
        usability_scores=usability_scores,
    )
    _draw_hypnogram(axes[1], hypnogram, t0_ns=t0_ns, duration_h=duration_h)
    axes[1].set_xlabel("Time (h)")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


def _draw_spectrogram(
    ax: plt.Axes,
    signal_data: np.ndarray,
    fs: float,
    *,
    duration_h: float | None,
) -> None:
    window_size_s = 4.0
    min_frequency_hz = 0.0
    max_frequency_hz = 30.0
    nperseg = min(int(window_size_s * fs), signal_data.size)
    if nperseg < 2:
        ax.text(0.5, 0.5, "Recording too short", ha="center", va="center", transform=ax.transAxes)
        return

    frequencies, times, power = spectrogram(
        signal_data,
        fs=fs,
        nperseg=nperseg,
        noverlap=0,
        window="hann",
        scaling="density",
    )
    power_db = 10.0 * np.log10(power + np.finfo(float).eps)
    freq_mask = np.logical_and(frequencies >= min_frequency_hz, frequencies <= max_frequency_hz)
    frequencies = frequencies[freq_mask]
    power_db = power_db[freq_mask, :]

    extent = (
        times[0] / 3600.0,
        times[-1] / 3600.0,
        float(frequencies[0]) if frequencies.size else min_frequency_hz,
        float(frequencies[-1]) if frequencies.size else max_frequency_hz,
    )
    ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
    )
    ax.set_ylim(min_frequency_hz, max_frequency_hz)
    _apply_shared_xlim(ax, duration_h)


def plot_channel_overview(
    recording: object,
    channel: str,
    usability_scores: object,
    usability_channel_idx: int,
    model_key: str,
    *,
    t0_ns: float | None = None,
    duration_h: float | None = None,
) -> Figure:
    """Aligned signal, spectrogram, and artifact-label panels for one EEG channel."""
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [1.2, 1.6, 0.9]},
    )
    signal_ax, spec_ax, artifact_ax = axes

    origin = float(recording.timestamps[0] if t0_ns is None else t0_ns)

    if channel not in recording.channel_index_map or recording.sample_rate is None:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                f"Channel {channel!r} unavailable",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        fig.tight_layout()
        return fig

    idx = recording.channel_index_map[channel]
    signal = np.asarray(recording.values[:, idx]).squeeze()
    times_h = _time_axis_hours(recording.timestamps, t0_ns=origin)
    plot_times, plot_signal = _downsample_series(times_h, signal)

    signal_ax.plot(plot_times, plot_signal, color="black", linewidth=0.4, alpha=0.85)
    signal_ax.set_ylabel("µV")
    signal_ax.set_title("Signal")
    signal_ax.grid(True, axis="x", alpha=0.25)
    _apply_shared_xlim(signal_ax, duration_h)

    _draw_spectrogram(spec_ax, signal, float(recording.sample_rate), duration_h=duration_h)
    spec_ax.set_ylabel("Hz")
    spec_ax.set_title("Spectrogram")

    label_map = _usability_label_map(model_key)
    art_times = _time_axis_hours(usability_scores.timestamps, t0_ns=origin)
    art_labels = usability_scores.values[:, usability_channel_idx].astype(int)
    if art_times.size:
        artifact_ax.step(art_times, art_labels, where="mid", color="steelblue", linewidth=1.2)
    artifact_ax.set_yticks(sorted(label_map))
    artifact_ax.set_yticklabels([label_map[i] for i in sorted(label_map)], fontsize="small")
    artifact_ax.set_ylabel("Artifact")
    artifact_ax.set_title("Artifact labels")
    artifact_ax.set_xlabel("Time (h)")
    artifact_ax.grid(True, axis="x", alpha=0.3)
    _apply_shared_xlim(artifact_ax, duration_h)

    fig.tight_layout()
    return fig


ZOOM_PAD_S = 0.5
MAX_ZOOM_PANELS_PER_EDGE = 8
ZOOM_GRID_COLS = 2
EEG_LEFT_COLOR = "#2563eb"
EEG_RIGHT_COLOR = "#ea580c"
MATCH_HIGHLIGHT_COLOR = "#dc2626"
ZOOM_FACE_COLOR = "#f8fafc"


def _edge_channel_signals(
    edge: EdgeEyeMovementResult,
    left_channel: str,
    right_channel: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    window = edge.window
    if window.sample_rate is None or window.n_samples == 0:
        return None
    if left_channel not in window.channel_index_map or right_channel not in window.channel_index_map:
        return None
    times_s = (window.timestamps.astype(np.float64) - window.timestamps[0]) / 1e9
    left = np.asarray(window.values[:, window.channel_index_map[left_channel]]).squeeze()
    right = np.asarray(window.values[:, window.channel_index_map[right_channel]]).squeeze()
    return times_s, left, right


def _signal_ylim(*series: np.ndarray, margin: float = 0.12) -> tuple[float, float]:
    data = np.concatenate([np.asarray(s, dtype=np.float64).ravel() for s in series if s.size])
    if data.size == 0:
        return -1.0, 1.0
    lo, hi = np.percentile(data, [2, 98])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(data))
        hi = float(np.max(data))
        if hi <= lo:
            return lo - 1.0, hi + 1.0
    pad = (hi - lo) * margin
    return float(lo - pad), float(hi + pad)


def _style_trace_axis(ax: plt.Axes, *, zoom: bool = False) -> None:
    ax.set_facecolor(ZOOM_FACE_COLOR if zoom else "white")
    ax.grid(True, axis="both", alpha=0.25, linewidth=0.6)
    for spine in ax.spines.values():
        spine.set_color("#cbd5e1")
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=8, colors="#475569")


def _draw_eeg_overlay(
    ax: plt.Axes,
    times_s: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    left_channel: str,
    right_channel: str,
    downsample: bool,
) -> None:
    if downsample:
        plot_times, plot_left = _downsample_series(times_s, left)
        _, plot_right = _downsample_series(times_s, right)
        lw = 0.55
    else:
        plot_times, plot_left, plot_right = times_s, left, right
        lw = 1.15
    ax.plot(plot_times, plot_left, color=EEG_LEFT_COLOR, linewidth=lw, alpha=0.95, label=left_channel)
    ax.plot(plot_times, plot_right, color=EEG_RIGHT_COLOR, linewidth=lw, alpha=0.95, label=right_channel)


def _overlay_events(
    ax: plt.Axes,
    events: list[Event],
    window_start_ns: int,
    *,
    color: str,
    label: str | None = None,
    alpha: float = 0.22,
) -> None:
    if not events:
        return
    labeled = False
    for event in events:
        start_s = (event.onset - window_start_ns) / 1e9
        end_s = start_s + event.duration / 1e9
        ax.axvspan(
            start_s,
            end_s,
            color=color,
            alpha=alpha,
            linewidth=0,
            label=label if (label and not labeled) else None,
            zorder=0,
        )
        labeled = True


def _zoom_limits_for_event(
    event: Event,
    window_start_ns: int,
    window_duration_s: float,
    *,
    pad_s: float = ZOOM_PAD_S,
) -> tuple[float, float]:
    start_s = (event.onset - window_start_ns) / 1e9
    end_s = start_s + event.duration / 1e9
    return max(0.0, start_s - pad_s), min(window_duration_s, end_s + pad_s)


def _plot_edge_overview(
    ax: plt.Axes,
    edge: EdgeEyeMovementResult,
    *,
    left_channel: str,
    right_channel: str,
    section_title: str,
) -> None:
    signals = _edge_channel_signals(edge, left_channel, right_channel)
    ax.set_title(section_title, fontsize=11, fontweight="bold", color="#0f172a", pad=8)
    if signals is None:
        ax.text(0.5, 0.5, "No edge-window data", ha="center", va="center", transform=ax.transAxes)
        _style_trace_axis(ax)
        return

    times_s, left, right = signals
    _draw_eeg_overlay(
        ax,
        times_s,
        left,
        right,
        left_channel=left_channel,
        right_channel=right_channel,
        downsample=True,
    )
    window_start_ns = int(edge.window.timestamps[0])
    _overlay_events(
        ax,
        edge.sequences,
        window_start_ns,
        color=MATCH_HIGHLIGHT_COLOR,
        label="Matched",
        alpha=0.2,
    )

    # Number markers at each match onset for cross-reference with zoom cards.
    for index, event in enumerate(edge.sequences[:MAX_ZOOM_PANELS_PER_EDGE], start=1):
        onset_s = (event.onset - window_start_ns) / 1e9
        ax.axvline(onset_s, color=MATCH_HIGHLIGHT_COLOR, linewidth=0.9, alpha=0.7, zorder=3)
        ymax = ax.get_ylim()[1]
        ax.text(
            onset_s,
            ymax,
            str(index),
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
            color=MATCH_HIGHLIGHT_COLOR,
            clip_on=False,
        )

    ax.set_ylabel("µV", fontsize=9)
    ax.set_xlabel("Time (s)", fontsize=9)
    _style_trace_axis(ax, zoom=False)
    ax.legend(loc="upper right", fontsize=8, frameon=False, ncols=3)


def _plot_sequence_zoom(
    ax: plt.Axes,
    edge: EdgeEyeMovementResult,
    event: Event,
    *,
    index: int,
    left_channel: str,
    right_channel: str,
    ylim: tuple[float, float] | None,
) -> None:
    signals = _edge_channel_signals(edge, left_channel, right_channel)
    label = str(event.label) if event.label is not None else "?"
    if signals is None:
        ax.set_title(f"{index}. {label}", fontsize=10)
        ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
        _style_trace_axis(ax, zoom=True)
        return

    times_s, left, right = signals
    window_start_ns = int(edge.window.timestamps[0])
    window_duration_s = edge.window.duration.total_seconds()
    x0, x1 = _zoom_limits_for_event(event, window_start_ns, window_duration_s)
    mask = (times_s >= x0) & (times_s <= x1)
    onset_s = (event.onset - window_start_ns) / 1e9
    end_s = onset_s + event.duration / 1e9
    duration_ms = event.duration / 1e6

    _draw_eeg_overlay(
        ax,
        times_s[mask],
        left[mask],
        right[mask],
        left_channel=left_channel,
        right_channel=right_channel,
        downsample=False,
    )
    ax.axvspan(onset_s, end_s, color=MATCH_HIGHLIGHT_COLOR, alpha=0.16, linewidth=0, zorder=0)
    ax.axvline(onset_s, color=MATCH_HIGHLIGHT_COLOR, linewidth=1.0, alpha=0.85, zorder=3)
    ax.axvline(end_s, color=MATCH_HIGHLIGHT_COLOR, linewidth=1.0, alpha=0.55, linestyle="--", zorder=3)

    ax.set_xlim(x0, x1)
    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.set_title(
        f"{index}.  {label}",
        fontsize=10,
        fontweight="bold",
        color="#0f172a",
        pad=6,
    )
    ax.text(
        0.99,
        0.96,
        f"@{onset_s:.2f}s · {duration_ms:.0f} ms",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#64748b",
    )
    ax.set_xlabel("Time (s)", fontsize=8)
    ax.set_ylabel("µV", fontsize=8)
    _style_trace_axis(ax, zoom=True)


def _add_edge_section(
    fig: Figure,
    outer: object,
    edge: EdgeEyeMovementResult,
    *,
    section_title: str,
    left_channel: str,
    right_channel: str,
) -> None:
    from matplotlib.gridspec import GridSpecFromSubplotSpec

    sequences = list(edge.sequences[:MAX_ZOOM_PANELS_PER_EDGE])
    n_zooms = len(sequences)

    if n_zooms == 0:
        overview_ax = fig.add_subplot(outer)
        _plot_edge_overview(
            overview_ax,
            edge,
            left_channel=left_channel,
            right_channel=right_channel,
            section_title=section_title,
        )
        return

    zoom_rows = int(np.ceil(n_zooms / ZOOM_GRID_COLS))
    section = GridSpecFromSubplotSpec(
        2,
        1,
        subplot_spec=outer,
        height_ratios=[1.45, zoom_rows],
        hspace=0.38,
    )
    overview_ax = fig.add_subplot(section[0])
    _plot_edge_overview(
        overview_ax,
        edge,
        left_channel=left_channel,
        right_channel=right_channel,
        section_title=section_title,
    )

    # Shared y-limits across zooms in this section for easier comparison.
    signals = _edge_channel_signals(edge, left_channel, right_channel)
    shared_ylim: tuple[float, float] | None = None
    if signals is not None:
        times_s, left, right = signals
        window_start_ns = int(edge.window.timestamps[0])
        window_duration_s = edge.window.duration.total_seconds()
        chunks: list[np.ndarray] = []
        for event in sequences:
            x0, x1 = _zoom_limits_for_event(event, window_start_ns, window_duration_s)
            mask = (times_s >= x0) & (times_s <= x1)
            chunks.extend([left[mask], right[mask]])
        shared_ylim = _signal_ylim(*chunks)

    zoom_grid = GridSpecFromSubplotSpec(
        zoom_rows,
        ZOOM_GRID_COLS,
        subplot_spec=section[1],
        wspace=0.28,
        hspace=0.5,
    )
    for index, event in enumerate(sequences, start=1):
        row, col = divmod(index - 1, ZOOM_GRID_COLS)
        ax = fig.add_subplot(zoom_grid[row, col])
        _plot_sequence_zoom(
            ax,
            edge,
            event,
            index=index,
            left_channel=left_channel,
            right_channel=right_channel,
            ylim=shared_ylim,
        )

    for filler in range(n_zooms, zoom_rows * ZOOM_GRID_COLS):
        row, col = divmod(filler, ZOOM_GRID_COLS)
        ax = fig.add_subplot(zoom_grid[row, col])
        ax.set_visible(False)


def plot_eye_movements(
    edge_start: EdgeEyeMovementResult,
    edge_end: EdgeEyeMovementResult,
    *,
    left_channel: str,
    right_channel: str,
) -> Figure:
    """Eye-movement block: edge overviews with a card grid of zoomed matches."""
    from matplotlib.gridspec import GridSpec

    n_start = min(len(edge_start.sequences), MAX_ZOOM_PANELS_PER_EDGE)
    n_end = min(len(edge_end.sequences), MAX_ZOOM_PANELS_PER_EDGE)
    start_zoom_rows = int(np.ceil(n_start / ZOOM_GRID_COLS)) if n_start else 0
    end_zoom_rows = int(np.ceil(n_end / ZOOM_GRID_COLS)) if n_end else 0
    start_weight = 1.6 + 1.05 * start_zoom_rows
    end_weight = 1.6 + 1.05 * end_zoom_rows
    fig_h = 2.4 * start_weight + 2.4 * end_weight

    fig = plt.figure(figsize=(12.5, max(6.5, fig_h)))
    outer = GridSpec(2, 1, figure=fig, height_ratios=[start_weight, end_weight], hspace=0.3)

    _add_edge_section(
        fig,
        outer[0],
        edge_start,
        section_title="First edge window",
        left_channel=left_channel,
        right_channel=right_channel,
    )
    _add_edge_section(
        fig,
        outer[1],
        edge_end,
        section_title="Last edge window",
        left_channel=left_channel,
        right_channel=right_channel,
    )
    fig.subplots_adjust(left=0.07, right=0.98, top=0.95, bottom=0.05)
    return fig


def build_plots(result: AnalysisResult) -> dict[str, Figure]:
    """Build per-channel EEG panels plus sleep and eye-movement plots.

    Full-recording plots share the recording start as time origin and the same
    x-axis span so they line up when compared visually.
    """
    config = result.config
    t0_ns, duration_h = _recording_time_span(result.recording)
    eeg_channels = list(result.eeg_channels) or available_eeg_channels(config, result.recording)

    plots: dict[str, Figure] = {}
    for channel in eeg_channels:
        usability_idx = _usability_channel_index(result.usability_scores, channel, config)
        plots[f"channel_{channel}"] = plot_channel_overview(
            result.recording,
            channel,
            result.usability_scores,
            usability_idx,
            config.usability_model,
            t0_ns=t0_ns,
            duration_h=duration_h,
        )

    plots["sleep_scoring"] = plot_sleep_scoring(
        result.hypnodensity,
        result.hypnogram,
        t0_ns=t0_ns,
        duration_h=duration_h,
        usability_scores=result.usability_scores,
    )
    plots["eye_movements"] = plot_eye_movements(
        result.edge_start,
        result.edge_end,
        left_channel=config.eeg_left,
        right_channel=config.eeg_right,
    )
    return plots
