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


def _plot_edge_overlay(
    ax: plt.Axes,
    edge: EdgeEyeMovementResult,
    *,
    left_channel: str,
    right_channel: str,
    panel_title: str,
) -> None:
    """Overlay left/right EEG for one edge window and highlight matched sequences."""
    window = edge.window
    ax.set_title(panel_title)
    if window.sample_rate is None or window.n_samples == 0:
        ax.text(0.5, 0.5, "No edge-window data", ha="center", va="center", transform=ax.transAxes)
        return

    times_s = (window.timestamps.astype(np.float64) - window.timestamps[0]) / 1e9
    missing = [
        channel
        for channel in (left_channel, right_channel)
        if channel not in window.channel_index_map
    ]
    if missing:
        ax.text(
            0.5,
            0.5,
            f"Missing channels: {', '.join(missing)}",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    left_idx = window.channel_index_map[left_channel]
    right_idx = window.channel_index_map[right_channel]
    left_signal = np.asarray(window.values[:, left_idx]).squeeze()
    right_signal = np.asarray(window.values[:, right_idx]).squeeze()
    plot_times, plot_left = _downsample_series(times_s, left_signal)
    _, plot_right = _downsample_series(times_s, right_signal)

    ax.plot(plot_times, plot_left, color="C0", linewidth=0.6, alpha=0.85, label=left_channel)
    ax.plot(plot_times, plot_right, color="C1", linewidth=0.6, alpha=0.85, label=right_channel)
    _overlay_events(
        ax,
        edge.sequences,
        int(window.timestamps[0]),
        color="crimson",
        label="Matched sequences",
    )
    ax.set_ylabel("µV")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize="small")


def plot_eye_movements(
    edge_start: EdgeEyeMovementResult,
    edge_end: EdgeEyeMovementResult,
    *,
    left_channel: str,
    right_channel: str,
) -> Figure:
    """Two-panel eye-movement block: first and last edge windows, channels overlaid."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)
    _plot_edge_overlay(
        axes[0],
        edge_start,
        left_channel=left_channel,
        right_channel=right_channel,
        panel_title="First edge window",
    )
    _plot_edge_overlay(
        axes[1],
        edge_end,
        left_channel=left_channel,
        right_channel=right_channel,
        panel_title="Last edge window",
    )
    axes[0].set_xlabel("Time (s)")
    axes[1].set_xlabel("Time (s)")
    fig.tight_layout()
    return fig


def _overlay_events(
    ax: plt.Axes,
    events: list[Event],
    window_start_ns: int,
    *,
    color: str,
    label: str,
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
            alpha=0.25,
            label=label if not labeled else None,
        )
        labeled = True


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
