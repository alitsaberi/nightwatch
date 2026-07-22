"""Matplotlib figures for reports and Streamlit."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.signal import spectrogram

from nightwatch.metrics import USABILITY_LABELS_BINARY, USABILITY_LABELS_MULTICLASS

if TYPE_CHECKING:
    from somnio.data.annotations import Event

    from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult

DEFAULT_STAGE_ORDER = ("W", "N1", "N2", "N3", "REM", "R")


def _figure(**kwargs: object) -> Figure:
    fig, _ = plt.subplots(**kwargs)  # type: ignore[arg-type]
    return fig


def _time_axis_hours(timestamps_ns: np.ndarray) -> np.ndarray:
    if timestamps_ns.size == 0:
        return np.array([], dtype=np.float64)
    t0 = float(timestamps_ns[0])
    return (timestamps_ns.astype(np.float64) - t0) / 3.6e12


def _stage_y_map(labels: tuple[str, ...] | list[str]) -> dict[str, int]:
    ordered = [stage for stage in DEFAULT_STAGE_ORDER if stage in labels]
    ordered.extend(stage for stage in labels if stage not in ordered)
    return {stage: idx for idx, stage in enumerate(ordered)}


def plot_hypnodensity(hypnodensity: object) -> Figure:
    """Stacked area plot of sleep-stage class probabilities over time."""
    fig = _figure(figsize=(12, 4))
    ax = fig.gca()

    times_h = _time_axis_hours(hypnodensity.timestamps)
    probs = hypnodensity.values
    stage_names = list(hypnodensity.channel_names)

    if times_h.size == 0 or probs.size == 0:
        ax.set_title("Hypnodensity")
        ax.text(0.5, 0.5, "No hypnodensity data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    ax.stackplot(times_h, probs.T, labels=stage_names, alpha=0.85)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Probability")
    ax.set_title("Hypnodensity")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize="small", ncol=min(3, len(stage_names)))
    fig.tight_layout()
    return fig


def plot_hypnogram(hypnogram: object) -> Figure:
    """Step plot of discrete sleep stages over time."""
    fig = _figure(figsize=(12, 3))
    ax = fig.gca()

    labels = [str(label) for label in hypnogram.labels]
    if not labels:
        ax.set_title("Hypnogram")
        ax.text(0.5, 0.5, "No hypnogram data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    y_map = _stage_y_map(labels)
    y_values = [y_map[label] for label in labels]
    epoch_s = hypnogram.period_length / 1e9
    onset_s = hypnogram.onset / 1e9
    times_h = (onset_s / 3600.0) + np.arange(len(labels), dtype=np.float64) * (epoch_s / 3600.0)

    ax.step(times_h, y_values, where="post", color="black", linewidth=1.2)
    ax.set_xlim(times_h[0], times_h[-1] + epoch_s / 3600.0)
    ax.set_yticks(list(y_map.values()))
    ax.set_yticklabels(list(y_map.keys()))
    ax.set_xlabel("Time (h)")
    ax.set_title("Hypnogram")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    return fig


def plot_spectrogram(recording: object, channel: str) -> Figure:
    """Spectrogram of one EEG channel across the full recording."""
    fig = _figure(figsize=(12, 4))
    ax = fig.gca()

    if recording.sample_rate is None:
        ax.set_title(f"Spectrogram ({channel})")
        ax.text(0.5, 0.5, "Sample rate unavailable", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

    if channel not in recording.channel_index_map:
        ax.set_title(f"Spectrogram ({channel})")
        ax.text(
            0.5,
            0.5,
            f"Channel {channel!r} not found",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        fig.tight_layout()
        return fig

    idx = recording.channel_index_map[channel]
    signal_data = np.asarray(recording.values[:, idx]).squeeze()
    fs = float(recording.sample_rate)

    window_size_s = 4.0
    min_frequency_hz = 0.0
    max_frequency_hz = 30.0
    nperseg = min(int(window_size_s * fs), signal_data.size)
    if nperseg < 2:
        ax.set_title(f"Spectrogram ({channel})")
        ax.text(0.5, 0.5, "Recording too short", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()
        return fig

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
    mesh = ax.imshow(
        power_db,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
    )
    ax.set_ylim(min_frequency_hz, max_frequency_hz)
    ax.set_xlabel("Time (h)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_title(f"Spectrogram ({channel})")
    fig.colorbar(mesh, ax=ax, label="Power (dB)")
    fig.tight_layout()
    return fig


def _usability_label_map(model_key: str) -> dict[int, str]:
    if model_key in {"binary", "lite_binary"}:
        return USABILITY_LABELS_BINARY
    return USABILITY_LABELS_MULTICLASS


def plot_usability_timeline(usability_scores: object, model_key: str) -> Figure:
    """Timeline of left/right usability labels."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 3), sharex=True)
    label_map = _usability_label_map(model_key)
    times_h = _time_axis_hours(usability_scores.timestamps)
    channel_titles = ("Left", "Right")

    for ax, channel_idx, title in zip(axes, (0, 1), channel_titles, strict=True):
        labels = usability_scores.values[:, channel_idx].astype(int)
        if times_h.size == 0:
            ax.set_title(f"Usability ({title})")
            continue

        ax.step(times_h, labels, where="post", color="steelblue", linewidth=1.2)
        ax.set_yticks(sorted(label_map))
        ax.set_yticklabels([label_map[i] for i in sorted(label_map)], fontsize="small")
        ax.set_ylabel(title)
        ax.set_title(f"Usability ({title})")
        ax.grid(True, axis="x", alpha=0.3)

    axes[-1].set_xlabel("Time (h)")
    fig.tight_layout()
    return fig


def _plot_edge_eye_movements(
    edge: EdgeEyeMovementResult,
    *,
    title: str,
    left_channel: str,
    right_channel: str,
) -> Figure:
    """Plot EEG traces for an edge window with detected eye-movement events."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
    window = edge.window
    if window.sample_rate is None or window.n_samples == 0:
        for ax in axes:
            ax.text(0.5, 0.5, "No edge-window data", ha="center", va="center", transform=ax.transAxes)
        fig.suptitle(title)
        fig.tight_layout()
        return fig

    times_s = (window.timestamps.astype(np.float64) - window.timestamps[0]) / 1e9
    channels = (left_channel, right_channel)

    for ax, channel in zip(axes, channels, strict=True):
        if channel not in window.channel_index_map:
            ax.set_title(channel)
            ax.text(
                0.5,
                0.5,
                f"Channel {channel!r} missing",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        idx = window.channel_index_map[channel]
        ax.plot(times_s, window.values[:, idx], color="black", linewidth=0.6, alpha=0.8)
        ax.set_ylabel(f"{channel} (µV)")
        ax.set_title(channel)
        ax.grid(True, alpha=0.3)

    _overlay_events(axes[0], edge.sequences, window.timestamps[0], color="crimson", label="sequences")
    _overlay_events(axes[1], edge.primitives, window.timestamps[0], color="darkorange", label="primitives")

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
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

    if labeled:
        ax.legend(loc="upper right", fontsize="small")


def build_plots(result: AnalysisResult) -> dict[str, Figure]:
    """Build hypnodensity, hypnogram, spectrogram, and auxiliary plots.

    Args:
        result: Completed pipeline output.

    Returns:
        Named matplotlib figures ready for embedding or display.
    """
    config = result.config
    return {
        "hypnodensity": plot_hypnodensity(result.hypnodensity),
        "hypnogram": plot_hypnogram(result.hypnogram),
        "spectrogram": plot_spectrogram(result.recording, config.spectrogram_channel),
        "usability": plot_usability_timeline(result.usability_scores, config.usability_model),
        "eye_movement_start": _plot_edge_eye_movements(
            result.edge_start,
            title="Eye movements (first edge window)",
            left_channel=config.eeg_left,
            right_channel=config.eeg_right,
        ),
        "eye_movement_end": _plot_edge_eye_movements(
            result.edge_end,
            title="Eye movements (last edge window)",
            left_channel=config.eeg_left,
            right_channel=config.eeg_right,
        ),
    }
