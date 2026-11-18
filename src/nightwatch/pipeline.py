"""End-to-end analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from somnio.data import Epochs, Event, TimeSeries
from somnio.tasks.eeg_usability import (
    get_usability_score,
    load_model as load_usability_model,
)
from somnio.tasks.eeg_usability.defaults import SAMPLE_RATE_HZ
from somnio.tasks.eye_movement_detection import detect_lr_eye_movements
from somnio.transforms.resample import apply_resample
from wu_sleep import labels_from_probabilities, score_sleep_stages

from nightwatch.config import (
    WU_SLEEP_CLASS_LABELS,
    AnalysisConfig,
    apply_zmax_view_defaults,
    resolve_wu_sleep_model_path,
)
from nightwatch.load import load_recording

SLEEP_EPOCH_NS = 30_000_000_000
USABILITY_EPOCH_NS = 10_000_000_000
MIN_USABLE_WINDOWS_PER_SLEEP_EPOCH = 2
UNUSABLE_LABEL = "Unusable"


@dataclass
class EdgeEyeMovementResult:
    """Eye-movement detection output for one recording edge window."""

    window: TimeSeries
    sequences: list[Event]
    primitives: list[Event]


@dataclass
class AnalysisResult:
    """Outputs from a completed analysis run.

    Optional fields are ``None`` when that view was skipped.
    """

    config: AnalysisConfig
    recording: TimeSeries
    raw_channel_names: tuple[str, ...]
    sleep_channels: tuple[str, ...]
    hypnodensity: TimeSeries | None
    hypnogram: Epochs | None
    usability_scores: TimeSeries | None
    usability_samples_to_keep: int | None
    usability_epoch_length: int | None
    edge_start: EdgeEyeMovementResult | None
    edge_end: EdgeEyeMovementResult | None


def _available_names(recording: TimeSeries) -> str:
    return ", ".join(recording.channel_names) or "(none)"


def require_channels(recording: TimeSeries, names: list[str] | tuple[str, ...]) -> None:
    """Raise if any requested channel is missing from the recording."""
    missing = [name for name in names if name not in recording.channel_index_map]
    if missing:
        raise ValueError(
            f"Recording is missing channel(s) {missing}. "
            f"Available: {_available_names(recording)}"
        )


def mark_unusable_hypnogram_epochs(
    hypnogram: Epochs,
    usability_scores: TimeSeries,
    *,
    min_usable_windows: int = MIN_USABLE_WINDOWS_PER_SLEEP_EPOCH,
) -> Epochs:
    """Label sleep epochs Unusable when too few 10 s windows are fully usable.

    A 10-second usability window counts as usable only when every scored
    electrode is label 0. A 30-second hypnogram epoch is Unusable when fewer
    than ``min_usable_windows`` such windows have midpoints inside it.
    """
    if len(hypnogram.labels) == 0 or usability_scores.n_samples == 0:
        return hypnogram

    scores = np.asarray(usability_scores.values)
    usable_windows = np.all(scores == 0, axis=1)
    midpoints = np.asarray(usability_scores.timestamps, dtype=np.int64)

    labels = [str(label) for label in hypnogram.labels]
    for index in range(len(labels)):
        start = hypnogram.onset + index * hypnogram.period_length
        stop = start + hypnogram.period_length
        in_epoch = (midpoints >= start) & (midpoints < stop)
        if int(np.sum(usable_windows[in_epoch])) < min_usable_windows:
            labels[index] = UNUSABLE_LABEL

    return Epochs(
        labels=np.asarray(labels, dtype=object),
        period_length=hypnogram.period_length,
        onset=hypnogram.onset,
    )


def edge_window_sample_count(ts: TimeSeries, edge_minutes: float) -> int:
    """Return the number of samples in one edge window, capped by recording length."""
    if ts.sample_rate is None:
        raise ValueError("Recording has no sample_rate metadata; cannot slice edge windows.")
    if edge_minutes <= 0:
        raise ValueError(f"edge_minutes must be positive, got {edge_minutes}")
    requested = int(round(edge_minutes * 60.0 * ts.sample_rate))
    return min(requested, ts.n_samples)


def slice_edge_windows(ts: TimeSeries, edge_minutes: float) -> tuple[TimeSeries, TimeSeries]:
    """Return the first and last ``edge_minutes`` slices of a recording."""
    n = edge_window_sample_count(ts, edge_minutes)
    if n == 0:
        raise ValueError("Recording has no samples; cannot slice edge windows.")
    return ts[:n], ts[-n:]


def _detect_edge_eye_movements(
    ts: TimeSeries,
    *,
    left: str,
    right: str,
    edge_minutes: float,
    pattern: str,
    at_start: bool,
) -> EdgeEyeMovementResult:
    """Run eye-movement detection on the first or last edge window."""
    start_window, end_window = slice_edge_windows(ts, edge_minutes)
    window = start_window if at_start else end_window
    eeg_window = window.select_channels([left, right])
    sequences, primitives = detect_lr_eye_movements(
        eeg_window,
        left=left,
        right=right,
        accepted_pattern=pattern,
    )
    return EdgeEyeMovementResult(
        window=window,
        sequences=sequences,
        primitives=primitives,
    )


def _run_sleep_scoring(
    recording: TimeSeries,
    config: AnalysisConfig,
) -> tuple[TimeSeries, Epochs, tuple[str, ...]] | None:
    """Score sleep stages with wu-sleep when sleep channels are configured."""
    channels = list(config.sleep_channels)
    if not channels:
        return None

    model_dir = resolve_wu_sleep_model_path(config.model_path, config.sleep_model)
    require_channels(recording, channels)
    if recording.sample_rate is None:
        raise ValueError("Recording has no sample_rate metadata; cannot score sleep stages.")
    if recording.n_samples == 0:
        raise ValueError("Recording has no samples; cannot score sleep stages.")

    selected = recording.select_channels(channels)
    class_labels = list(WU_SLEEP_CLASS_LABELS)
    probs = score_sleep_stages(
        np.asarray(selected.values),
        sample_rate_hz=float(recording.sample_rate),
        model_path=model_dir,
        channel_names=list(channels),
        output="probs",
    )
    probs = np.asarray(probs, dtype=np.float64)
    if probs.ndim != 2 or probs.shape[1] != len(class_labels):
        raise ValueError(
            f"wu-sleep returned unexpected probability shape {probs.shape}; "
            f"expected (n_epochs, {len(class_labels)})."
        )

    labels = labels_from_probabilities(probs, class_labels)
    onset_ns = int(recording.timestamps[0])
    n_epochs = int(probs.shape[0])
    timestamps = onset_ns + np.arange(n_epochs, dtype=np.int64) * SLEEP_EPOCH_NS + SLEEP_EPOCH_NS // 2
    hypnodensity = TimeSeries(
        values=probs,
        timestamps=timestamps,
        channel_names=class_labels,
        units=["1"] * len(class_labels),
        sample_rate=1.0 / 30.0,
    )
    hypnogram = Epochs(
        labels=np.asarray(labels, dtype=object),
        period_length=SLEEP_EPOCH_NS,
        onset=onset_ns,
    )
    return hypnodensity, hypnogram, tuple(channels)


def _run_artifact_detection(
    recording: TimeSeries,
    config: AnalysisConfig,
) -> tuple[TimeSeries, int, int] | None:
    """Score EEG usability when EEG + movement channels are set; else ``None``."""
    eeg_channels = list(config.artifact_eeg_channels)
    movement = config.movement_channel
    if not eeg_channels or movement is None:
        return None

    require_channels(recording, [*eeg_channels, movement])
    selected = recording.select_channels([*eeg_channels, movement])
    if selected.sample_rate is None:
        raise ValueError("Recording has no sample_rate metadata; cannot score usability.")
    if not np.isclose(selected.sample_rate, SAMPLE_RATE_HZ, rtol=0.0, atol=1e-6):
        selected = apply_resample(selected, SAMPLE_RATE_HZ)

    usability_model = load_usability_model(config.usability_model)
    score_columns: list[np.ndarray] = []
    samples_to_keep = 0
    epoch_length = 0
    timestamps: np.ndarray | None = None

    for eeg in eeg_channels:
        scores, samples_to_keep, epoch_length = get_usability_score(
            selected,
            usability_model,
            eeg=eeg,
            movement=movement,
            output_channel=eeg,
        )
        score_columns.append(np.asarray(scores.values).reshape(-1))
        timestamps = np.asarray(scores.timestamps, dtype=np.int64)

    assert timestamps is not None
    usability = TimeSeries(
        values=np.column_stack(score_columns),
        timestamps=timestamps,
        channel_names=list(eeg_channels),
        units=["1"] * len(eeg_channels),
        sample_rate=scores.sample_rate,
    )
    return usability, samples_to_keep, epoch_length


def _run_eye_movements(
    recording: TimeSeries,
    config: AnalysisConfig,
) -> tuple[EdgeEyeMovementResult, EdgeEyeMovementResult] | None:
    """Detect edge eye movements when left and right channels are set."""
    left = config.eye_left
    right = config.eye_right
    if left is None or right is None:
        return None

    require_channels(recording, [left, right])
    edge_start = _detect_edge_eye_movements(
        recording,
        left=left,
        right=right,
        edge_minutes=config.edge_minutes,
        pattern=config.eye_movement_pattern,
        at_start=True,
    )
    edge_end = _detect_edge_eye_movements(
        recording,
        left=left,
        right=right,
        edge_minutes=config.edge_minutes,
        pattern=config.eye_movement_pattern,
        at_start=False,
    )
    return edge_start, edge_end


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run enabled views: sleep scoring, usability, and edge eye-movement detection.

    Empty channel sets skip that view. Sleep scoring uses wu-sleep and resolves
    ``sleep_model`` under ``model_path`` (or a discoverable wu-sleep ``model/``
    tree) when ``sleep_channels`` is non-empty. ZMax empty-view configs are filled
    with EEG_L / EEG_R / MOVEMENT defaults after load when those channels exist.

    Args:
        config: Analysis settings.

    Raises:
        FileNotFoundError: If the wu-sleep model directory does not exist.
        ValueError: If required channels are missing or inputs are invalid.
    """
    loaded = load_recording(config)
    recording = loaded.timeseries
    config = apply_zmax_view_defaults(config, recording.channel_names)

    plot_names = list(dict.fromkeys([*config.raw_channels, *config.spectrogram_channels]))
    if plot_names:
        require_channels(recording, plot_names)

    sleep_out = _run_sleep_scoring(recording, config)
    if sleep_out is None:
        hypnodensity: TimeSeries | None = None
        hypnogram: Epochs | None = None
        sleep_channels: tuple[str, ...] = ()
    else:
        hypnodensity, hypnogram, sleep_channels = sleep_out

    artifact_out = _run_artifact_detection(recording, config)
    if artifact_out is None:
        usability_scores: TimeSeries | None = None
        samples_to_keep: int | None = None
        epoch_length: int | None = None
    else:
        usability_scores, samples_to_keep, epoch_length = artifact_out
        if hypnogram is not None:
            hypnogram = mark_unusable_hypnogram_epochs(hypnogram, usability_scores)

    eye_out = _run_eye_movements(recording, config)
    if eye_out is None:
        edge_start: EdgeEyeMovementResult | None = None
        edge_end: EdgeEyeMovementResult | None = None
    else:
        edge_start, edge_end = eye_out

    return AnalysisResult(
        config=config,
        recording=recording,
        raw_channel_names=loaded.raw_channel_names,
        sleep_channels=sleep_channels,
        hypnodensity=hypnodensity,
        hypnogram=hypnogram,
        usability_scores=usability_scores,
        usability_samples_to_keep=samples_to_keep,
        usability_epoch_length=epoch_length,
        edge_start=edge_start,
        edge_end=edge_end,
    )
