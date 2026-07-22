"""End-to-end analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from somnio.data import Epochs, Event, TimeSeries
from somnio.tasks.eeg_usability import get_usability_scores, load_model as load_usability_model
from somnio.tasks.eye_movement_detection import detect_lr_eye_movements
from somnio.tasks.sleep_scoring import score_sleep_stages
from somnio.tasks.sleep_scoring.models import OnnxSleepScoringModel
from somnio.tasks.sleep_scoring.schema import ModelMetadata
from somnio.transforms.clip import apply_clip_iqr
from somnio.transforms.filter import apply_fir_filter
from somnio.transforms.resample import apply_resample
from somnio.transforms.scale import apply_scale

from nightwatch.config import AnalysisConfig
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
    """Outputs from a completed analysis run."""

    config: AnalysisConfig
    recording: TimeSeries
    raw_channel_names: tuple[str, ...]
    eeg_channels: tuple[str, ...]
    hypnodensity: TimeSeries
    hypnogram: Epochs
    usability_scores: TimeSeries
    usability_samples_to_keep: int
    usability_epoch_length: int
    edge_start: EdgeEyeMovementResult
    edge_end: EdgeEyeMovementResult


def available_eeg_channels(config: AnalysisConfig, recording: TimeSeries) -> list[str]:
    """Return configured EEG channels that are present in the recording."""
    channels = [
        name
        for name in (config.eeg_left, config.eeg_right)
        if name in recording.channel_index_map
    ]
    if not channels:
        raise ValueError(
            f"Recording has none of the configured EEG channels "
            f"({config.eeg_left!r}, {config.eeg_right!r})."
        )
    return channels


def _prepare_sleep_scoring_channel(
    ts: TimeSeries,
    channel: str,
    metadata: ModelMetadata,
) -> TimeSeries:
    """Select one EEG channel and apply sleep-scoring preprocessing."""
    if channel not in ts.channel_index_map:
        raise ValueError(f"Recording is missing sleep-scoring channel: {channel!r}")
    if ts.sample_rate is None:
        raise ValueError("Recording has no sample_rate metadata; cannot score sleep stages.")
    if metadata.n_channels != 1:
        raise ValueError(
            f"Soft-fused sleep scoring requires a 1-channel model; "
            f"got n_channels={metadata.n_channels}."
        )

    selected = ts.select_channels([channel])
    target_hz = float(metadata.sample_rate_hz)
    selected = apply_resample(selected, target_hz)
    selected = apply_fir_filter(selected, low_cutoff=0.3, high_cutoff=35.0)
    selected = apply_scale(selected, method="robust")
    return apply_clip_iqr(selected, iqr_factor=20.0)


def soft_fuse_hypnodensities(hypnodensities: list[TimeSeries]) -> TimeSeries:
    """Average per-channel hypnodensity probabilities (soft fusion)."""
    if not hypnodensities:
        raise ValueError("At least one hypnodensity is required for soft fusion.")
    if len(hypnodensities) == 1:
        return hypnodensities[0]

    ref = hypnodensities[0]
    for other in hypnodensities[1:]:
        if list(other.channel_names) != list(ref.channel_names):
            raise ValueError("Hypnodensity class labels must match for soft fusion.")
        if other.values.shape != ref.values.shape:
            raise ValueError(
                "Hypnodensity shapes must match for soft fusion: "
                f"{ref.values.shape} vs {other.values.shape}."
            )

    stacked = np.stack([hd.values for hd in hypnodensities], axis=0)
    fused = np.mean(stacked, axis=0)
    row_sums = fused.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0.0, row_sums, 1.0)
    fused = fused / row_sums

    return TimeSeries(
        values=np.asarray(fused, dtype=np.float64),
        timestamps=ref.timestamps.copy(),
        channel_names=list(ref.channel_names),
        units=list(ref.units),
        sample_rate=ref.sample_rate,
    )


def hypnogram_from_hypnodensity(
    hypnodensity: TimeSeries,
    *,
    onset_ns: int,
    epoch_ns: int = SLEEP_EPOCH_NS,
) -> Epochs:
    """Aggregate hypnodensity into fixed-length epochs via mean-then-argmax."""
    times = np.asarray(hypnodensity.timestamps, dtype=np.int64)
    probs = np.asarray(hypnodensity.values, dtype=np.float64)
    stage_names = [str(name) for name in hypnodensity.channel_names]

    if times.size == 0 or probs.size == 0:
        return Epochs(labels=np.array([], dtype=object), period_length=epoch_ns, onset=onset_ns)

    n_epochs = max(1, int((int(times[-1]) - onset_ns) // epoch_ns) + 1)
    labels: list[str] = []
    for index in range(n_epochs):
        start = onset_ns + index * epoch_ns
        stop = start + epoch_ns
        mask = (times >= start) & (times < stop)
        if not np.any(mask):
            labels.append(UNUSABLE_LABEL)
            continue
        mean_prob = probs[mask].mean(axis=0)
        labels.append(stage_names[int(np.argmax(mean_prob))])

    return Epochs(
        labels=np.asarray(labels, dtype=object),
        period_length=epoch_ns,
        onset=onset_ns,
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
    config: AnalysisConfig,
    *,
    edge_minutes: float,
    at_start: bool,
) -> EdgeEyeMovementResult:
    """Run eye-movement detection on the first or last edge window."""
    start_window, end_window = slice_edge_windows(ts, edge_minutes)
    window = start_window if at_start else end_window
    eeg_window = window.select_channels([config.eeg_left, config.eeg_right])
    sequences, primitives = detect_lr_eye_movements(
        eeg_window,
        left=config.eeg_left,
        right=config.eeg_right,
        accepted_pattern=config.eye_movement_pattern,
    )
    return EdgeEyeMovementResult(
        window=window,
        sequences=sequences,
        primitives=primitives,
    )


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run sleep scoring, usability, and edge eye-movement detection.

    Sleep scoring runs independently on each available EEG channel using a
    1-channel model; hypnodensities are soft-fused (averaged). Hypnogram epochs
    with fewer than two fully usable 10-second windows are labeled Unusable.

    Args:
        config: Analysis settings.

    Raises:
        FileNotFoundError: If the sleep-scoring model path does not exist.
        ValueError: If required channels are missing or inputs are invalid.
    """
    if not config.model_path.is_file():
        raise FileNotFoundError(config.model_path)

    loaded = load_recording(config)
    recording = loaded.timeseries
    eeg_channels = available_eeg_channels(config, recording)

    sleep_model = OnnxSleepScoringModel.load(config.model_path)
    per_channel: list[TimeSeries] = []
    for channel in eeg_channels:
        sleep_ts = _prepare_sleep_scoring_channel(recording, channel, sleep_model.metadata)
        hypnodensity = score_sleep_stages(
            sleep_ts,
            backend=sleep_model,
            metadata=sleep_model.metadata,
            output="probs_timeseries",
        )
        assert isinstance(hypnodensity, TimeSeries)
        per_channel.append(hypnodensity)

    fused_hypnodensity = soft_fuse_hypnodensities(per_channel)
    onset_ns = int(recording.timestamps[0]) if recording.n_samples else 0
    hypnogram = hypnogram_from_hypnodensity(fused_hypnodensity, onset_ns=onset_ns)

    usability_model = load_usability_model(config.usability_model)
    usability_scores, samples_to_keep, epoch_length = get_usability_scores(
        recording,
        usability_model,
        eeg_left=config.eeg_left,
        eeg_right=config.eeg_right,
        movement=config.movement,
    )
    hypnogram = mark_unusable_hypnogram_epochs(hypnogram, usability_scores)

    edge_start = _detect_edge_eye_movements(
        recording,
        config,
        edge_minutes=config.edge_minutes,
        at_start=True,
    )
    edge_end = _detect_edge_eye_movements(
        recording,
        config,
        edge_minutes=config.edge_minutes,
        at_start=False,
    )

    return AnalysisResult(
        config=config,
        recording=recording,
        raw_channel_names=loaded.raw_channel_names,
        eeg_channels=tuple(eeg_channels),
        hypnodensity=fused_hypnodensity,
        hypnogram=hypnogram,
        usability_scores=usability_scores,
        usability_samples_to_keep=samples_to_keep,
        usability_epoch_length=epoch_length,
        edge_start=edge_start,
        edge_end=edge_end,
    )
