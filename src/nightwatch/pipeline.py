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
from somnio.transforms.resample import apply_resample

from nightwatch.config import AnalysisConfig
from nightwatch.load import load_recording


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
    hypnodensity: TimeSeries
    hypnogram: Epochs
    usability_scores: TimeSeries
    usability_samples_to_keep: int
    usability_epoch_length: int
    edge_start: EdgeEyeMovementResult
    edge_end: EdgeEyeMovementResult


def _sleep_scoring_channel_names(config: AnalysisConfig, metadata: ModelMetadata) -> list[str]:
    """Return ordered channel names to feed the sleep-scoring model."""
    if metadata.n_channels == 1:
        return [config.eeg_left]
    if metadata.n_channels == 2:
        return [config.eeg_left, config.eeg_right]

    raise ValueError(
        f"Sleep model expects {metadata.n_channels} channels; "
        f"nightwatch only configures left/right EEG ({config.eeg_left!r}, "
        f"{config.eeg_right!r})."
    )


def _prepare_sleep_scoring_timeseries(
    ts: TimeSeries,
    config: AnalysisConfig,
    metadata: ModelMetadata,
) -> TimeSeries:
    """Select model channels and resample to the model sample rate when needed."""
    channel_names = _sleep_scoring_channel_names(config, metadata)
    missing = [name for name in channel_names if name not in ts.channel_index_map]
    if missing:
        raise ValueError(f"Recording is missing sleep-scoring channels: {missing}")

    selected = ts.select_channels(channel_names)
    target_hz = float(metadata.sample_rate_hz)
    if selected.sample_rate is None:
        raise ValueError("Recording has no sample_rate metadata; cannot score sleep stages.")
    if not np.isclose(selected.sample_rate, target_hz, rtol=0.0, atol=1e-3):
        selected = apply_resample(selected, target_hz)
    return selected


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
    )
    return EdgeEyeMovementResult(
        window=window,
        sequences=sequences,
        primitives=primitives,
    )


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run sleep scoring, usability, and edge eye-movement detection.

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

    sleep_model = OnnxSleepScoringModel.load(config.model_path)
    sleep_ts = _prepare_sleep_scoring_timeseries(
        recording,
        config,
        sleep_model.metadata,
    )
    hypnodensity = score_sleep_stages(
        sleep_ts,
        backend=sleep_model,
        metadata=sleep_model.metadata,
        output="probs_timeseries",
    )
    hypnogram = score_sleep_stages(
        sleep_ts,
        backend=sleep_model,
        metadata=sleep_model.metadata,
        output="labels_epochs",
    )
    assert isinstance(hypnodensity, TimeSeries)
    assert isinstance(hypnogram, Epochs)

    usability_model = load_usability_model(config.usability_model)
    usability_scores, samples_to_keep, epoch_length = get_usability_scores(
        recording,
        usability_model,
        eeg_left=config.eeg_left,
        eeg_right=config.eeg_right,
        movement=config.movement,
    )

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
        hypnodensity=hypnodensity,
        hypnogram=hypnogram,
        usability_scores=usability_scores,
        usability_samples_to_keep=samples_to_keep,
        usability_epoch_length=epoch_length,
        edge_start=edge_start,
        edge_end=edge_end,
    )
