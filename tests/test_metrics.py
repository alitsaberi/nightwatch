"""Tests for metrics aggregation."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest

from somnio.data import Epochs, Event, TimeSeries

from nightwatch.config import AnalysisConfig
from nightwatch.metrics import compute_metrics
from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult


def _make_recording(*, n: int = 256 * 60 * 2, sample_rate: float = 256.0) -> TimeSeries:
    step = int(round(1e9 / sample_rate))
    base = int(datetime(2021, 6, 15, 22, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    timestamps = np.arange(n, dtype=np.int64) * step + base
    channel_names = ("EEG_L", "EEG_R", "MOVEMENT")
    values = np.zeros((n, len(channel_names)))
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=channel_names,
        units=("uV", "uV", "1"),
        sample_rate=sample_rate,
    )


def _make_result() -> AnalysisResult:
    recording = _make_recording()
    hypnogram = Epochs(
        labels=np.array(["W", "N1", "N2", "N3", "R"], dtype=object),
        period_length=30_000_000_000,
        onset=int(recording.timestamps[0]),
    )
    usability = TimeSeries(
        values=np.array([[0, 0], [2, 1], [0, 4], [1, 0]], dtype=np.float64),
        timestamps=recording.timestamps[::512][:4],
        channel_names=("usability_left", "usability_right"),
        units=("1", "1"),
        sample_rate=0.1,
    )
    edge_window = recording[:256]
    edge = EdgeEyeMovementResult(
        window=edge_window,
        sequences=[],
        primitives=[],
    )
    config = AnalysisConfig(
        recording_path="/tmp/recording",
        model_path="/tmp/model.onnx",
    )
    hypnodensity = TimeSeries(
        values=np.ones((5, 5)) / 5.0,
        timestamps=recording.timestamps[:5],
        channel_names=("W", "N1", "N2", "N3", "R"),
        units=("1",) * 5,
        sample_rate=1.0 / 30.0,
    )
    return AnalysisResult(
        config=config,
        recording=recording,
        raw_channel_names=("EEG_L", "EEG_R"),
        eeg_channels=("EEG_L", "EEG_R"),
        hypnodensity=hypnodensity,
        hypnogram=hypnogram,
        usability_scores=usability,
        usability_samples_to_keep=recording.n_samples - 512,
        usability_epoch_length=2560,
        edge_start=edge,
        edge_end=edge,
    )


def test_compute_metrics_recording_summary() -> None:
    result = _make_result()
    metrics = compute_metrics(result)

    assert metrics["recording"]["format"] == "zmax"
    assert metrics["recording"]["sample_rate_hz"] == 256.0
    assert metrics["recording"]["duration_seconds"] == pytest.approx(120.0, rel=1e-3)
    assert metrics["recording"]["duration_hms"] == "0:02:00"
    assert metrics["recording"]["channels"] == ["EEG_L", "EEG_R"]


def test_compute_metrics_sleep_stats() -> None:
    metrics = compute_metrics(_make_result())["sleep"]

    assert metrics["trt_minutes"] == pytest.approx(2.5)
    assert metrics["tst_minutes"] == pytest.approx(2.0)
    assert metrics["sleep_efficiency_pct"] == pytest.approx(80.0)
    assert metrics["sol_minutes"] == pytest.approx(0.5)
    assert metrics["waso_minutes"] == pytest.approx(0.0)
    assert metrics["stage_minutes"]["W"] == pytest.approx(0.5)
    assert metrics["stage_minutes"]["N2"] == pytest.approx(0.5)


def test_compute_metrics_artifact_percentages() -> None:
    metrics = compute_metrics(_make_result())["artifacts"]

    assert metrics["usable_epoch_pct_left"] == pytest.approx(50.0)
    assert metrics["usable_epoch_pct_right"] == pytest.approx(50.0)
    assert metrics["left"]["Good"] == pytest.approx(50.0)
    assert metrics["right"]["M-shaped Noise"] == pytest.approx(25.0)
    assert metrics["samples_to_keep"] == _make_result().recording.n_samples - 512


def test_compute_metrics_eye_movement_counts() -> None:
    metrics = compute_metrics(_make_result())["eye_movement"]

    assert metrics["edge_minutes"] == 30.0
    assert metrics["has_matches"] is False
    assert metrics["start"]["has_matches"] is False
    assert metrics["end"]["has_matches"] is False
    assert metrics["pattern"]


def test_compute_metrics_eye_movement_reports_matching_sequences() -> None:
    result = _make_result()
    edge_window = result.recording[:256]
    matched = EdgeEyeMovementResult(
        window=edge_window,
        sequences=[
            Event(
                onset=int(edge_window.timestamps[50]),
                duration=500_000_000,
                type="eye_movement",
                label="LRL",
            ),
        ],
        primitives=[],
    )
    result.edge_start = matched
    result.edge_end = EdgeEyeMovementResult(window=edge_window, sequences=[], primitives=[])

    metrics = compute_metrics(result)["eye_movement"]
    assert metrics["has_matches"] is True
    assert metrics["start"]["has_matches"] is True
    assert metrics["start"]["sequence_count"] == 1
    assert metrics["start"]["sequence_label_histogram"] == {"LRL": 1}
    assert metrics["end"]["has_matches"] is False


def test_compute_metrics_waso_after_sleep_onset() -> None:
    result = _make_result()
    result.hypnogram = Epochs(
        labels=np.array(["W", "N2", "W", "N3", "R"], dtype=object),
        period_length=30_000_000_000,
        onset=int(result.recording.timestamps[0]),
    )
    sleep = compute_metrics(result)["sleep"]

    assert sleep["sol_minutes"] == pytest.approx(0.5)
    assert sleep["waso_minutes"] == pytest.approx(0.5)
    assert sleep["tst_minutes"] == pytest.approx(1.5)


def test_compute_metrics_excludes_unusable_from_tst() -> None:
    result = _make_result()
    result.hypnogram = Epochs(
        labels=np.array(["W", "N2", "Unusable", "N3", "R"], dtype=object),
        period_length=30_000_000_000,
        onset=int(result.recording.timestamps[0]),
    )
    sleep = compute_metrics(result)["sleep"]

    assert sleep["unusable_minutes"] == pytest.approx(0.5)
    assert sleep["tst_minutes"] == pytest.approx(1.5)
    assert sleep["wake_epoch_count"] == 1
    assert sleep["sleep_epoch_count"] == 3


def test_compute_metrics_binary_usability_labels() -> None:
    result = _make_result()
    result.config = AnalysisConfig(
        recording_path="/tmp/recording",
        model_path="/tmp/model.onnx",
        usability_model="binary",
    )
    result.usability_scores = TimeSeries(
        values=np.array([[0, 1], [1, 0], [0, 0]], dtype=np.float64),
        timestamps=result.recording.timestamps[:3],
        channel_names=("usability_left", "usability_right"),
        units=("1", "1"),
        sample_rate=1.0,
    )
    artifacts = compute_metrics(result)["artifacts"]

    assert artifacts["left"]["Usable"] == pytest.approx(100.0 * 2 / 3)
    assert artifacts["right"]["Not Usable"] == pytest.approx(100.0 / 3)
    assert artifacts["usable_epoch_pct_left"] == pytest.approx(100.0 * 2 / 3)
    assert artifacts["usable_epoch_pct_right"] == pytest.approx(100.0 * 2 / 3)


def test_compute_metrics_empty_edge_events() -> None:
    result = _make_result()
    empty_window = result.recording[:256]
    empty_edge = EdgeEyeMovementResult(window=empty_window, sequences=[], primitives=[])
    result.edge_start = empty_edge
    result.edge_end = empty_edge

    edge = compute_metrics(result)["eye_movement"]["start"]

    assert edge["sequence_count"] == 0
    assert edge["has_matches"] is False
    assert edge["sequence_label_histogram"] == {}
