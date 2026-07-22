"""Tests for the analysis pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from somnio.data import Epochs, Event, TimeSeries

from nightwatch.config import AnalysisConfig
from nightwatch.load import LoadedRecording
from nightwatch.pipeline import (
    UNUSABLE_LABEL,
    AnalysisResult,
    EdgeEyeMovementResult,
    _detect_edge_eye_movements,
    _prepare_sleep_scoring_channel,
    available_eeg_channels,
    edge_window_sample_count,
    hypnogram_from_hypnodensity,
    mark_unusable_hypnogram_epochs,
    run_analysis,
    slice_edge_windows,
    soft_fuse_hypnodensities,
)


def _make_recording(
    *,
    n: int = 256 * 60 * 10,
    sample_rate: float = 256.0,
) -> TimeSeries:
    step = int(round(1e9 / sample_rate))
    base = int(datetime(2021, 6, 15, tzinfo=timezone.utc).timestamp() * 1e9)
    timestamps = np.arange(n, dtype=np.int64) * step + base
    channel_names = ("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z", "MOVEMENT")
    values = np.random.default_rng(0).random((n, len(channel_names)))
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=channel_names,
        units=tuple("uV" if name.startswith("EEG") else "1" for name in channel_names),
        sample_rate=sample_rate,
    )


def test_edge_window_sample_count_caps_to_recording_length() -> None:
    short = _make_recording(n=256 * 60)
    assert edge_window_sample_count(short, 30.0) == 256 * 60

    long = _make_recording(n=256 * 60 * 15)
    assert edge_window_sample_count(long, 10.0) == 256 * 60 * 10


def test_slice_edge_windows_returns_first_and_last_segments() -> None:
    ts = _make_recording(n=256 * 60 * 15)
    start, end = slice_edge_windows(ts, 10.0)
    expected = 256 * 60 * 10

    assert start.n_samples == expected
    assert end.n_samples == expected
    np.testing.assert_array_equal(start.timestamps, ts.timestamps[:expected])
    np.testing.assert_array_equal(end.timestamps, ts.timestamps[-expected:])


def test_slice_edge_windows_overlaps_when_recording_shorter_than_two_windows() -> None:
    ts = _make_recording(n=256 * 60 * 15)
    start, end = slice_edge_windows(ts, 10.0)

    overlap = np.intersect1d(start.timestamps, end.timestamps)
    assert overlap.size == 256 * 60 * 5


def test_edge_window_sample_count_raises_without_sample_rate() -> None:
    ts = _make_recording()
    ts = TimeSeries(
        values=ts.values,
        timestamps=ts.timestamps,
        channel_names=ts.channel_names,
        units=ts.units,
        sample_rate=None,
    )
    with pytest.raises(ValueError, match="sample_rate"):
        edge_window_sample_count(ts, 10.0)


@pytest.mark.parametrize("edge_minutes", [0.0, -1.0])
def test_edge_window_sample_count_raises_for_non_positive_minutes(edge_minutes: float) -> None:
    with pytest.raises(ValueError, match="edge_minutes must be positive"):
        edge_window_sample_count(_make_recording(), edge_minutes)


def test_slice_edge_windows_raises_on_empty_recording() -> None:
    ts = _make_recording(n=0)
    with pytest.raises(ValueError, match="no samples"):
        slice_edge_windows(ts, 10.0)


def test_detect_edge_eye_movements_passes_only_eeg_channels() -> None:
    recording = _make_recording()
    config = AnalysisConfig(
        recording_path=Path("recording"),
        model_path=Path("model.onnx"),
    )

    with patch(
        "nightwatch.pipeline.detect_lr_eye_movements",
        return_value=([], []),
    ) as detect_mock:
        result = _detect_edge_eye_movements(
            recording,
            config,
            edge_minutes=10.0,
            at_start=True,
        )

    passed_ts = detect_mock.call_args.args[0]
    assert list(passed_ts.channel_names) == ["EEG_L", "EEG_R"]
    assert detect_mock.call_args.kwargs["accepted_pattern"] == config.eye_movement_pattern
    assert result.window.n_samples == edge_window_sample_count(recording, 10.0)


def test_available_eeg_channels_returns_present_only() -> None:
    recording = _make_recording()
    config = AnalysisConfig(
        recording_path=Path("recording"),
        model_path=Path("model.onnx"),
    )
    assert available_eeg_channels(config, recording) == ["EEG_L", "EEG_R"]

    left_only = recording.select_channels(["EEG_L", "MOVEMENT"])
    assert available_eeg_channels(config, left_only) == ["EEG_L"]


def test_prepare_sleep_scoring_channel_requires_one_channel_model() -> None:
    recording = _make_recording(sample_rate=512.0)
    metadata = MagicMock()
    metadata.n_channels = 2
    metadata.sample_rate_hz = 256.0

    with pytest.raises(ValueError, match="1-channel model"):
        _prepare_sleep_scoring_channel(recording, "EEG_L", metadata)


def test_prepare_sleep_scoring_channel_applies_preprocessing() -> None:
    recording = _make_recording(sample_rate=512.0)
    metadata = MagicMock()
    metadata.n_channels = 1
    metadata.sample_rate_hz = 256.0

    with (
        patch("nightwatch.pipeline.apply_resample", side_effect=lambda ts, hz: ts) as resample_mock,
        patch("nightwatch.pipeline.apply_fir_filter", side_effect=lambda ts, **_: ts) as filter_mock,
        patch("nightwatch.pipeline.apply_scale", side_effect=lambda ts, **_: ts) as scale_mock,
        patch("nightwatch.pipeline.apply_clip_iqr", side_effect=lambda ts, **_: ts) as clip_mock,
    ):
        result = _prepare_sleep_scoring_channel(recording, "EEG_L", metadata)

    assert list(result.channel_names) == ["EEG_L"]
    resample_mock.assert_called_once()
    assert resample_mock.call_args.args[1] == 256.0
    filter_mock.assert_called_once_with(
        resample_mock.call_args.args[0],
        low_cutoff=0.3,
        high_cutoff=35.0,
    )
    scale_mock.assert_called_once_with(filter_mock.call_args.args[0], method="robust")
    clip_mock.assert_called_once_with(scale_mock.call_args.args[0], iqr_factor=20.0)


def test_soft_fuse_hypnodensities_averages_probabilities() -> None:
    timestamps = np.arange(3, dtype=np.int64)
    left = TimeSeries(
        values=np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=np.float64),
        timestamps=timestamps,
        channel_names=("W", "N2"),
        units=("1", "1"),
        sample_rate=1.0 / 30.0,
    )
    right = TimeSeries(
        values=np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]], dtype=np.float64),
        timestamps=timestamps,
        channel_names=("W", "N2"),
        units=("1", "1"),
        sample_rate=1.0 / 30.0,
    )
    fused = soft_fuse_hypnodensities([left, right])
    np.testing.assert_allclose(fused.values, [[0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])


def test_hypnogram_from_hypnodensity_aggregates_to_30s() -> None:
    onset = 1_000_000_000_000
    # Two predictions inside first 30s epoch favoring N2, one in second favoring W.
    timestamps = np.array(
        [onset, onset + 10_000_000_000, onset + 40_000_000_000],
        dtype=np.int64,
    )
    values = np.array(
        [
            [0.1, 0.9],
            [0.2, 0.8],
            [0.7, 0.3],
        ],
        dtype=np.float64,
    )
    hd = TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("W", "N2"),
        units=("1", "1"),
        sample_rate=1.0 / 30.0,
    )
    hypnogram = hypnogram_from_hypnodensity(hd, onset_ns=onset)
    assert list(hypnogram.labels) == ["N2", "W"]
    assert hypnogram.period_length == 30_000_000_000


def test_mark_unusable_hypnogram_epochs_requires_two_good_windows() -> None:
    onset = 0
    hypnogram = Epochs(
        labels=np.array(["N2", "N2"], dtype=object),
        period_length=30_000_000_000,
        onset=onset,
    )
    # Epoch 0: midpoints at 5/15/25s → labels good,bad,good → 2 usable → keep
    # Epoch 1: midpoints at 35/45/55s → bad,bad,good → 1 usable → Unusable
    usability = TimeSeries(
        values=np.array(
            [
                [0, 0],
                [1, 0],
                [0, 0],
                [2, 2],
                [1, 1],
                [0, 0],
            ],
            dtype=np.float64,
        ),
        timestamps=np.array(
            [
                5_000_000_000,
                15_000_000_000,
                25_000_000_000,
                35_000_000_000,
                45_000_000_000,
                55_000_000_000,
            ],
            dtype=np.int64,
        ),
        channel_names=("usability_left", "usability_right"),
        units=("1", "1"),
        sample_rate=0.1,
    )
    masked = mark_unusable_hypnogram_epochs(hypnogram, usability)
    assert list(masked.labels) == ["N2", UNUSABLE_LABEL]


def test_run_analysis_wires_somnio_tasks(tmp_path: Path) -> None:
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"onnx")
    recording_path = tmp_path / "recording"
    recording_path.mkdir()

    config = AnalysisConfig(
        recording_path=recording_path,
        model_path=model_path,
    )
    recording = _make_recording()
    hypnodensity_left = TimeSeries(
        values=np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64),
        timestamps=recording.timestamps[:2],
        channel_names=("W", "N2"),
        units=("1", "1"),
        sample_rate=1.0 / 30.0,
    )
    hypnodensity_right = TimeSeries(
        values=np.array([[0.2, 0.8], [0.9, 0.1]], dtype=np.float64),
        timestamps=recording.timestamps[:2],
        channel_names=("W", "N2"),
        units=("1", "1"),
        sample_rate=1.0 / 30.0,
    )
    usability = TimeSeries(
        values=np.zeros((5, 2), dtype=np.int64),
        timestamps=recording.timestamps[::512][:5],
        channel_names=("usability_left", "usability_right"),
        units=("1", "1"),
        sample_rate=0.1,
    )
    edge_result = EdgeEyeMovementResult(
        window=recording[:256],
        sequences=[Event(onset=0, duration=1_000_000, type="eye_movement", label="L")],
        primitives=[],
    )

    mock_sleep_model = MagicMock()
    mock_sleep_model.metadata.n_channels = 1
    mock_sleep_model.metadata.sample_rate_hz = 256.0

    loaded = LoadedRecording(
        timeseries=recording,
        raw_channel_names=("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z"),
    )

    with (
        patch("nightwatch.pipeline.load_recording", return_value=loaded) as load_mock,
        patch(
            "nightwatch.pipeline.OnnxSleepScoringModel.load",
            return_value=mock_sleep_model,
        ) as sleep_load_mock,
        patch(
            "nightwatch.pipeline.score_sleep_stages",
            side_effect=[hypnodensity_left, hypnodensity_right],
        ) as score_mock,
        patch("nightwatch.pipeline.load_usability_model", return_value=object()) as usability_load_mock,
        patch(
            "nightwatch.pipeline.get_usability_scores",
            return_value=(usability, recording.n_samples, 2560),
        ) as usability_mock,
        patch(
            "nightwatch.pipeline._detect_edge_eye_movements",
            side_effect=[edge_result, edge_result],
        ) as edge_mock,
        patch(
            "nightwatch.pipeline._prepare_sleep_scoring_channel",
            side_effect=lambda ts, channel, metadata: ts.select_channels([channel]),
        ),
    ):
        result = run_analysis(config)

    load_mock.assert_called_once_with(config)
    sleep_load_mock.assert_called_once_with(model_path)
    assert score_mock.call_count == 2
    usability_load_mock.assert_called_once_with("lite")
    usability_mock.assert_called_once()
    assert edge_mock.call_count == 2

    assert isinstance(result, AnalysisResult)
    assert result.recording is recording
    assert result.raw_channel_names == loaded.raw_channel_names
    assert result.eeg_channels == ("EEG_L", "EEG_R")
    np.testing.assert_allclose(result.hypnodensity.values, [[0.5, 0.5], [0.5, 0.5]])
    assert result.usability_scores is usability
    assert result.usability_samples_to_keep == recording.n_samples
    assert result.usability_epoch_length == 2560
    assert result.edge_start is edge_result
    assert result.edge_end is edge_result


def test_run_analysis_missing_model_raises(tmp_path: Path) -> None:
    config = AnalysisConfig(
        recording_path=tmp_path / "recording",
        model_path=tmp_path / "missing.onnx",
    )
    with pytest.raises(FileNotFoundError):
        run_analysis(config)
