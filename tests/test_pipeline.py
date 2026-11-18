"""Tests for the analysis pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from somnio.data import Epochs, Event, TimeSeries

from nightwatch.config import AnalysisConfig, apply_zmax_view_defaults, resolve_wu_sleep_model_path
from nightwatch.load import LoadedRecording
from nightwatch.pipeline import (
    UNUSABLE_LABEL,
    AnalysisResult,
    EdgeEyeMovementResult,
    _detect_edge_eye_movements,
    edge_window_sample_count,
    mark_unusable_hypnogram_epochs,
    require_channels,
    run_analysis,
    slice_edge_windows,
)


def _make_recording(
    *,
    n: int = 256 * 60 * 10,
    sample_rate: float = 256.0,
    channel_names: tuple[str, ...] = ("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z", "MOVEMENT"),
) -> TimeSeries:
    step = int(round(1e9 / sample_rate))
    base = int(datetime(2021, 6, 15, tzinfo=timezone.utc).timestamp() * 1e9)
    timestamps = np.arange(n, dtype=np.int64) * step + base
    values = np.random.default_rng(0).random((n, len(channel_names)))
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=channel_names,
        units=tuple("uV" if "EEG" in name or name.startswith("C") else "1" for name in channel_names),
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

    with patch(
        "nightwatch.pipeline.detect_lr_eye_movements",
        return_value=([], []),
    ) as detect_mock:
        result = _detect_edge_eye_movements(
            recording,
            left="EEG_L",
            right="EEG_R",
            edge_minutes=10.0,
            pattern=r"^(?!.*([LR])\1)[LR]{3,}$",
            at_start=True,
        )

    passed_ts = detect_mock.call_args.args[0]
    assert list(passed_ts.channel_names) == ["EEG_L", "EEG_R"]
    assert result.window.n_samples == edge_window_sample_count(recording, 10.0)


def test_require_channels_lists_available() -> None:
    recording = _make_recording()
    with pytest.raises(ValueError, match="Available: EEG_L"):
        require_channels(recording, ["missing_ch"])


def test_apply_zmax_view_defaults_fills_empty() -> None:
    config = AnalysisConfig(recording_path=Path("rec"), format="zmax")
    filled = apply_zmax_view_defaults(
        config,
        ("EEG_L", "EEG_R", "MOVEMENT"),
    )
    assert filled.raw_channels == ["EEG_L", "EEG_R"]
    assert filled.sleep_channels == ["EEG_L", "EEG_R"]
    assert filled.movement_channel == "MOVEMENT"
    assert filled.eye_left == "EEG_L"


def test_apply_zmax_view_defaults_skips_edf() -> None:
    config = AnalysisConfig(recording_path=Path("rec.edf"), format="edf")
    filled = apply_zmax_view_defaults(config, ("C3", "C4"))
    assert filled.raw_channels == []
    assert filled.sleep_channels == []


def test_resolve_wu_sleep_model_path_from_parent_root(tmp_path: Path) -> None:
    root = tmp_path / "model"
    strategy = root / "ft-am"
    strategy.mkdir(parents=True)
    (strategy / "model.yaml").write_text("name: ft-am\n")

    resolved = resolve_wu_sleep_model_path(root, "ft-am")
    assert resolved == strategy.resolve()


def test_resolve_wu_sleep_model_path_from_strategy_dir(tmp_path: Path) -> None:
    strategy = tmp_path / "ft-am"
    strategy.mkdir()
    (strategy / "saved_model.pb").write_bytes(b"pb")

    resolved = resolve_wu_sleep_model_path(strategy, "ft-noam")
    assert resolved == strategy.resolve()


def test_mark_unusable_hypnogram_epochs_requires_two_good_windows() -> None:
    onset = 0
    hypnogram = Epochs(
        labels=np.array(["N2", "N2"], dtype=object),
        period_length=30_000_000_000,
        onset=onset,
    )
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
        channel_names=("EEG_L", "EEG_R"),
        units=("1", "1"),
        sample_rate=0.1,
    )
    masked = mark_unusable_hypnogram_epochs(hypnogram, usability)
    assert list(masked.labels) == ["N2", UNUSABLE_LABEL]


def test_run_analysis_wires_wu_sleep(tmp_path: Path) -> None:
    model_root = tmp_path / "model"
    strategy = model_root / "ft-am"
    strategy.mkdir(parents=True)
    (strategy / "model.yaml").write_text("name: ft-am\n")
    recording_path = tmp_path / "recording"
    recording_path.mkdir()

    config = AnalysisConfig(
        recording_path=recording_path,
        model_path=model_root,
        sleep_model="ft-am",
        format="zmax",
    )
    recording = _make_recording()
    onset_ns = int(recording.timestamps[0])
    probs = np.array(
        [
            [0.5, 0.1, 0.2, 0.1, 0.1],
            [0.1, 0.1, 0.6, 0.1, 0.1],
        ],
        dtype=np.float64,
    )
    # Two fully usable 10 s windows in each 30 s sleep epoch.
    usability = TimeSeries(
        values=np.zeros((4, 1), dtype=np.int64),
        timestamps=np.array(
            [
                onset_ns + 5_000_000_000,
                onset_ns + 15_000_000_000,
                onset_ns + 35_000_000_000,
                onset_ns + 45_000_000_000,
            ],
            dtype=np.int64,
        ),
        channel_names=("EEG_L",),
        units=("1",),
        sample_rate=0.1,
    )
    edge_result = EdgeEyeMovementResult(
        window=recording[:256],
        sequences=[Event(onset=0, duration=1_000_000, type="eye_movement", label="L")],
        primitives=[],
    )

    loaded = LoadedRecording(
        timeseries=recording,
        raw_channel_names=("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z"),
    )

    with (
        patch("nightwatch.pipeline.load_recording", return_value=loaded) as load_mock,
        patch(
            "nightwatch.pipeline.score_sleep_stages",
            return_value=probs,
        ) as score_mock,
        patch(
            "nightwatch.pipeline.labels_from_probabilities",
            return_value=np.array(["W", "N2"], dtype=object),
        ) as labels_mock,
        patch("nightwatch.pipeline.load_usability_model", return_value=object()),
        patch(
            "nightwatch.pipeline.get_usability_score",
            return_value=(usability, recording.n_samples, 2560),
        ) as usability_mock,
        patch(
            "nightwatch.pipeline._detect_edge_eye_movements",
            side_effect=[edge_result, edge_result],
        ) as edge_mock,
    ):
        result = run_analysis(config)

    load_mock.assert_called_once()
    score_mock.assert_called_once()
    assert score_mock.call_args.kwargs["model_path"] == strategy.resolve()
    assert score_mock.call_args.kwargs["channel_names"] == ["EEG_L", "EEG_R"]
    labels_mock.assert_called_once()
    assert usability_mock.call_count == 2
    assert edge_mock.call_count == 2

    assert isinstance(result, AnalysisResult)
    assert result.recording is recording
    assert result.raw_channel_names == loaded.raw_channel_names
    assert result.sleep_channels == ("EEG_L", "EEG_R")
    np.testing.assert_allclose(result.hypnodensity.values, probs)
    assert list(result.hypnogram.labels) == ["W", "N2"]
    assert result.usability_scores is not None
    assert list(result.usability_scores.channel_names) == ["EEG_L", "EEG_R"]
    assert result.edge_start is edge_result
    assert result.edge_end is edge_result


def test_run_analysis_skips_all_views_for_empty_edf_config(tmp_path: Path) -> None:
    edf_path = tmp_path / "rec.edf"
    edf_path.write_bytes(b"edf")
    recording = _make_recording(channel_names=("C3", "C4", "ACC"))
    loaded = LoadedRecording(timeseries=recording, raw_channel_names=("C3", "C4", "ACC"))
    config = AnalysisConfig(recording_path=edf_path, format="edf")

    with patch("nightwatch.pipeline.load_recording", return_value=loaded):
        result = run_analysis(config)

    assert result.hypnodensity is None
    assert result.hypnogram is None
    assert result.usability_scores is None
    assert result.edge_start is None
    assert result.edge_end is None
    assert result.sleep_channels == ()


def test_run_analysis_sleep_requires_resolvable_model(tmp_path: Path) -> None:
    edf_path = tmp_path / "rec.edf"
    edf_path.write_bytes(b"edf")
    recording = _make_recording(channel_names=("C3", "C4"))
    loaded = LoadedRecording(timeseries=recording, raw_channel_names=("C3", "C4"))
    config = AnalysisConfig(
        recording_path=edf_path,
        format="edf",
        sleep_channels=["C3"],
        model_path=None,
    )

    with (
        patch("nightwatch.pipeline.load_recording", return_value=loaded),
        patch("nightwatch.config.discover_wu_sleep_model_root", return_value=None),
    ):
        with pytest.raises(ValueError, match="model_path"):
            run_analysis(config)


def test_run_analysis_missing_model_raises(tmp_path: Path) -> None:
    recording_path = tmp_path / "recording"
    recording_path.mkdir()
    recording = _make_recording()
    loaded = LoadedRecording(
        timeseries=recording,
        raw_channel_names=("EEG_L", "EEG_R"),
    )
    config = AnalysisConfig(
        recording_path=recording_path,
        model_path=tmp_path / "missing-model",
        format="zmax",
        sleep_channels=["EEG_L"],
        raw_channels=[],
        spectrogram_channels=[],
        artifact_eeg_channels=[],
        movement_channel=None,
        eye_left=None,
        eye_right=None,
    )
    with patch("nightwatch.pipeline.load_recording", return_value=loaded):
        with pytest.raises(FileNotFoundError):
            run_analysis(config)
