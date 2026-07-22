"""Tests for recording loaders."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from somnio.data.timeseries import TimeSeries
from somnio.tasks.eeg_usability.defaults import SAMPLE_RATE_HZ

from nightwatch.config import AnalysisConfig
from nightwatch.load import (
    LoadedRecording,
    append_channel,
    derive_movement,
    load_recording,
    load_zmax_recording,
    zmax_stems_for,
)


def _make_ts(
    n: int = 256,
    *,
    sample_rate: float = 256.0,
    channel_names: tuple[str, ...] = ("ACC_X", "ACC_Y", "ACC_Z"),
    values: np.ndarray | None = None,
) -> TimeSeries:
    step = int(round(1e9 / sample_rate))
    base = int(datetime(2021, 6, 15, tzinfo=timezone.utc).timestamp() * 1e9)
    timestamps = np.arange(n, dtype=np.int64) * step + base
    if values is None:
        values = np.zeros((n, len(channel_names)), dtype=np.float64)
        if len(channel_names) >= 1:
            values[:, 0] = 3.0
        if len(channel_names) >= 2:
            values[:, 1] = 4.0
        if len(channel_names) >= 3:
            values[:, 2] = 0.0
    return TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=channel_names,
        units=tuple("m/s^2" for _ in channel_names),
        sample_rate=sample_rate,
    )


def test_zmax_stems_for_discovers_all_edf_files(tmp_path: Path) -> None:
    root = tmp_path / "rec"
    root.mkdir()
    for stem in ("EEG L", "EEG R", "dX", "dY", "dZ"):
        (root / f"{stem}.edf").touch()
    assert zmax_stems_for(root) == ["EEG L", "EEG R", "dX", "dY", "dZ"]

    (root / "NOISE.edf").touch()
    assert zmax_stems_for(root) == ["EEG L", "EEG R", "NOISE", "dX", "dY", "dZ"]


def test_derive_movement_appends_magnitude() -> None:
    ts = _make_ts()
    out = derive_movement(ts)

    assert out.channel_names[-1] == "MOVEMENT"
    np.testing.assert_allclose(out.values[:, -1], 5.0)


def test_derive_movement_missing_accel_raises() -> None:
    ts = _make_ts(channel_names=("ACC_X", "ACC_Y"))
    with pytest.raises(ValueError, match="Missing accelerometer channels"):
        derive_movement(ts)


def test_append_channel_rejects_duplicate_name() -> None:
    ts = _make_ts(channel_names=("ACC_X",))
    with pytest.raises(ValueError, match="already present"):
        append_channel(ts, "ACC_X", np.zeros(ts.n_samples), "1")


def test_load_zmax_recording_derives_movement_and_resamples(tmp_path: Path) -> None:
    base_ts = _make_ts(
        n=128,
        sample_rate=128.0,
        channel_names=("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z"),
        values=np.random.default_rng(0).random((128, 5)),
    )

    with patch("nightwatch.load.read_zmax_multi", return_value=base_ts) as read_mock:
        out = load_zmax_recording(tmp_path)

    read_mock.assert_called_once()
    assert "MOVEMENT" in out.timeseries.channel_names
    assert out.raw_channel_names == base_ts.channel_names
    assert out.timeseries.sample_rate == SAMPLE_RATE_HZ
    assert out.timeseries.n_samples > base_ts.n_samples


def test_load_zmax_recording_keeps_256_hz_without_resampling(tmp_path: Path) -> None:
    base_ts = _make_ts(
        channel_names=("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z"),
        values=np.random.default_rng(1).random((256, 5)),
    )

    with patch("nightwatch.load.read_zmax_multi", return_value=base_ts):
        out = load_zmax_recording(tmp_path)

    assert out.timeseries.sample_rate == SAMPLE_RATE_HZ
    assert out.timeseries.n_samples == base_ts.n_samples


def test_load_recording_dispatches_zmax(tmp_path: Path) -> None:
    recording = tmp_path / "recording"
    recording.mkdir()
    config = AnalysisConfig(
        recording_path=recording,
        model_path=tmp_path / "model.onnx",
    )
    expected = LoadedRecording(
        timeseries=_make_ts(channel_names=("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z", "MOVEMENT")),
        raw_channel_names=("EEG_L", "EEG_R", "ACC_X", "ACC_Y", "ACC_Z"),
    )

    with patch("nightwatch.load.load_zmax_recording", return_value=expected) as load_mock:
        out = load_recording(config)

    load_mock.assert_called_once_with(recording, movement="MOVEMENT")
    assert out is expected


def test_load_recording_missing_path_raises(tmp_path: Path) -> None:
    config = AnalysisConfig(
        recording_path=tmp_path / "missing",
        model_path=tmp_path / "model.onnx",
    )
    with pytest.raises(FileNotFoundError):
        load_recording(config)


def test_load_zmax_recording_requires_directory(tmp_path: Path) -> None:
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("x")
    with pytest.raises(NotADirectoryError):
        load_zmax_recording(file_path)
