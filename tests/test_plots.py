"""Tests for matplotlib plot builders."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from somnio.data import Epochs, Event, TimeSeries

from nightwatch.config import AnalysisConfig
from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult
from nightwatch.plots import build_plots, plot_hypnodensity, plot_hypnogram, plot_spectrogram


def _make_result() -> AnalysisResult:
    step = int(round(1e9 / 256.0))
    base = int(datetime(2021, 6, 15, 22, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    n = 256 * 60
    timestamps = np.arange(n, dtype=np.int64) * step + base
    rng = np.random.default_rng(0)
    values = rng.standard_normal((n, 2))
    recording = TimeSeries(
        values=values,
        timestamps=timestamps,
        channel_names=("EEG_L", "EEG_R"),
        units=("uV", "uV"),
        sample_rate=256.0,
    )
    hypnodensity = TimeSeries(
        values=rng.random((20, 5)),
        timestamps=timestamps[:20],
        channel_names=("W", "N1", "N2", "N3", "R"),
        units=("1",) * 5,
        sample_rate=1.0 / 30.0,
    )
    hypnogram = Epochs(
        labels=np.array(["W", "N2", "N2", "R"], dtype=object),
        period_length=30_000_000_000,
        onset=int(timestamps[0]),
    )
    usability = TimeSeries(
        values=np.array([[0, 1], [2, 0]], dtype=np.float64),
        timestamps=timestamps[::512][:2],
        channel_names=("usability_left", "usability_right"),
        units=("1", "1"),
        sample_rate=0.1,
    )
    edge = EdgeEyeMovementResult(
        window=recording[:512],
        sequences=[],
        primitives=[],
    )
    config = AnalysisConfig(
        recording_path="/tmp/recording",
        model_path="/tmp/model.onnx",
    )
    return AnalysisResult(
        config=config,
        recording=recording,
        hypnodensity=hypnodensity,
        hypnogram=hypnogram,
        usability_scores=usability,
        usability_samples_to_keep=n - 100,
        usability_epoch_length=2560,
        edge_start=edge,
        edge_end=edge,
    )


def test_plot_hypnodensity_returns_figure() -> None:
    result = _make_result()
    fig = plot_hypnodensity(result.hypnodensity)
    assert isinstance(fig, Figure)
    assert fig.axes
    plt_close(fig)


def test_plot_hypnogram_returns_figure() -> None:
    result = _make_result()
    fig = plot_hypnogram(result.hypnogram)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Hypnogram"
    plt_close(fig)


def test_plot_spectrogram_returns_figure() -> None:
    result = _make_result()
    fig = plot_spectrogram(result.recording, "EEG_L")
    assert isinstance(fig, Figure)
    assert "Spectrogram" in fig.axes[0].get_title()
    plt_close(fig)


def test_build_plots_returns_named_figures() -> None:
    plots = build_plots(_make_result())
    assert set(plots) == {
        "hypnodensity",
        "hypnogram",
        "spectrogram",
        "usability",
        "eye_movement_start",
        "eye_movement_end",
    }
    for fig in plots.values():
        assert isinstance(fig, Figure)
        plt_close(fig)


def plt_close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
