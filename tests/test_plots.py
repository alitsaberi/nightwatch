"""Tests for matplotlib plot builders."""

from __future__ import annotations

from datetime import datetime, timezone

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest
from matplotlib.figure import Figure

from somnio.data import Epochs, Event, TimeSeries

from nightwatch.config import AnalysisConfig
from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult
from nightwatch.plots import build_plots, plot_channel_overview, plot_hypnodensity, plot_hypnogram


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
        raw_channel_names=("EEG_L", "EEG_R"),
        eeg_channels=("EEG_L", "EEG_R"),
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
    fig = plot_hypnodensity(result.hypnodensity, usability_scores=result.usability_scores)
    assert isinstance(fig, Figure)
    assert fig.axes
    plt_close(fig)


def test_plot_hypnogram_returns_figure() -> None:
    result = _make_result()
    fig = plot_hypnogram(result.hypnogram)
    assert isinstance(fig, Figure)
    assert fig.axes[0].get_title() == "Hypnogram"
    plt_close(fig)


def test_plot_channel_overview_returns_figure() -> None:
    result = _make_result()
    fig = plot_channel_overview(
        result.recording,
        "EEG_L",
        result.usability_scores,
        0,
        "default",
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 3
    plt_close(fig)


def test_build_plots_returns_named_figures() -> None:
    plots = build_plots(_make_result())
    assert set(plots) == {
        "channel_EEG_L",
        "channel_EEG_R",
        "sleep_scoring",
        "eye_movements",
    }
    # No matched sequences in fixture → overview panels only.
    assert len(plots["eye_movements"].axes) == 2
    for fig in plots.values():
        assert isinstance(fig, Figure)
        assert fig._suptitle is None
        plt_close(fig)


def test_eye_movement_plot_includes_zoom_panels_for_matches() -> None:
    result = _make_result()
    window = result.recording[:512]
    result.edge_start = EdgeEyeMovementResult(
        window=window,
        sequences=[
            Event(
                onset=int(window.timestamps[100]),
                duration=300_000_000,
                type="eye_movement",
                label="LRL",
            ),
            Event(
                onset=int(window.timestamps[300]),
                duration=400_000_000,
                type="eye_movement",
                label="RLR",
            ),
        ],
        primitives=[],
    )
    result.edge_end = EdgeEyeMovementResult(window=window, sequences=[], primitives=[])

    fig = build_plots(result)["eye_movements"]
    visible = [ax for ax in fig.axes if ax.get_visible()]
    titles = [ax.get_title() for ax in visible]
    assert any(t.startswith("First edge window") for t in titles)
    assert any(t.startswith("Last edge window") for t in titles)
    assert any("LRL" in t for t in titles)
    assert any("RLR" in t for t in titles)
    # Overview + two zoom cards + last overview.
    assert len(visible) == 4
    plt_close(fig)


def test_full_recording_plots_share_time_axis() -> None:
    result = _make_result()
    duration_h = result.recording.n_samples / float(result.recording.sample_rate) / 3600.0
    plots = build_plots(result)

    full_recording_keys = ("channel_EEG_L", "channel_EEG_R", "sleep_scoring")
    for key in full_recording_keys:
        for ax in plots[key].axes:
            xlim = ax.get_xlim()
            assert xlim[0] == 0.0
            assert xlim[1] == pytest.approx(duration_h)

    sleep_fig = plots["sleep_scoring"]
    assert len(sleep_fig.axes) == 2
    assert sleep_fig.axes[0].get_xlim() == sleep_fig.axes[1].get_xlim()

    for fig in plots.values():
        plt_close(fig)


def plt_close(fig: Figure) -> None:
    import matplotlib.pyplot as plt

    plt.close(fig)
