"""Tests for HTML report rendering."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure

from somnio.data import Epochs, Event, TimeSeries

from nightwatch.config import AnalysisConfig
from nightwatch.metrics import compute_metrics
from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult
from nightwatch.plots import build_plots
from nightwatch.report import render


def _make_result() -> AnalysisResult:
    step = int(round(1e9 / 256.0))
    base = int(datetime(2021, 6, 15, 22, 0, tzinfo=timezone.utc).timestamp() * 1e9)
    n = 256 * 60
    timestamps = np.arange(n, dtype=np.int64) * step + base
    recording = TimeSeries(
        values=np.zeros((n, 3)),
        timestamps=timestamps,
        channel_names=("EEG_L", "EEG_R", "MOVEMENT"),
        units=("uV", "uV", "1"),
        sample_rate=256.0,
    )
    hypnogram = Epochs(
        labels=np.array(["W", "N2", "R"], dtype=object),
        period_length=30_000_000_000,
        onset=int(timestamps[0]),
    )
    hypnodensity = TimeSeries(
        values=np.ones((3, 5)) / 5.0,
        timestamps=timestamps[:3],
        channel_names=("W", "N1", "N2", "N3", "R"),
        units=("1",) * 5,
        sample_rate=1.0 / 30.0,
    )
    usability = TimeSeries(
        values=np.array([[0, 1], [2, 0]], dtype=np.float64),
        timestamps=timestamps[::512][:2],
        channel_names=("usability_left", "usability_right"),
        units=("1", "1"),
        sample_rate=0.1,
    )
    edge_window = recording[:256]
    edge = EdgeEyeMovementResult(
        window=edge_window,
        sequences=[
            Event(onset=int(edge_window.timestamps[50]), duration=500_000_000, type="eye_movement", label="LR"),
        ],
        primitives=[],
    )
    config = AnalysisConfig(
        recording_path=Path("/tmp/recording"),
        model_path=Path("/tmp/model.onnx"),
    )
    return AnalysisResult(
        config=config,
        recording=recording,
        hypnodensity=hypnodensity,
        hypnogram=hypnogram,
        usability_scores=usability,
        usability_samples_to_keep=n - 512,
        usability_epoch_length=2560,
        edge_start=edge,
        edge_end=edge,
    )


def test_render_produces_self_contained_html() -> None:
    result = _make_result()
    metrics = compute_metrics(result)
    plots = build_plots(result)

    html = render(metrics, plots)

    assert "<!DOCTYPE html>" in html
    assert "/tmp/recording" in html
    assert "Total sleep time (TST)" in html
    assert "data:image/png;base64," in html
    assert "Hypnodensity" in html
    assert "Eye Movements (First Edge Window)" in html


def test_render_accepts_empty_plots() -> None:
    metrics = compute_metrics(_make_result())
    empty_figure = Figure()
    html = render(metrics, {"custom": empty_figure})

    assert "Nightwatch Report" in html
    assert "data:image/png;base64," in html
