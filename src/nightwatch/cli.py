"""Typer CLI entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from nightwatch import __version__

app = typer.Typer(
    name="nightwatch",
    help="Sleep recording QC and review powered by somnio.",
    no_args_is_help=True,
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"nightwatch {__version__}")
        raise typer.Exit()


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _parse_channel_list(value: str | None) -> list[str]:
    if value is None or not value.strip():
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    """Nightwatch CLI."""


@app.command("inspect")
def inspect_recording(
    recording: Path = typer.Argument(..., help="ZMax directory or EDF file."),
    format: Literal["zmax", "edf"] = typer.Option("zmax", help="Input recording format."),
) -> None:
    """List channel names for one recording."""
    from nightwatch.config import AnalysisConfig
    from nightwatch.load import load_recording

    if not recording.exists():
        _fail(f"Recording path does not exist: {recording}")

    config = AnalysisConfig(recording_path=recording, format=format)
    try:
        loaded = load_recording(config)
    except (FileNotFoundError, NotADirectoryError, IsADirectoryError, ValueError) as exc:
        _fail(str(exc))

    typer.echo(f"Format: {format}")
    typer.echo(f"Path: {recording}")
    typer.echo(f"Sample rate: {loaded.timeseries.sample_rate} Hz")
    typer.echo(f"Samples: {loaded.timeseries.n_samples}")
    typer.echo("Channels:")
    for name in loaded.timeseries.channel_names:
        typer.echo(f"  {name}")


@app.command("run")
def run(
    recording: Path = typer.Argument(..., help="Path to the recording (ZMax directory or EDF file)."),
    format: Literal["zmax", "edf"] = typer.Option("zmax", help="Input recording format."),
    model: Path | None = typer.Option(None, help="Path to sleep-scoring ONNX model."),
    edge_minutes: float = typer.Option(30.0, help="Minutes at start/end for eye-movement detection."),
    usability_model: Literal["lite", "lite_binary"] = typer.Option(
        "lite",
        help="EEG usability model variant.",
    ),
    eye_movement_pattern: str = typer.Option(
        r"^(?!.*([LR])\1)[LR]{3,}$",
        help="Regex that eye-movement sequence labels must fully match.",
    ),
    raw_channels: str | None = typer.Option(
        None,
        help="Comma-separated channels for raw traces.",
    ),
    spectrogram_channels: str | None = typer.Option(
        None,
        help="Comma-separated channels for spectrograms.",
    ),
    sleep_channels: str | None = typer.Option(
        None,
        help="Comma-separated channels for sleep scoring.",
    ),
    artifact_eeg: str | None = typer.Option(
        None,
        help="Comma-separated EEG channels for artifact detection.",
    ),
    movement: str | None = typer.Option(None, help="Movement channel for artifact detection."),
    eye_left: str | None = typer.Option(None, help="Left channel for eye-movement detection."),
    eye_right: str | None = typer.Option(None, help="Right channel for eye-movement detection."),
    output: Path = typer.Option(Path("report.html"), help="Output HTML report path."),
) -> None:
    """Analyze a recording and write an HTML report.

    ZMax with no view flags enables the full default suite. EDF with no view
    flags writes a recording-info-only report.
    """
    import matplotlib

    matplotlib.use("Agg")

    from nightwatch.config import AnalysisConfig
    from nightwatch.metrics import compute_metrics
    from nightwatch.pipeline import run_analysis
    from nightwatch.plots import build_plots
    from nightwatch.report import render

    if not recording.exists():
        _fail(f"Recording path does not exist: {recording}")
    if model is not None and not model.is_file():
        _fail(f"Sleep-scoring model not found: {model}")

    config = AnalysisConfig(
        recording_path=recording,
        format=format,
        model_path=model,
        edge_minutes=edge_minutes,
        usability_model=usability_model,
        eye_movement_pattern=eye_movement_pattern,
        output_path=output,
        raw_channels=_parse_channel_list(raw_channels),
        spectrogram_channels=_parse_channel_list(spectrogram_channels),
        sleep_channels=_parse_channel_list(sleep_channels),
        artifact_eeg_channels=_parse_channel_list(artifact_eeg),
        movement_channel=movement,
        eye_left=eye_left,
        eye_right=eye_right,
    )

    try:
        result = run_analysis(config)
    except FileNotFoundError as exc:
        _fail(str(exc))
    except ValueError as exc:
        _fail(str(exc))

    metrics = compute_metrics(result)
    plots = build_plots(result)
    html = render(metrics, plots)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")
    typer.echo(f"Report written to {output}")
