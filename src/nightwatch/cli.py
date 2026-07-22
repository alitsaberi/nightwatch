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


@app.command("run")
def run(
    recording: Path = typer.Argument(..., help="Path to the recording (ZMax directory)."),
    format: Literal["zmax"] = typer.Option("zmax", help="Input recording format."),
    model: Path = typer.Option(..., help="Path to sleep-scoring ONNX model."),
    edge_minutes: float = typer.Option(30.0, help="Minutes at start/end for eye-movement detection."),
    usability_model: Literal["default", "lite", "binary", "lite_binary"] = typer.Option(
        "default",
        help="EEG usability model variant.",
    ),
    eye_movement_pattern: str = typer.Option(
        r"^(?!.*([LR])\1)[LR]{3,}$",
        help="Regex that eye-movement sequence labels must fully match.",
    ),
    output: Path = typer.Option(Path("report.html"), help="Output HTML report path."),
) -> None:
    """Analyze a recording and write an HTML report."""
    import matplotlib

    matplotlib.use("Agg")

    from nightwatch.config import AnalysisConfig
    from nightwatch.metrics import compute_metrics
    from nightwatch.pipeline import run_analysis
    from nightwatch.plots import build_plots
    from nightwatch.report import render

    if not recording.exists():
        _fail(f"Recording path does not exist: {recording}")
    if not model.is_file():
        _fail(f"Sleep-scoring model not found: {model}")

    config = AnalysisConfig(
        recording_path=recording,
        format=format,
        model_path=model,
        edge_minutes=edge_minutes,
        usability_model=usability_model,
        eye_movement_pattern=eye_movement_pattern,
        output_path=output,
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
