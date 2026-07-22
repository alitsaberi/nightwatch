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
    output: Path = typer.Option(Path("report.html"), help="Output HTML report path."),
) -> None:
    """Analyze a recording and write an HTML report."""
    from nightwatch.config import AnalysisConfig
    from nightwatch.pipeline import run_analysis
    from nightwatch.report import render

    config = AnalysisConfig(
        recording_path=recording,
        format=format,
        model_path=model,
        edge_minutes=edge_minutes,
        usability_model=usability_model,
        output_path=output,
    )
    result = run_analysis(config)
    html = render({}, {})
    output.write_text(html, encoding="utf-8")
    typer.echo(f"Report written to {output}")
