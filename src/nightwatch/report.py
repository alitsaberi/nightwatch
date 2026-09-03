"""Jinja2 HTML report rendering."""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from jinja2 import Environment, FileSystemLoader, select_autoescape
from matplotlib.figure import Figure

PLOT_TITLES: dict[str, str] = {
    "raw_traces": "Raw traces",
    "artifacts": "Artifacts",
    "sleep_scoring": "Sleep scoring",
    "eye_movements": "Eye movements",
}

PLOT_ORDER: tuple[str, ...] = (
    "raw_traces",
    "artifacts",
    "sleep_scoring",
    "eye_movements",
)


def plot_display_order(plots: dict[str, Figure]) -> list[str]:
    """Return plot keys: fixed views first, then spectrograms, then extras."""
    ordered = [key for key in PLOT_ORDER if key in plots]
    spectrogram_keys = sorted(key for key in plots if key.startswith("spectrogram_"))
    extras = [
        key
        for key in plots
        if key not in ordered and key not in spectrogram_keys
    ]
    # Insert spectrograms after raw traces when present.
    if "raw_traces" in ordered:
        idx = ordered.index("raw_traces") + 1
        ordered = ordered[:idx] + spectrogram_keys + ordered[idx:]
    else:
        ordered = spectrogram_keys + ordered
    return ordered + extras


def plot_title(key: str) -> str:
    """Human-readable title for a plot key."""
    if key in PLOT_TITLES:
        return PLOT_TITLES[key]
    if key.startswith("spectrogram_"):
        channel = key.removeprefix("spectrogram_")
        return f"Spectrogram — {channel}"
    return key.replace("_", " ").title()


def _default_template_dir() -> Path:
    return Path(__file__).resolve().parent / "templates"


def _figure_to_png_base64(fig: Figure) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _format_float(value: object, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}"


def _format_pct(value: object, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{float(value):.{decimals}f}%"


def _encode_plots(plots: dict[str, Figure]) -> list[dict[str, str]]:
    encoded: list[dict[str, str]] = []
    for key in plot_display_order(plots):
        fig = plots[key]
        encoded.append(
            {
                "key": key,
                "title": plot_title(key),
                "png_base64": _figure_to_png_base64(fig),
            }
        )
    return encoded


def render(
    metrics: dict[str, Any],
    plots: dict[str, Figure],
    *,
    template_dir: Path | None = None,
) -> str:
    """Render a self-contained HTML report.

    Args:
        metrics: Summary tables and scalar values from ``compute_metrics``.
        plots: Named matplotlib figures to embed as PNG.
        template_dir: Optional override for the Jinja2 template directory.

    Returns:
        Self-contained HTML document string.
    """
    resolved_template_dir = template_dir or _default_template_dir()
    env = Environment(
        loader=FileSystemLoader(resolved_template_dir),
        autoescape=select_autoescape(["html", "xml"]),
    )
    env.filters["fmt"] = _format_float
    env.filters["pct"] = _format_pct

    recording = metrics.get("recording", {})
    context = {
        "title": "Recording analysis",
        "recording_path": recording.get("path", ""),
        "metrics": metrics,
        "plots": _encode_plots(plots),
    }
    template = env.get_template("report.html")
    return template.render(**context)
