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
    "hypnodensity": "Hypnodensity",
    "hypnogram": "Hypnogram",
    "spectrogram": "Spectrogram",
    "usability": "EEG Usability",
    "eye_movement_start": "Eye Movements (First Edge Window)",
    "eye_movement_end": "Eye Movements (Last Edge Window)",
}

PLOT_ORDER: tuple[str, ...] = (
    "hypnodensity",
    "hypnogram",
    "spectrogram",
    "usability",
    "eye_movement_start",
    "eye_movement_end",
)


def _default_template_dir() -> Path:
    package_dir = Path(__file__).resolve().parent
    package_templates = package_dir / "templates"
    if package_templates.is_dir():
        return package_templates
    return package_dir.parent.parent / "templates"


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
    for key in PLOT_ORDER:
        fig = plots.get(key)
        if fig is None:
            continue
        encoded.append(
            {
                "key": key,
                "title": PLOT_TITLES.get(key, key.replace("_", " ").title()),
                "png_base64": _figure_to_png_base64(fig),
            }
        )
    for key, fig in plots.items():
        if key in PLOT_ORDER:
            continue
        encoded.append(
            {
                "key": key,
                "title": PLOT_TITLES.get(key, key.replace("_", " ").title()),
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
