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
    "sleep_scoring": "Sleep scoring",
    "eye_movements": "Eye movements",
}

PLOT_ORDER: tuple[str, ...] = (
    "sleep_scoring",
    "eye_movements",
)

CHANNEL_TITLE_BY_SUFFIX: dict[str, str] = {
    "EEG_L": "EEG Left",
    "EEG_R": "EEG Right",
}


def plot_display_order(plots: dict[str, Figure]) -> list[str]:
    """Return plot keys: EEG Left/Right blocks first, then sleep scoring and edges."""
    preferred_channels = ("channel_EEG_L", "channel_EEG_R")
    channel_keys = [key for key in preferred_channels if key in plots]
    channel_keys.extend(
        sorted(key for key in plots if key.startswith("channel_") and key not in channel_keys)
    )
    ordered = [key for key in PLOT_ORDER if key in plots]
    extras = [key for key in plots if key not in channel_keys and key not in ordered]
    return channel_keys + ordered + extras


def plot_title(key: str) -> str:
    """Human-readable title for a plot key."""
    if key in PLOT_TITLES:
        return PLOT_TITLES[key]
    if key.startswith("channel_"):
        suffix = key.removeprefix("channel_")
        return CHANNEL_TITLE_BY_SUFFIX.get(suffix, suffix.replace("_", " "))
    return key.replace("_", " ").title()


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
