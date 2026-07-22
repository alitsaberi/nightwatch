"""Jinja2 HTML report rendering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from matplotlib.figure import Figure


def render(
    metrics: dict[str, Any],
    plots: dict[str, Figure],
    *,
    template_dir: Path | None = None,
) -> str:
    """Render a self-contained HTML report.

    Args:
        metrics: Summary tables and scalar values.
        plots: Named matplotlib figures to embed as PNG.
        template_dir: Optional override for the Jinja2 template directory.

    Raises:
        NotImplementedError: Report rendering is not implemented yet.
    """
    raise NotImplementedError("Report rendering is not implemented yet.")
