"""Matplotlib figures for reports and Streamlit."""

from __future__ import annotations

from typing import Any

from matplotlib.figure import Figure


def build_plots(result: Any) -> dict[str, Figure]:
    """Build hypnodensity, hypnogram, spectrogram, and auxiliary plots.

    Args:
        result: Completed pipeline output.

    Raises:
        NotImplementedError: Plot generation is not implemented yet.
    """
    raise NotImplementedError("Plot generation is not implemented yet.")
