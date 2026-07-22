"""Recording loaders (format dispatch)."""

from __future__ import annotations

from somnio.data import TimeSeries

from nightwatch.config import AnalysisConfig


def load_recording(config: AnalysisConfig) -> TimeSeries:
    """Load a recording and return a somnio TimeSeries.

    Args:
        config: Analysis settings including path and format.

    Raises:
        NotImplementedError: Loaders are not implemented yet.
    """
    raise NotImplementedError("Recording loaders are not implemented yet.")
