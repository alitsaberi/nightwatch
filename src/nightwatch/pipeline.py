"""End-to-end analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from nightwatch.config import AnalysisConfig


@dataclass
class AnalysisResult:
    """Outputs from a completed analysis run."""

    config: AnalysisConfig


def run_analysis(config: AnalysisConfig) -> AnalysisResult:
    """Run sleep scoring, usability, and edge eye-movement detection.

    Args:
        config: Analysis settings.

    Raises:
        NotImplementedError: Pipeline wiring is not implemented yet.
    """
    raise NotImplementedError("Analysis pipeline is not implemented yet.")
