"""Summary metrics derived from analysis outputs."""

from __future__ import annotations

from typing import Any


def compute_metrics(result: Any) -> dict[str, Any]:
    """Compute report metrics from an analysis result.

    Args:
        result: Completed pipeline output.

    Raises:
        NotImplementedError: Metrics computation is not implemented yet.
    """
    raise NotImplementedError("Metrics computation is not implemented yet.")
