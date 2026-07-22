"""Analysis configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Alternating L/R sequences of length ≥ 3 (no consecutive same direction).
DEFAULT_EYE_MOVEMENT_PATTERN = r"^(?!.*([LR])\1)[LR]{3,}$"


class AnalysisConfig(BaseModel):
    """Settings for a single recording analysis run."""

    recording_path: Path
    format: Literal["zmax"] = "zmax"
    model_path: Path
    edge_minutes: float = Field(default=30.0, gt=0)
    usability_model: Literal["default", "lite", "binary", "lite_binary"] = "lite"
    eye_movement_pattern: str = Field(
        default=DEFAULT_EYE_MOVEMENT_PATTERN,
        min_length=1,
        description="Regex that sequence labels must fully match to count as detections.",
    )
    output_path: Path = Path("report.html")
    eeg_left: str = "EEG_L"
    eeg_right: str = "EEG_R"
    movement: str = "MOVEMENT"
