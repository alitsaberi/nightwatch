"""Analysis configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


class AnalysisConfig(BaseModel):
    """Settings for a single recording analysis run."""

    recording_path: Path
    format: Literal["zmax"] = "zmax"
    model_path: Path
    edge_minutes: float = Field(default=30.0, gt=0)
    usability_model: Literal["default", "lite", "binary", "lite_binary"] = "default"
    output_path: Path = Path("report.html")
    eeg_left: str = "EEG_L"
    eeg_right: str = "EEG_R"
    movement: str = "MOVEMENT"
    spectrogram_channel: str = "EEG_L"
