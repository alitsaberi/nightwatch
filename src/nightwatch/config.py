"""Analysis configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

# Alternating L/R sequences of length ≥ 3 (no consecutive same direction).
DEFAULT_EYE_MOVEMENT_PATTERN = r"^(?!.*([LR])\1)[LR]{3,}$"

ZMAX_DEFAULT_EEG_LEFT = "EEG_L"
ZMAX_DEFAULT_EEG_RIGHT = "EEG_R"
ZMAX_DEFAULT_MOVEMENT = "MOVEMENT"


class AnalysisConfig(BaseModel):
    """Settings for a single recording analysis run.

    Each view is independent. Empty channel lists / ``None`` skip that view.
    ``model_path`` is required only when ``sleep_channels`` is non-empty.
    """

    recording_path: Path
    format: Literal["zmax", "edf"] = "zmax"
    model_path: Path | None = None
    edge_minutes: float = Field(default=30.0, gt=0)
    usability_model: Literal["lite", "lite_binary"] = Field(default="lite")
    eye_movement_pattern: str = Field(
        default=DEFAULT_EYE_MOVEMENT_PATTERN,
        min_length=1,
        description="Regex that sequence labels must fully match to count as detections.",
    )
    output_path: Path = Path("report.html")
    raw_channels: list[str] = Field(default_factory=list)
    spectrogram_channels: list[str] = Field(default_factory=list)
    sleep_channels: list[str] = Field(default_factory=list)
    artifact_eeg_channels: list[str] = Field(default_factory=list)
    movement_channel: str | None = None
    eye_left: str | None = None
    eye_right: str | None = None


def apply_zmax_view_defaults(config: AnalysisConfig, channel_names: tuple[str, ...] | list[str]) -> AnalysisConfig:
    """Fill empty ZMax view fields with EEG_L / EEG_R / MOVEMENT when present.

    EDF configs and already-populated fields are left unchanged. Only applied when
    every independent view field is still at its empty default (CLI "no flags" case).
    """
    if config.format != "zmax":
        return config

    available = set(channel_names)
    left = ZMAX_DEFAULT_EEG_LEFT if ZMAX_DEFAULT_EEG_LEFT in available else None
    right = ZMAX_DEFAULT_EEG_RIGHT if ZMAX_DEFAULT_EEG_RIGHT in available else None
    eeg = [name for name in (left, right) if name is not None]
    movement = ZMAX_DEFAULT_MOVEMENT if ZMAX_DEFAULT_MOVEMENT in available else None

    updates: dict[str, object] = {}
    if not config.raw_channels and eeg:
        updates["raw_channels"] = list(eeg)
    if not config.spectrogram_channels and eeg:
        updates["spectrogram_channels"] = list(eeg)
    if not config.sleep_channels and eeg:
        updates["sleep_channels"] = list(eeg)
    if not config.artifact_eeg_channels and eeg:
        updates["artifact_eeg_channels"] = list(eeg)
    if config.movement_channel is None and movement is not None:
        updates["movement_channel"] = movement
    if config.eye_left is None and left is not None:
        updates["eye_left"] = left
    if config.eye_right is None and right is not None:
        updates["eye_right"] = right

    if not updates:
        return config
    return config.model_copy(update=updates)
