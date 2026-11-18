"""Analysis configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, Field

# Alternating L/R sequences of length ≥ 3 (no consecutive same direction).
DEFAULT_EYE_MOVEMENT_PATTERN = r"^(?!.*([LR])\1)[LR]{3,}$"

ZMAX_DEFAULT_EEG_LEFT = "EEG_L"
ZMAX_DEFAULT_EEG_RIGHT = "EEG_R"
ZMAX_DEFAULT_MOVEMENT = "MOVEMENT"

WU_SLEEP_MODELS = ("ft-am", "ft-noam", "tfs-am", "dt")
WuSleepModel = Literal["ft-am", "ft-noam", "tfs-am", "dt"]
DEFAULT_WU_SLEEP_MODEL: WuSleepModel = "ft-am"
WU_SLEEP_CLASS_LABELS = ("W", "N1", "N2", "N3", "REM")


def discover_wu_sleep_model_root() -> Path | None:
    """Return a wu-sleep ``model/`` directory when discoverable from the install."""
    try:
        import wu_sleep
    except ImportError:
        return None

    package_dir = Path(wu_sleep.__file__).resolve().parent
    for candidate in (package_dir.parent.parent / "model", package_dir.parent / "model"):
        if (candidate / DEFAULT_WU_SLEEP_MODEL).is_dir():
            return candidate
    return None


def resolve_wu_sleep_model_path(
    model_path: Path | None,
    sleep_model: WuSleepModel | str = DEFAULT_WU_SLEEP_MODEL,
) -> Path:
    """Resolve a WU-Sleep SavedModel directory from a root or direct path.

    ``model_path`` may be:

    - a strategy directory (contains ``model.yaml`` / ``saved_model.pb``), or
    - a parent ``model/`` directory containing ``ft-am`` / ``ft-noam`` / …
    - omitted, in which case a discoverable wu-sleep ``model/`` root is used
    """
    strategy = str(sleep_model)
    if strategy not in get_args(WuSleepModel):
        allowed = ", ".join(get_args(WuSleepModel))
        raise ValueError(f"sleep_model must be one of {allowed}, got {sleep_model!r}")

    root = model_path
    if root is None:
        root = discover_wu_sleep_model_root()
    if root is None:
        raise ValueError(
            "Sleep scoring requires --model / model_path pointing to a wu-sleep "
            f"model directory (or parent of {strategy}/), or an editable wu-sleep "
            "install with a local model/ tree."
        )

    root = root.expanduser().resolve()
    strategy_dir = root / strategy
    if strategy_dir.is_dir():
        return strategy_dir

    if root.is_dir() and (
        (root / "model.yaml").is_file()
        or (root / "saved_model.pb").is_file()
        or (root / "variables").is_dir()
    ):
        return root

    raise FileNotFoundError(
        f"WU-Sleep model not found at {root} or {strategy_dir}. "
        f"Pass a SavedModel directory or a parent containing {strategy}/."
    )


class AnalysisConfig(BaseModel):
    """Settings for a single recording analysis run.

    Each view is independent. Empty channel lists / ``None`` skip that view.
    Sleep scoring uses [wu-sleep](https://github.com/alitsaberi/wu-sleep); provide
    ``model_path`` (SavedModel dir or parent ``model/`` root) unless an editable
    wu-sleep install exposes its ``model/`` tree.
    """

    recording_path: Path
    format: Literal["zmax", "edf"] = "zmax"
    model_path: Path | None = None
    sleep_model: WuSleepModel = DEFAULT_WU_SLEEP_MODEL
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
