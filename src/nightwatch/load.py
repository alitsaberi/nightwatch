"""Recording loaders (format dispatch)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np

from somnio.data.timeseries import TimeSeries
from somnio.data.units import UV, Dimension, convert_values
from somnio.io.edf import read_zmax_multi
from somnio.tasks.eeg_usability.defaults import SAMPLE_RATE_HZ
from somnio.transforms.resample import apply_resample

from nightwatch.config import AnalysisConfig

ZMAX_STEM_ALIASES: Final[dict[str, str]] = {
    "EEG L": "EEG_L",
    "EEG R": "EEG_R",
    "dX": "ACC_X",
    "dY": "ACC_Y",
    "dZ": "ACC_Z",
}
ZMAX_ACC_CHANNELS: Final[tuple[str, str, str]] = ("ACC_X", "ACC_Y", "ACC_Z")


@dataclass(frozen=True)
class LoadedRecording:
    """A loaded recording and the channel names read from disk."""

    timeseries: TimeSeries
    raw_channel_names: tuple[str, ...]


def zmax_stems_for(path: Path) -> list[str]:
    """Return sorted stems of every ``*.edf`` file in a ZMax recording directory."""
    return sorted(p.stem for p in path.glob("*.edf"))


def append_channel(
    ts: TimeSeries,
    name: str,
    values: np.ndarray,
    unit: str,
) -> TimeSeries:
    """Return a copy of ``ts`` with one extra channel column."""
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (ts.n_samples,):
        raise ValueError(
            f"values must have shape ({ts.n_samples},), got {values.shape}"
        )
    if name in ts.channel_names:
        raise ValueError(f"Channel {name!r} already present")
    return TimeSeries(
        values=np.column_stack([ts.values, values]),
        timestamps=ts.timestamps.copy(),
        channel_names=(*ts.channel_names, name),
        units=(*ts.units, unit),
        sample_rate=ts.sample_rate,
    )


def derive_movement(
    ts: TimeSeries,
    *,
    acc_x: str = ZMAX_ACC_CHANNELS[0],
    acc_y: str = ZMAX_ACC_CHANNELS[1],
    acc_z: str = ZMAX_ACC_CHANNELS[2],
    movement: str = "MOVEMENT",
) -> TimeSeries:
    """Derive accelerometer magnitude and append it as a movement channel."""
    missing = [ch for ch in (acc_x, acc_y, acc_z) if ch not in ts.channel_index_map]
    if missing:
        raise ValueError(f"Missing accelerometer channels: {missing}")

    idx_x = ts.channel_index_map[acc_x]
    idx_y = ts.channel_index_map[acc_y]
    idx_z = ts.channel_index_map[acc_z]
    magnitude = np.sqrt(
        ts.values[:, idx_x] ** 2
        + ts.values[:, idx_y] ** 2
        + ts.values[:, idx_z] ** 2
    )
    return append_channel(ts, movement, magnitude, "1")


def convert_voltage_channels_to_microvolts(ts: TimeSeries) -> TimeSeries:
    """Convert voltage channels from volts to microvolts (µV)."""
    values = ts.values.copy()
    new_units: list[str | object] = list(ts.units)
    changed = False

    for channel_index, unit in enumerate(ts.units):
        if unit.dimension is not Dimension.VOLTAGE:
            continue
        values[:, channel_index] = convert_values(values[:, channel_index], unit, UV)
        new_units[channel_index] = UV
        changed = True

    if not changed:
        return ts

    return TimeSeries(
        values=values,
        timestamps=ts.timestamps.copy(),
        channel_names=ts.channel_names,
        units=new_units,
        sample_rate=ts.sample_rate,
    )


def _ensure_usability_sample_rate(ts: TimeSeries) -> TimeSeries:
    """Resample to 256 Hz when needed for EEG usability scoring."""
    if ts.sample_rate is None:
        raise ValueError(
            "Recording has no sample_rate metadata; cannot resample for usability."
        )
    if np.isclose(ts.sample_rate, SAMPLE_RATE_HZ, rtol=0.0, atol=1e-6):
        return ts
    return apply_resample(ts, SAMPLE_RATE_HZ)


def load_zmax_recording(
    path: Path | str,
    *,
    movement: str = "MOVEMENT",
) -> LoadedRecording:
    """Load a ZMax multi-EDF directory with aliases and derived movement."""
    root = Path(path)
    if not root.is_dir():
        raise NotADirectoryError(root)

    ts = read_zmax_multi(
        root,
        stems=zmax_stems_for(root),
        stem_aliases=ZMAX_STEM_ALIASES,
        verbose="ERROR",
    )
    raw_channel_names = tuple(ts.channel_names)
    ts = convert_voltage_channels_to_microvolts(ts)
    ts = derive_movement(ts, movement=movement)
    return LoadedRecording(
        timeseries=_ensure_usability_sample_rate(ts),
        raw_channel_names=raw_channel_names,
    )


def load_recording(config: AnalysisConfig) -> LoadedRecording:
    """Load a recording and return a somnio TimeSeries."""
    path = config.recording_path
    if not path.exists():
        raise FileNotFoundError(path)

    if config.format == "zmax":
        return load_zmax_recording(path, movement=config.movement)

    raise ValueError(f"Unsupported recording format: {config.format!r}")
