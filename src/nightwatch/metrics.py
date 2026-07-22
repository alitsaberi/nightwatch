"""Summary metrics derived from analysis outputs."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from typing import Any

import numpy as np

from nightwatch.pipeline import AnalysisResult, EdgeEyeMovementResult

USABILITY_LABELS_MULTICLASS: dict[int, str] = {
    0: "Good",
    1: "No Data",
    2: "High Noise",
    3: "Spiky Noise",
    4: "M-shaped Noise",
}

USABILITY_LABELS_BINARY: dict[int, str] = {
    0: "Usable",
    1: "Not Usable",
}

WAKE_STAGE_LABELS = frozenset({"W", "Wake", "wake", "0"})
UNUSABLE_STAGE_LABELS = frozenset({"Unusable", "unusable", "UNUSABLE"})


def _ns_to_datetime(ns: int) -> datetime:
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def _format_duration_hms(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"


def _epoch_length_seconds(hypnogram: Any) -> float:
    return hypnogram.period_length / 1e9


def _normalize_stage_label(label: object) -> str:
    if isinstance(label, (np.integer, int)):
        return str(int(label))
    return str(label)


def _is_unusable_label(label: object) -> bool:
    return _normalize_stage_label(label) in UNUSABLE_STAGE_LABELS


def _is_wake_label(label: object) -> bool:
    if _is_unusable_label(label):
        return False
    normalized = _normalize_stage_label(label)
    return normalized in WAKE_STAGE_LABELS or normalized.upper() == "W"


def _is_sleep_label(label: object) -> bool:
    return not _is_wake_label(label) and not _is_unusable_label(label)


def _usability_label_map(model_key: str) -> dict[int, str]:
    if model_key in {"binary", "lite_binary"}:
        return USABILITY_LABELS_BINARY
    return USABILITY_LABELS_MULTICLASS


def _label_percentages(values: np.ndarray, label_map: dict[int, str]) -> dict[str, float]:
    if values.size == 0:
        return {name: 0.0 for name in label_map.values()}

    counts = Counter(int(v) for v in values.ravel())
    total = sum(counts.values())
    return {
        label_map.get(label, f"Unknown ({label})"): 100.0 * count / total
        for label, count in sorted(counts.items())
    }


def _compute_sleep_metrics(hypnogram: Any) -> dict[str, Any]:
    labels = hypnogram.labels
    epoch_s = _epoch_length_seconds(hypnogram)
    n_epochs = len(labels)
    trt_seconds = n_epochs * epoch_s

    stage_minutes: dict[str, float] = {}
    for label in labels:
        key = _normalize_stage_label(label)
        stage_minutes[key] = stage_minutes.get(key, 0.0) + epoch_s / 60.0

    stage_pct = {
        stage: (minutes * 60.0 / trt_seconds * 100.0 if trt_seconds else 0.0)
        for stage, minutes in stage_minutes.items()
    }

    sleep_epoch_count = sum(1 for label in labels if _is_sleep_label(label))
    wake_epoch_count = sum(1 for label in labels if _is_wake_label(label))
    unusable_epoch_count = sum(1 for label in labels if _is_unusable_label(label))
    tst_seconds = sleep_epoch_count * epoch_s

    first_sleep_idx = next(
        (i for i, label in enumerate(labels) if _is_sleep_label(label)),
        None,
    )
    sol_seconds = first_sleep_idx * epoch_s if first_sleep_idx is not None else trt_seconds

    waso_seconds = 0.0
    if first_sleep_idx is not None:
        for label in labels[first_sleep_idx + 1 :]:
            if _is_wake_label(label):
                waso_seconds += epoch_s

    se_pct = (tst_seconds / trt_seconds * 100.0) if trt_seconds else 0.0

    return {
        "tst_minutes": tst_seconds / 60.0,
        "trt_minutes": trt_seconds / 60.0,
        "sleep_efficiency_pct": se_pct,
        "sol_minutes": sol_seconds / 60.0,
        "waso_minutes": waso_seconds / 60.0,
        "wake_epoch_count": wake_epoch_count,
        "sleep_epoch_count": sleep_epoch_count,
        "unusable_epoch_count": unusable_epoch_count,
        "unusable_minutes": unusable_epoch_count * epoch_s / 60.0,
        "epoch_length_seconds": epoch_s,
        "stage_pct": stage_pct,
        "stage_minutes": stage_minutes,
    }


def _compute_artifact_metrics(result: AnalysisResult) -> dict[str, Any]:
    label_map = _usability_label_map(result.config.usability_model)
    left = result.usability_scores.values[:, 0].astype(int)
    right = result.usability_scores.values[:, 1].astype(int)

    usable_left_pct = 100.0 * np.mean(left == 0) if left.size else 0.0
    usable_right_pct = 100.0 * np.mean(right == 0) if right.size else 0.0
    samples_total = result.recording.n_samples
    samples_to_keep = result.usability_samples_to_keep

    return {
        "left": _label_percentages(left, label_map),
        "right": _label_percentages(right, label_map),
        "usable_epoch_pct_left": usable_left_pct,
        "usable_epoch_pct_right": usable_right_pct,
        "samples_to_keep": samples_to_keep,
        "samples_total": samples_total,
        "samples_to_keep_pct": (100.0 * samples_to_keep / samples_total if samples_total else 0.0),
        "epoch_length_seconds": result.usability_epoch_length / (
            result.recording.sample_rate or 1.0
        ),
    }


def _event_label_histogram(events: list[Any]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for event in events:
        label = event.label
        key = str(label) if label is not None else "None"
        counts[key] += 1
    return dict(sorted(counts.items()))


def _compute_edge_eye_movement_metrics(edge: EdgeEyeMovementResult) -> dict[str, Any]:
    duration_seconds = edge.window.duration.total_seconds()
    sequence_count = len(edge.sequences)

    return {
        "duration_seconds": duration_seconds,
        "duration_hms": _format_duration_hms(duration_seconds),
        "has_matches": sequence_count > 0,
        "sequence_count": sequence_count,
        "sequence_label_histogram": _event_label_histogram(edge.sequences),
    }


def compute_metrics(result: AnalysisResult) -> dict[str, Any]:
    """Compute report metrics from an analysis result.

    Args:
        result: Completed pipeline output.

    Returns:
        Nested dict with recording, artifact, sleep, and eye-movement summaries.
    """
    recording = result.recording
    start_ns = int(recording.timestamps[0]) if recording.n_samples else 0
    end_ns = int(recording.timestamps[-1]) if recording.n_samples else start_ns
    duration_seconds = recording.duration.total_seconds()
    edge_start = _compute_edge_eye_movement_metrics(result.edge_start)
    edge_end = _compute_edge_eye_movement_metrics(result.edge_end)

    return {
        "recording": {
            "path": str(result.config.recording_path),
            "format": result.config.format,
            "duration_hms": _format_duration_hms(duration_seconds),
            "duration_seconds": duration_seconds,
            "sample_rate_hz": recording.sample_rate,
            "start": _ns_to_datetime(start_ns).isoformat(),
            "end": _ns_to_datetime(end_ns).isoformat(),
            "channels": list(result.raw_channel_names),
        },
        "artifacts": _compute_artifact_metrics(result),
        "sleep": _compute_sleep_metrics(result.hypnogram),
        "eye_movement": {
            "edge_minutes": result.config.edge_minutes,
            "pattern": result.config.eye_movement_pattern,
            "has_matches": edge_start["has_matches"] or edge_end["has_matches"],
            "start": edge_start,
            "end": edge_end,
        },
    }
