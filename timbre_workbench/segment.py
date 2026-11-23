"""Frame extraction and deterministic double-threshold speech segmentation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .config import stage_fingerprint
from .util import normalize_relpath, sha256_bytes, stable_float


@dataclass(frozen=True)
class Frame:
    index: int
    start_sample: int
    end_sample: int
    rms_dbfs: float


def _db(value: float, floor: float) -> float:
    if value <= 0.0:
        return floor
    return max(floor, 20.0 * math.log10(value))


def iter_frames(samples: np.ndarray, sample_rate: int, segmentation: dict[str, Any]) -> list[Frame]:
    frame_length = max(1, int(round(sample_rate * float(segmentation["frame_length_ms"]) / 1000.0)))
    frame_shift = max(1, int(round(sample_rate * float(segmentation["frame_shift_ms"]) / 1000.0)))
    floor = float(segmentation.get("energy_floor_dbfs", -70.0))
    frames: list[Frame] = []
    start = 0
    index = 0
    while start < len(samples):
        end = min(len(samples), start + frame_length)
        block = samples[start:end].astype(np.float64) / 32768.0
        if block.size:
            rms = float(np.sqrt(np.mean(np.square(block), dtype=np.float64)))
        else:
            rms = 0.0
        frames.append(Frame(index=index, start_sample=start, end_sample=end, rms_dbfs=stable_float(_db(rms, floor))))
        index += 1
        if end >= len(samples):
            break
        start += frame_shift
    return frames


def detect_speech_segments(
    samples: np.ndarray,
    sample_rate: int,
    segmentation: dict[str, Any],
) -> list[dict[str, int]]:
    frames = iter_frames(samples, sample_rate, segmentation)
    start_threshold = float(segmentation["start_threshold_dbfs"])
    continue_threshold = float(segmentation["continue_threshold_dbfs"])
    raw_intervals: list[tuple[int, int]] = []
    active_start: int | None = None
    last_active_end: int | None = None

    for frame in frames:
        above_start = frame.rms_dbfs >= start_threshold
        above_continue = frame.rms_dbfs >= continue_threshold
        if active_start is None:
            if above_start:
                active_start = frame.start_sample
                last_active_end = frame.end_sample
        else:
            if above_continue:
                last_active_end = frame.end_sample
            else:
                raw_intervals.append((active_start, int(last_active_end or frame.start_sample)))
                active_start = None
                last_active_end = None
    if active_start is not None:
        raw_intervals.append((active_start, int(last_active_end or len(samples))))

    min_segment = int(round(sample_rate * float(segmentation["min_segment_ms"]) / 1000.0))
    min_gap = int(round(sample_rate * float(segmentation["min_gap_ms"]) / 1000.0))

    kept = [(start, end) for start, end in raw_intervals if end - start >= min_segment]
    merged: list[tuple[int, int]] = []
    for start, end in kept:
        if merged and start - merged[-1][1] < min_gap:
            previous_start, _ = merged[-1]
            merged[-1] = (previous_start, end)
        else:
            merged.append((start, end))
    final = [(start, end) for start, end in merged if end - start >= min_segment]
    final.sort(key=lambda interval: (interval[0], interval[1]))
    return [
        {
            "start_sample": int(start),
            "end_sample": int(end),
            "sample_count": int(end - start),
            "ordinal_in_file": index,
        }
        for index, (start, end) in enumerate(final)
    ]


def segmentation_fingerprint(config: dict[str, Any], sample_rate: int) -> str:
    return stage_fingerprint(config, "segmentation", extra={"sample_rate": sample_rate, "algorithm": "hysteresis-v1"})


def segment_content_fp(
    samples: np.ndarray,
    start_sample: int,
    end_sample: int,
    config: dict[str, Any],
    sample_rate: int,
) -> str:
    content = np.asarray(samples[start_sample:end_sample], dtype="<i2").tobytes(order="C")
    seg_fp = stage_fingerprint(
        config,
        "audio",
        "segmentation",
        "features",
        extra={"algorithm": "segment-content-v1", "sample_rate": sample_rate},
    )
    return sha256_bytes(b"segment-content-v1\0" + bytes.fromhex(seg_fp) + content)


def annotation_for_interval(
    labels: dict[str, Any],
    logical_name: str,
    start_sample: int,
    end_sample: int,
) -> str:
    ranges = labels.get("files", {}).get(normalize_relpath(logical_name), [])
    votes: dict[str, int] = {}
    for row in ranges:
        overlap = max(0, min(end_sample, int(row["end_sample"])) - max(start_sample, int(row["start_sample"])))
        if overlap:
            speaker = str(row["speaker_id"])
            votes[speaker] = votes.get(speaker, 0) + overlap
    if not votes:
        return "spk_unknown"
    return sorted(votes.items(), key=lambda item: (-item[1], item[0].casefold(), item[0]))[0][0]


def segment_id_for(logical_name: str, start_sample: int, end_sample: int) -> str:
    stem = normalize_relpath(logical_name)
    return f"{stem}::seg_{start_sample:08d}_{end_sample:08d}"


def build_segment_defs(
    logical_name: str,
    intervals: Iterable[dict[str, int] | tuple[int, int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    normalized: list[tuple[int, int]] = []
    for interval in intervals:
        if isinstance(interval, dict):
            normalized.append((int(interval["start_sample"]), int(interval["end_sample"])))
        else:
            normalized.append((int(interval[0]), int(interval[1])))
    normalized.sort(key=lambda row: (row[0], row[1]))
    for index, (start, end) in enumerate(normalized):
        rows.append(
            {
                "file": logical_name,
                "start_sample": start,
                "end_sample": end,
                "ordinal_in_file": index,
                "segment_id": segment_id_for(logical_name, start, end),
            }
        )
    return rows
