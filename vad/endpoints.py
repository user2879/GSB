"""Endpoint interval merging, filtering, ordering and count validation."""

from __future__ import annotations

from typing import Any

import numpy as np

from .state_machine import ENDPOINT_STATES
from .util import PRODUCT_VERSIONS, round_float


def build_endpoints(
    features: dict[str, Any],
    state_result: dict[str, Any],
    config: dict[str, Any],
    expected_range: tuple[int, int],
) -> dict[str, Any]:
    frames = features["frames"]
    frame_states = state_result["frame_states"]
    hop_ms = float(features["hop_ms"])
    duration_ms = float(features["duration_ms"])
    raw: list[dict[str, int]] = []
    start: int | None = None
    for index, state in enumerate(frame_states):
        is_endpoint = state["state"] in ENDPOINT_STATES
        if is_endpoint and start is None:
            start = index
        elif not is_endpoint and start is not None:
            raw.append({"start": start, "end": index})
            start = None
    if start is not None:
        raw.append({"start": start, "end": len(frame_states)})

    min_silence = float(config["min_silence_ms"])
    merged: list[dict[str, int]] = []
    for interval in raw:
        if not merged:
            merged.append(dict(interval))
            continue
        gap_ms = float(interval["start"] - merged[-1]["end"]) * hop_ms
        if gap_ms < min_silence:
            merged[-1]["end"] = max(merged[-1]["end"], interval["end"])
        else:
            merged.append(dict(interval))

    min_speech = float(config["min_speech_ms"])
    filtered = [
        interval
        for interval in merged
        if float(interval["end"] - interval["start"]) * hop_ms >= min_speech
    ]
    segments: list[dict[str, Any]] = []
    for interval in filtered:
        start_frame = int(interval["start"])
        end_frame = int(interval["end"])
        included = frames[start_frame:end_frame]
        start_ms = float(included[0]["start_ms"])
        end_ms = min(duration_ms, float(end_frame) * hop_ms)
        mean_energy = float(
            np.mean(
                np.asarray([float(frame["rms_log"]) for frame in included], dtype=np.float64)
            )
        )
        segments.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "end_frame_inclusive": end_frame - 1,
                "start_ms": round(start_ms, 9),
                "end_ms": round(end_ms, 9),
                "duration_ms": round(end_ms - start_ms, 9),
                "mean_energy_ln": round_float(mean_energy, 10),
            }
        )
    segments.sort(key=lambda segment: (segment["start_ms"], segment["end_ms"]))
    lower, upper = int(expected_range[0]), int(expected_range[1])
    count = len(segments)
    return {
        "schema_version": "1.0",
        "product_version": PRODUCT_VERSIONS["endpoints"],
        "segments": segments,
        "segment_count": count,
        "expected_segment_range": [lower, upper],
        "segment_count_ok": bool(lower <= count <= upper),
        "frame_index_semantics": "start_frame inclusive, end_frame exclusive; ms is half-open hop timeline",
        "raw_interval_count": len(raw),
        "merged_interval_count": len(merged),
        "discarded_short_interval_count": len(merged) - len(filtered),
        "min_speech_ms": min_speech,
        "min_silence_ms": min_silence,
    }
