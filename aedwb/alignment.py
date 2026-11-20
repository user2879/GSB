"""Fixed grid realignment with explicit interpolation provenance and residuals."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def _lerp_vector(left: list[float], right: list[float], fraction: float) -> list[float]:
    if fraction <= 0.0:
        return list(left)
    if fraction >= 1.0:
        return list(right)
    return [
        float(a + (b - a) * fraction)
        for a, b in zip(left, right)
    ]


def _probability_space(row: list[float], score_type: str) -> np.ndarray:
    values = np.asarray(row, dtype=np.float64)
    if score_type == "logit":
        exponentials = np.exp(values - float(np.max(values)))
        total = float(np.sum(exponentials))
        return exponentials / total if total > 0.0 else np.zeros_like(values)
    total = float(np.sum(values))
    return values / total if total > 0.0 else np.zeros_like(values)


def align_packet_frames(
    mapping_payload: dict[str, Any],
    global_frame_shift_ms: int,
) -> dict[str, Any]:
    """Map local frame timestamps onto the global frame grid.

    The fixed rule is linear interpolation at each global frame timestamp
    ``start_ms + local_index * local_shift``.  At source support edges the
    endpoint value is held.  Every target row records both source indices and
    the probability-space residual, so neither truncation nor zero padding is
    silent.  Edge holds and uncovered header intervals are also reported.
    """

    start_ms = int(mapping_payload["start_ms"])
    end_ms = int(mapping_payload["end_ms"])
    local_shift = int(mapping_payload["frame_shift_ms"])
    rows: list[list[float]] = mapping_payload["mapped_rows"]
    frame_count = len(rows)
    if frame_count == 0:
        return {
            **{key: mapping_payload[key] for key in (
                "packet_id",
                "source_file",
                "source_score_type",
                "content_sha256",
                "audio_file",
                "start_ms",
                "end_ms",
                "frame_shift_ms",
                "frame_grid_version",
                "class_codebook_version",
                "detector_version",
                "mapping_entries",
                "mapping_by_name",
                "unmapped_frame_count",
                "unmapped_frames",
                "unmapped_row_indices",
                "injection_ids",
            )},
            "original_frame_count": 0,
            "target_frame_count": 0,
            "resampled": False,
            "alignment_method": "linear_interp_edge_hold_v1",
            "uncovered_intervals_ms": [[start_ms, end_ms]],
            "edge_hold_frame_count": 0,
            "frames": [],
            "mean_alignment_residual": 0.0,
            "max_alignment_residual": 0.0,
        }

    global_start = math.ceil(start_ms / global_frame_shift_ms)
    global_stop = math.ceil(end_ms / global_frame_shift_ms)
    target_count = max(0, global_stop - global_start)
    resampled = local_shift != global_frame_shift_ms or target_count != frame_count
    aligned: list[dict[str, Any]] = []
    residuals: list[float] = []
    edge_holds = 0
    unmapped_indices = set(mapping_payload.get("unmapped_row_indices", []))

    for global_index in range(global_start, global_stop):
        global_time_ms = global_index * global_frame_shift_ms
        source_position = (global_time_ms - start_ms) / float(local_shift)
        edge_hold = False
        if source_position <= 0.0:
            left_index = 0
            right_index = 0
            fraction = 0.0
            edge_hold = source_position < 0.0
        elif source_position >= frame_count - 1:
            left_index = frame_count - 1
            right_index = frame_count - 1
            fraction = 0.0
            edge_hold = source_position > frame_count - 1
        else:
            left_index = int(math.floor(source_position))
            right_index = min(left_index + 1, frame_count - 1)
            fraction = source_position - left_index

        value = _lerp_vector(rows[left_index], rows[right_index], fraction)
        nearest_index = left_index if fraction <= 0.5 else right_index
        nearest = rows[nearest_index]
        value_probability = _probability_space(value, mapping_payload["source_score_type"])
        nearest_probability = _probability_space(nearest, mapping_payload["source_score_type"])
        residual = float(
            np.mean(np.abs(value_probability - nearest_probability))
        )
        residuals.append(float(residual))
        if edge_hold:
            edge_holds += 1
        original_indices = [left_index] if left_index == right_index else [left_index, right_index]
        unmapped = left_index in unmapped_indices or right_index in unmapped_indices
        if unmapped:
            value = [0.0] * len(value)
            residual = 0.0
        mapping_residual = float(mapping_payload["mapping_residuals"][nearest_index])
        aligned.append(
            {
                "global_frame_index": global_index,
                "global_time_ms": global_time_ms,
                "original_frame_indices": original_indices,
                "original_frame_index_anchor": nearest_index,
                "interpolation_fraction": float(fraction),
                "resampled": resampled,
                "edge_hold": edge_hold,
                "unmapped": unmapped,
                "mapped_scores": value,
                "alignment_residual": float(residual),
                "mapping_residual": mapping_residual,
            }
        )

    uncovered: list[list[int]] = []
    coverage_start = global_start * global_frame_shift_ms
    coverage_end = global_stop * global_frame_shift_ms
    if coverage_start > start_ms:
        uncovered.append([start_ms, coverage_start])
    if coverage_end < end_ms:
        uncovered.append([coverage_end, end_ms])

    return {
        **{key: mapping_payload[key] for key in (
            "packet_id",
            "source_file",
            "source_score_type",
            "content_sha256",
            "audio_file",
            "start_ms",
            "end_ms",
            "frame_shift_ms",
            "frame_grid_version",
            "class_codebook_version",
            "detector_version",
            "mapping_entries",
            "mapping_by_name",
            "unmapped_frame_count",
            "unmapped_frames",
            "unmapped_row_indices",
            "injection_ids",
        )},
        "original_frame_count": frame_count,
        "target_frame_count": target_count,
        "resampled": resampled,
        "alignment_method": "linear_interp_edge_hold_v1",
        "global_frame_range": [global_start, global_stop - 1] if target_count else [],
        "uncovered_intervals_ms": uncovered,
        "edge_hold_frame_count": edge_holds,
        "frames": aligned,
        "mean_alignment_residual": float(sum(residuals) / len(residuals)) if residuals else 0.0,
        "max_alignment_residual": float(max(residuals)) if residuals else 0.0,
    }
