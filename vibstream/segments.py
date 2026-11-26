"""Window-to-segment aggregation with a deterministic shortest-segment rule."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import AppConfig
from .fingerprints import segment_cache_key
from .windows import Window


@dataclass
class SegmentDraft:
    windows: list[Window]
    cluster_id: str


def _coalesce(segments: list[SegmentDraft]) -> None:
    changed = True
    while changed:
        changed = False
        for index in range(len(segments) - 1):
            if segments[index].cluster_id == segments[index + 1].cluster_id:
                segments[index].windows.extend(segments[index + 1].windows)
                del segments[index + 1]
                changed = True
                break


def build_segments(
    *,
    retained_windows: list[Window],
    assignments: dict[str, str],
    observations: dict[str, dict[str, Any]],
    config: AppConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = sorted(retained_windows, key=lambda item: item.index)
    segments: list[SegmentDraft] = []
    for window in ordered:
        if window.window_id not in assignments:
            continue
        cluster_id = assignments[window.window_id]
        if (
            segments
            and segments[-1].cluster_id == cluster_id
            and segments[-1].windows[-1].index + 1 == window.index
        ):
            segments[-1].windows.append(window)
        else:
            segments.append(SegmentDraft([window], cluster_id))

    smoothing_records: list[dict[str, Any]] = []
    if not segments:
        return [], smoothing_records
    while len(segments) > 1:
        short_indexes = [index for index, segment in enumerate(segments) if len(segment.windows) < config.segment.min_windows]
        if not short_indexes:
            break
        index = min(short_indexes, key=lambda item: (len(segments[item].windows), segments[item].windows[0].index))
        segment = segments[index]
        neighbor_indexes = [candidate for candidate in (index - 1, index + 1) if 0 <= candidate < len(segments)]
        neighbor_index = min(
            neighbor_indexes,
            key=lambda candidate: (-len(segments[candidate].windows), segments[candidate].windows[0].index),
        )
        neighbor = segments[neighbor_index]
        old_cluster = segment.cluster_id
        smoothing_records.append(
            {
                "short_segment_windows": [window.window_id for window in segment.windows],
                "short_segment_length": len(segment.windows),
                "old_cluster_id": old_cluster,
                "new_cluster_id": neighbor.cluster_id,
                "basis": "shorter_segment_merged_into_longer_adjacent_segment",
                "tie_break": "earlier_segment",
            }
        )
        segment.cluster_id = neighbor.cluster_id
        _coalesce(segments)

    cluster_ids = sorted({segment.cluster_id for segment in segments})
    condition_for = {cluster_id: f"condition_{rank}" for rank, cluster_id in enumerate(cluster_ids)}
    results: list[dict[str, Any]] = []
    for segment in segments:
        block_refs = sorted(
            {block_id for window in segment.windows for block_id in window.block_refs}
        )
        entries = [
            {
                "window_id": window.window_id,
                "window_index": window.index,
                "observation_key": observations[window.window_id]["cache_key"],
                "cluster_id": assignments[window.window_id],
                "effective_cluster_id": segment.cluster_id,
            }
            for window in segment.windows
        ]
        key = segment_cache_key(entries, config)
        results.append(
            {
                "start_window_index": segment.windows[0].index,
                "end_window_index": segment.windows[-1].index,
                "window_ids": [window.window_id for window in segment.windows],
                "window_count": len(segment.windows),
                "cluster_id": segment.cluster_id,
                "condition": condition_for[segment.cluster_id],
                "block_refs": block_refs,
                "cache_key": key,
                "below_minimum_after_smoothing": len(segment.windows) < config.segment.min_windows,
            }
        )
    return results, smoothing_records
