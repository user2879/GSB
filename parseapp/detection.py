"""Deterministic table-line detection and table/chart region scoring."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .config import ParseConfig
from .validation import PageInvalid


PRE_FILTER_LENGTH = 8


def _runs(mask: np.ndarray, minimum: int) -> list[tuple[int, int]]:
    padded = np.empty(mask.size + 2, dtype=np.bool_)
    padded[0] = False
    padded[-1] = False
    padded[1:-1] = mask
    changes = np.flatnonzero(padded[1:] != padded[:-1])
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    selected = lengths >= minimum
    return [(int(start), int(end - 1)) for start, end in zip(starts[selected], ends[selected])]


def enumerate_candidates(binary: np.ndarray) -> list[dict[str, Any]]:
    """Enumerate dark runs that are plausible line fragments.

    Every run of at least PRE_FILTER_LENGTH pixels consumes one detection point.
    Runs below min_line_length are recorded as E_LINE_TOO_SHORT candidates.
    """

    height, width = binary.shape
    candidates: list[dict[str, Any]] = []
    for y in range(height):
        for start, end in _runs(binary[y, :], PRE_FILTER_LENGTH):
            candidates.append(
                {"axis": "h", "fixed": y, "start": start, "end": end, "length": end - start + 1}
            )
    for x in range(width):
        for start, end in _runs(binary[:, x], PRE_FILTER_LENGTH):
            candidates.append(
                {"axis": "v", "fixed": x, "start": start, "end": end, "length": end - start + 1}
            )

    candidates.sort(
        key=lambda item: (
            0 if item["axis"] == "h" else 1,
            item["fixed"],
            item["start"],
            item["end"],
        )
    )
    for index, candidate in enumerate(candidates, start=1):
        prefix = "HS" if candidate["axis"] == "h" else "VS"
        candidate["source_id"] = f"{prefix}{index:06d}"
    return candidates


def _interval_overlap(first: Mapping[str, Any], second: Mapping[str, Any]) -> int:
    return max(0, min(first["end"], second["end"]) - max(first["start"], second["start"]) + 1)


def _group_candidates(
    candidates: Sequence[Mapping[str, Any]], tolerance: int
) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for candidate in candidates:
        best_group: dict[str, Any] | None = None
        for group in groups:
            if abs(candidate["fixed"] - group["max_fixed"]) <= tolerance and _interval_overlap(
                candidate, group
            ) > 0:
                best_group = group
                break
        if best_group is None:
            best_group = {
                "axis": candidate["axis"],
                "min_fixed": candidate["fixed"],
                "max_fixed": candidate["fixed"],
                "start": candidate["start"],
                "end": candidate["end"],
                "candidates": [],
            }
            groups.append(best_group)
        best_group["min_fixed"] = min(best_group["min_fixed"], candidate["fixed"])
        best_group["max_fixed"] = max(best_group["max_fixed"], candidate["fixed"])
        best_group["start"] = min(best_group["start"], candidate["start"])
        best_group["end"] = max(best_group["end"], candidate["end"])
        best_group["candidates"].append(dict(candidate))
    return groups


def detect_lines(
    binary: np.ndarray,
    config: ParseConfig,
    candidates: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    height, width = binary.shape
    prepared = list(candidates) if candidates is not None else enumerate_candidates(binary)
    minimum = config.min_line_length
    tolerance = config.line_merge_tolerance

    short_drops: list[dict[str, Any]] = []
    long_candidates: list[dict[str, Any]] = []
    for candidate in prepared:
        if candidate["length"] < minimum:
            short_drops.append(
                {
                    "source_id": candidate["source_id"],
                    "axis": candidate["axis"],
                    "coord": int(candidate["fixed"]),
                    "start": int(candidate["start"]),
                    "end": int(candidate["end"]),
                    "length": int(candidate["length"]),
                    "reason": "E_LINE_TOO_SHORT",
                }
            )
        else:
            long_candidates.append(dict(candidate))

    groups: list[dict[str, Any]] = []
    for axis in ("h", "v"):
        axis_candidates = [item for item in long_candidates if item["axis"] == axis]
        groups.extend(_group_candidates(axis_candidates, tolerance))

    merged: list[dict[str, Any]] = []
    width_drops: list[dict[str, Any]] = []
    bounds_drops: list[dict[str, Any]] = []
    for group in groups:
        thickness = group["max_fixed"] - group["min_fixed"] + 1
        coord = (group["min_fixed"] + group["max_fixed"]) // 2
        sources = [candidate["source_id"] for candidate in group["candidates"]]
        source_candidates = [
            {
                "source_id": candidate["source_id"],
                "axis": candidate["axis"],
                "coord": int(candidate["fixed"]),
                "start": int(candidate["start"]),
                "end": int(candidate["end"]),
            }
            for candidate in group["candidates"]
        ]
        if thickness > 2 * tolerance + 1:
            width_drops.append(
                {
                    "axis": group["axis"],
                    "coord": coord,
                    "start": int(group["start"]),
                    "end": int(group["end"]),
                    "thickness": thickness,
                    "source_ids": sources,
                    "source_candidates": source_candidates,
                    "reason": "E_LINE_WIDTH_EXCEEDS_TOLERANCE",
                }
            )
            continue
        if group["axis"] == "h":
            coord_limit = height - 1
            extent_limit = width - 1
        else:
            coord_limit = width - 1
            extent_limit = height - 1
        if (
            coord < 0
            or coord > coord_limit
            or group["start"] < 0
            or group["end"] > extent_limit
        ):
            bounds_drops.append(
                {
                    "axis": group["axis"],
                    "coord": coord,
                    "start": int(group["start"]),
                    "end": int(group["end"]),
                    "source_ids": sources,
                    "source_candidates": source_candidates,
                    "reason": "E_LINE_OUT_OF_BOUNDS",
                }
            )
            continue
        merged.append(
            {
                "axis": group["axis"],
                "coord": coord,
                "start": int(group["start"]),
                "end": int(group["end"]),
                "length": int(group["end"] - group["start"] + 1),
                "thickness": thickness,
                "source_ids": sources,
                "source_candidates": source_candidates,
                "merge": {
                    "rule": "ADJACENT_WITHIN_TOLERANCE_AND_OVERLAPPING",
                    "source_count": len(sources),
                    "tolerance": tolerance,
                },
            }
        )

    merged.sort(key=lambda item: (0 if item["axis"] == "h" else 1, item["coord"], item["start"], item["end"]))
    lines: list[dict[str, Any]] = []
    for index, segment in enumerate(merged, start=1):
        prefix = "HL" if segment["axis"] == "h" else "VL"
        segment["line_id"] = f"{prefix}{index:06d}"
        lines.append(segment)

    drops = short_drops + width_drops + bounds_drops
    drops.sort(
        key=lambda item: (
            0 if item["axis"] == "h" else 1,
            item["coord"],
            item["start"],
            item["end"],
            item["reason"],
        )
    )
    drop_counts = {
        reason: sum(1 for item in drops if item["reason"] == reason)
        for reason in (
            "E_LINE_TOO_SHORT",
            "E_LINE_WIDTH_EXCEEDS_TOLERANCE",
            "E_LINE_OUT_OF_BOUNDS",
        )
    }
    param_hash = config.stage_param_hash("line")
    payload = {
        "axis": "image",
        "content_length": int(binary.size),
        "params": config.stage_params("line"),
    }
    key_material = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    param_key = hashlib.sha256(key_material.encode("utf-8")).hexdigest()[:20]

    return {
        "segments": lines,
        "dropped_candidates": drops,
        "drop_counts": drop_counts,
        "candidate_count": len(prepared),
        "points_consumed": len(prepared),
        "param_hash": param_hash,
        "param_key": param_key,
        "merge_rule": {
            "tolerance": tolerance,
            "max_thickness": 2 * tolerance + 1,
            "prefilter_length": PRE_FILTER_LENGTH,
        },
    }


def line_cache_key(content_fingerprint: str, config: ParseConfig) -> str:
    payload = {
        "stage": "line",
        "content": content_fingerprint,
        "params": config.stage_params("line"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _subtract_exclusions(
    start: int,
    end: int,
    axis: str,
    coord: int,
    excluded_boxes: Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    intervals = [(start, end)]
    for left, top, right, bottom in excluded_boxes:
        if axis == "h":
            if not (top <= coord <= bottom):
                continue
            cut_start, cut_end = left, right
        else:
            if not (left <= coord <= right):
                continue
            cut_start, cut_end = top, bottom
        updated: list[tuple[int, int]] = []
        for interval_start, interval_end in intervals:
            if cut_end < interval_start or cut_start > interval_end:
                updated.append((interval_start, interval_end))
                continue
            if interval_start < cut_start:
                updated.append((interval_start, cut_start - 1))
            if cut_end + 1 <= interval_end:
                updated.append((cut_end + 1, interval_end))
        intervals = updated
    return [(a, b) for a, b in intervals if a <= b]


def _union_length(intervals: Iterable[tuple[int, int]]) -> int:
    ordered = sorted(intervals)
    if not ordered:
        return 0
    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end + 1:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start + 1
            current_start, current_end = start, end
    total += current_end - current_start + 1
    return total


def _axis_stats(
    segments: Sequence[Mapping[str, Any]],
    axis: str,
    box: Sequence[int],
    excluded_boxes: Sequence[Sequence[int]],
) -> tuple[int, float]:
    left, top, right, bottom = box
    if axis == "h":
        lower, upper = left, right
        span = right - left + 1
    else:
        lower, upper = top, bottom
        span = bottom - top + 1
    by_coord: dict[int, list[tuple[int, int]]] = {}
    for segment in segments:
        if segment["axis"] != axis:
            continue
        coord = int(segment["coord"])
        if axis == "h":
            if not top <= coord <= bottom:
                continue
        else:
            if not left <= coord <= right:
                continue
        clipped = _subtract_exclusions(
            max(int(segment["start"]), lower),
            min(int(segment["end"]), upper),
            axis,
            coord,
            excluded_boxes,
        )
        clipped = [(start, end) for start, end in clipped if end - start + 1 >= 4]
        if clipped:
            by_coord.setdefault(coord, []).extend(clipped)
    if not by_coord:
        return 0, 0.0
    coverage = max(
        _union_length(intervals) / float(max(1, span)) for intervals in by_coord.values()
    )
    return len(by_coord), min(1.0, coverage)


def region_line_stats(
    segments: Sequence[Mapping[str, Any]],
    box: Sequence[int],
    excluded_boxes: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    horizontal_count, horizontal_coverage = _axis_stats(
        segments, "h", box, excluded_boxes
    )
    vertical_count, vertical_coverage = _axis_stats(
        segments, "v", box, excluded_boxes
    )
    score = (
        min(1.0, horizontal_count / 2.0)
        * min(1.0, vertical_count / 2.0)
        * min(horizontal_coverage, vertical_coverage)
    )
    return {
        "horizontal_lines": horizontal_count,
        "vertical_lines": vertical_count,
        "horizontal_coverage": round(horizontal_coverage, 6),
        "vertical_coverage": round(vertical_coverage, 6),
        "table_grid_score": round(score, 6),
    }


def _intersection(first: Sequence[int], second: Sequence[int]) -> list[int] | None:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    if left >= right or top >= bottom:
        return None
    return [left, top, right, bottom]


def classify_regions(
    segments: Sequence[Mapping[str, Any]],
    truth: Mapping[str, Any],
    threshold: float,
) -> dict[str, Any]:
    table_decisions: dict[str, Any] = {}
    chart_decisions: dict[str, Any] = {}

    for table in truth["tables"]:
        stats = region_line_stats(segments, table["region"])
        if stats["table_grid_score"] < threshold:
            raise PageInvalid(
                "E_REGION_ROLE_MISMATCH",
                f"table {table['table_id']} scored {stats['table_grid_score']} below "
                f"chart_decision_threshold {threshold}",
            )
        table_decisions[table["table_id"]] = {
            "declared_role": "table",
            "classified_role": "table",
            "rationale": "TABLE_GRID_SCORE_GTE_THRESHOLD",
            "chart_decision_threshold": threshold,
            **stats,
        }

    table_boxes = [table["region"] for table in truth["tables"]]
    for chart in truth["charts"]:
        overlaps: list[dict[str, Any]] = []
        excluded_boxes: list[list[int]] = []
        for table in truth["tables"]:
            intersection = _intersection(chart["region"], table["region"])
            if intersection is not None:
                excluded_boxes.append(intersection)
                overlaps.append(
                    {
                        "table_id": table["table_id"],
                        "intersection": intersection,
                        "priority": "table",
                        "rationale": "TABLE_PRIORITY_ON_OVERLAP",
                    }
                )
        stats = region_line_stats(segments, chart["region"], excluded_boxes)
        if stats["table_grid_score"] >= threshold:
            raise PageInvalid(
                "E_REGION_ROLE_MISMATCH",
                f"chart {chart['chart_id']} scored {stats['table_grid_score']} at or above "
                f"chart_decision_threshold {threshold}",
            )
        chart_decisions[chart["chart_id"]] = {
            "declared_role": "chart",
            "classified_role": "chart",
            "rationale": "TABLE_GRID_SCORE_LT_THRESHOLD",
            "chart_decision_threshold": threshold,
            "overlap": overlaps,
            "excluded_boxes": excluded_boxes,
            **stats,
        }

    return {"tables": table_decisions, "charts": chart_decisions}
