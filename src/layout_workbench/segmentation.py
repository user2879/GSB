from __future__ import annotations

from typing import Any

import numpy as np

from .config import RecognitionConfig


def _initial_runs(projection: np.ndarray, x_offset: int) -> list[dict[str, int]]:
    active = np.flatnonzero(projection > 0)
    if len(active) == 0:
        return []
    runs: list[dict[str, int]] = []
    start = int(active[0])
    previous = int(active[0])
    for raw in active[1:]:
        value = int(raw)
        if value != previous + 1:
            runs.append(
                {
                    "x0": x_offset + start,
                    "x1": x_offset + previous + 1,
                    "width": previous - start + 1,
                }
            )
            start = value
        previous = value
    runs.append(
        {
            "x0": x_offset + start,
            "x1": x_offset + previous + 1,
            "width": previous - start + 1,
        }
    )
    return runs


def _range_height(mask: np.ndarray, x0: int, x1: int) -> int:
    sub = mask[:, x0:x1]
    rows = np.flatnonzero(sub.sum(axis=1) > 0)
    return 0 if len(rows) == 0 else int(rows[-1] - rows[0] + 1)


def _merge_narrow_fragments(
    segments: list[dict[str, int]],
    mask: np.ndarray,
    config: RecognitionConfig,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    current = [dict(segment) for segment in segments]
    event_index = 0
    while True:
        narrow = [i for i, segment in enumerate(current) if segment["width"] < config.char_min_width]
        if not narrow:
            break
        index = min(narrow, key=lambda i: (current[i]["width"], current[i]["x0"]))
        segment = current[index]
        candidates: list[tuple[int, int, str]] = []
        if index > 0:
            left = current[index - 1]
            candidates.append((segment["x0"] - left["x1"], index - 1, "left"))
        if index + 1 < len(current):
            right = current[index + 1]
            candidates.append((right["x0"] - segment["x1"], index + 1, "right"))
        if not candidates:
            records.append(
                {
                    "event_index": event_index,
                    "event": "unmerged_narrow_retained",
                    "rule": "no_neighbor_available",
                    "original_range": [segment["x0"], segment["x1"]],
                    "original_width": segment["width"],
                    "classification": "retained",
                }
            )
            break
        if config.fragment_merge_preference == "nearest_then_right":
            candidates.sort(key=lambda item: (item[0], 0 if item[2] == "right" else 1))
        else:
            candidates.sort(key=lambda item: (item[0], 0 if item[2] == "left" else 1))
        gap, neighbor_index, direction = candidates[0]
        neighbor = current[neighbor_index]
        first = min(index, neighbor_index)
        second = max(index, neighbor_index)
        left_segment = current[first]
        right_segment = current[second]
        merged = {
            "x0": left_segment["x0"],
            "x1": right_segment["x1"],
            "width": right_segment["x1"] - left_segment["x0"],
        }
        left_height = _range_height(mask, left_segment["x0"] - 0, left_segment["x1"] - 0)
        right_height = _range_height(mask, right_segment["x0"] - 0, right_segment["x1"] - 0)
        normal = gap <= config.fragment_merge_gap and min(left_height, right_height) >= config.min_char_height
        records.append(
            {
                "event_index": event_index,
                "event": "merge_narrow",
                "rule": f"nearest_then_{direction}",
                "direction": direction,
                "merged_segments": [
                    {
                        "x0": left_segment["x0"],
                        "x1": left_segment["x1"],
                        "width": left_segment["width"],
                        "ink_height": left_height,
                    },
                    {
                        "x0": right_segment["x0"],
                        "x1": right_segment["x1"],
                        "width": right_segment["width"],
                        "ink_height": right_height,
                    },
                ],
                "original_width_ranges": [
                    [left_segment["x0"], left_segment["x1"]],
                    [right_segment["x0"], right_segment["x1"]],
                ],
                "gap": int(gap),
                "result_range": [merged["x0"], merged["x1"]],
                "result_width": merged["width"],
                "classification": "normal_intra_glyph_gap" if normal else "forced_swallow",
            }
        )
        event_index += 1
        current[first : second + 1] = [merged]
    return current, records


def _split_wide_fragments(
    segments: list[dict[str, int]],
    projection: np.ndarray,
    config: RecognitionConfig,
    start_event_index: int,
) -> tuple[list[dict[str, int]], list[dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    current = [dict(segment) for segment in segments]
    event_index = start_event_index
    guard = 0
    while True:
        guard += 1
        if guard > 10000:
            raise RuntimeError("wide-fragment splitting did not converge")
        wide = [i for i, segment in enumerate(current) if segment["width"] > config.char_max_width]
        if not wide:
            break
        index = min(wide, key=lambda i: (-current[i]["width"], current[i]["x0"]))
        segment = current[index]
        lo = segment["x0"] + config.char_min_width
        hi = segment["x1"] - config.char_min_width
        if lo > hi:
            records.append(
                {
                    "event_index": event_index,
                    "event": "wide_split_skipped",
                    "rule": "minimum_child_width",
                    "original_range": [segment["x0"], segment["x1"]],
                    "original_width": segment["width"],
                    "classification": "unresolved",
                }
            )
            event_index += 1
            segment["width"] = config.char_max_width - 1
            continue
        midpoint = (segment["x0"] + segment["x1"]) / 2.0
        split_at = min(
            range(lo, hi + 1),
            key=lambda c: (
                int(projection[c]),
                abs(c - midpoint),
                c,
            ),
        )
        valley = int(projection[split_at])
        peak = max(1, int(projection[segment["x0"] : segment["x1"]].max(initial=0)))
        ratio = valley / peak
        left = {
            "x0": segment["x0"],
            "x1": split_at,
            "width": split_at - segment["x0"],
        }
        right = {
            "x0": split_at,
            "x1": segment["x1"],
            "width": segment["x1"] - split_at,
        }
        records.append(
            {
                "event_index": event_index,
                "event": "split_wide",
                "rule": "lowest_projection_valley",
                "original_range": [segment["x0"], segment["x1"]],
                "original_width": segment["width"],
                "split_at": int(split_at),
                "valley_projection": valley,
                "valley_ratio": round(ratio, 6),
                "result_ranges": [
                    [left["x0"], left["x1"]],
                    [right["x0"], right["x1"]],
                ],
                "classification": "normal_projection_valley" if ratio <= 0.20 else "forced_width_split",
            }
        )
        event_index += 1
        current[index : index + 1] = [left, right]
    return current, records


def _offset_ranges(value: Any, offset: int) -> Any:
    if isinstance(value, list):
        return [_offset_ranges(item, offset) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if key in {"original_range", "result_range"} and isinstance(item, list):
                result[key] = [int(v) + offset for v in item]
            elif key in {"original_width_ranges", "result_ranges"} and isinstance(item, list):
                result[key] = [
                    [int(v) + offset for v in subrange] if isinstance(subrange, list) else subrange
                    for subrange in item
                ]
            elif key in {"x0", "x1", "split_at"} and isinstance(item, int):
                result[key] = item + offset
            else:
                result[key] = _offset_ranges(item, offset)
        return result
    if isinstance(value, tuple):
        return [_offset_ranges(item, offset) for item in value]
    return value


def segment_line(gray: np.ndarray, line: dict[str, Any], config: RecognitionConfig) -> dict[str, Any]:
    x0, y0, x1, y1 = line["bbox"]
    region = gray[y0:y1, x0:x1]
    mask = region <= config.binarization_threshold
    projection = mask.sum(axis=0).astype(np.int64)
    initial = _initial_runs(projection, 0)
    for fragment in initial:
        fragment["x0"] += x0
        fragment["x1"] += x0

    local_mask = mask
    local_initial = [dict(fragment, x0=fragment["x0"] - x0, x1=fragment["x1"] - x0) for fragment in initial]
    merged, merge_records = _merge_narrow_fragments(local_initial, local_mask, config)
    split, split_records = _split_wide_fragments(merged, projection, config, len(merge_records))
    merge_records = [_offset_ranges(record, x0) for record in merge_records]
    split_records = [_offset_ranges(record, x0) for record in split_records]
    for fragment in split:
        fragment["x0"] += x0
        fragment["x1"] += x0

    segments: list[dict[str, Any]] = []
    for segment_index, fragment in enumerate(sorted(split, key=lambda s: (s["x0"], s["x1"]))):
        sx0, sx1 = fragment["x0"], fragment["x1"]
        local = mask[:, max(0, sx0 - x0) : max(0, sx1 - x0)]
        ys_all, xs_all = np.nonzero(local)
        if len(ys_all) == 0:
            bbox = [sx0, y0, sx1, y1]
        else:
            column_counts = local.sum(axis=0)
            valid_columns = np.flatnonzero(column_counts >= config.line_connectivity_min_pixels)
            if len(valid_columns) == 0:
                xs_min, xs_max = int(xs_all.min()), int(xs_all.max())
            else:
                xs_min, xs_max = int(valid_columns[0]), int(valid_columns[-1])
            column_slice = local[:, xs_min : xs_max + 1]
            row_hits = np.flatnonzero(column_slice.sum(axis=1) > 0)
            if len(row_hits) == 0:
                ys_min, ys_max = int(ys_all.min()), int(ys_all.max())
            else:
                ys_min, ys_max = int(row_hits[0]), int(row_hits[-1])
            bbox = [
                sx0 + xs_min,
                y0 + ys_min,
                sx0 + xs_max + 1,
                y0 + ys_max + 1,
            ]
        segments.append(
            {
                "segment_index": segment_index,
                "bbox": bbox,
                "range": [sx0, sx1],
                "width": sx1 - sx0,
                "height": bbox[3] - bbox[1],
            }
        )
    return {
        "stage": "segmentation",
        "line_bbox": [x0, y0, x1, y1],
        "parameters": config.segmentation_params,
        "initial_fragments": [
            {"range": [f["x0"], f["x1"]], "width": f["width"]} for f in initial
        ],
        "segments": segments,
        "records": merge_records + split_records,
    }
