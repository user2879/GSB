"""Chart classification helpers and image-driven data-point extraction."""

from __future__ import annotations

from collections import deque
import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from .config import ParseConfig
from .validation import PageInvalid, region_signature


_POINT_CACHE_VERSION = "point-v1"


def _erode_3x3(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.bool_), 1, mode="constant", constant_values=False)
    return (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )


def _connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=np.bool_)
    components: list[dict[str, Any]] = []
    for start_y, start_x in zip(*np.nonzero(mask)):
        if visited[start_y, start_x]:
            continue
        queue: deque[tuple[int, int]] = deque([(int(start_y), int(start_x))])
        visited[start_y, start_x] = True
        count = 0
        sum_x = 0
        sum_y = 0
        min_x = max_x = int(start_x)
        min_y = max_y = int(start_y)
        while queue:
            y, x = queue.popleft()
            count += 1
            sum_x += x
            sum_y += y
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)
            for delta_y in (-1, 0, 1):
                for delta_x in (-1, 0, 1):
                    if delta_x == 0 and delta_y == 0:
                        continue
                    neighbor_y = y + delta_y
                    neighbor_x = x + delta_x
                    if (
                        0 <= neighbor_y < height
                        and 0 <= neighbor_x < width
                        and mask[neighbor_y, neighbor_x]
                        and not visited[neighbor_y, neighbor_x]
                    ):
                        visited[neighbor_y, neighbor_x] = True
                        queue.append((neighbor_y, neighbor_x))
        components.append(
            {
                "area": count,
                "bbox": [min_x, min_y, max_x, max_y],
                "centroid": [sum_x / count, sum_y / count],
            }
        )
    components.sort(
        key=lambda item: (item["centroid"][0], item["centroid"][1], item["bbox"])
    )
    return components


def _roi_mask(
    binary: np.ndarray,
    box: Sequence[int],
    excluded_boxes: Sequence[Sequence[int]] = (),
) -> np.ndarray:
    left, top, right, bottom = box
    mask = np.zeros_like(binary, dtype=np.bool_)
    mask[top : bottom + 1, left : right + 1] = binary[top : bottom + 1, left : right + 1]
    for ex_left, ex_top, ex_right, ex_bottom in excluded_boxes:
        mask[ex_top : ex_bottom + 1, ex_left : ex_right + 1] = False
    return mask


def detect_chart_kind(
    binary: np.ndarray,
    chart: Mapping[str, Any],
    excluded_boxes: Sequence[Sequence[int]] = (),
) -> dict[str, Any]:
    left, top, right, bottom = chart["plot_area"]
    mask = _roi_mask(binary, chart["plot_area"], excluded_boxes)
    rows = np.arange(top, bottom + 1, dtype=np.int64)[:, None]
    mask[top : bottom + 1, left : right + 1] &= rows < chart["axis"]["y"]["pixel_min"] - 1
    eroded = _erode_3x3(mask)
    components = _connected_components(eroded)
    largest_area = max((component["area"] for component in components), default=0)
    detected = "bar" if largest_area >= 100 else "line"
    return {
        "kind": detected,
        "largest_eroded_component_area": largest_area,
        "rule": "BAR_IF_LARGEST_3X3_ERODED_COMPONENT_GTE_100",
        "components": len(components),
    }


def _bar_clusters(
    mask: np.ndarray,
    plot_area: Sequence[int],
    minimum_column_count: int,
) -> list[tuple[int, int]]:
    left, top, right, bottom = plot_area
    counts = mask[top : bottom + 1, left : right + 1].sum(axis=0)
    active = np.flatnonzero(counts >= minimum_column_count)
    if active.size == 0:
        return []
    clusters: list[tuple[int, int]] = []
    start = int(active[0])
    previous = int(active[0])
    for value in active[1:]:
        value = int(value)
        if value > previous + 1:
            clusters.append((start + left, previous + left))
            start = value
        previous = value
    clusters.append((start + left, previous + left))
    return clusters


def _axis_refs(
    segments: Sequence[Mapping[str, Any]], chart: Mapping[str, Any]
) -> dict[str, list[str]]:
    region = chart["region"]
    y_baseline = chart["axis"]["y"]["pixel_min"]
    x_candidates = []
    y_candidates = []
    for segment in segments:
        coord = int(segment["coord"])
        if segment["axis"] == "h":
            if region[1] <= coord <= region[3]:
                distance = abs(coord - y_baseline)
                y_candidates.append((distance, coord, segment["line_id"]))
        else:
            if region[0] <= coord <= region[2]:
                distance = abs(coord - chart["plot_area"][0])
                x_candidates.append((distance, coord, segment["line_id"]))
    horizontal = [item[2] for item in sorted(y_candidates)[:2]]
    vertical = [item[2] for item in sorted(x_candidates)[:2]]
    return {"horizontal": horizontal, "vertical": vertical}


def _normalization_record(chart: Mapping[str, Any], method: str) -> dict[str, Any]:
    return {
        "method": method,
        "x": dict(chart["axis"]["x"]),
        "y": dict(chart["axis"]["y"]),
    }


def _map_point(
    pixel_x: float,
    pixel_y: float,
    chart: Mapping[str, Any],
    method: str,
) -> dict[str, Any]:
    x_axis = chart["axis"]["x"]
    y_axis = chart["axis"]["y"]
    x_pixel_ratio = (pixel_x - x_axis["pixel_min"]) / (
        x_axis["pixel_max"] - x_axis["pixel_min"]
    )
    y_pixel_ratio = (y_axis["pixel_min"] - pixel_y) / (
        y_axis["pixel_min"] - y_axis["pixel_max"]
    )
    x_value = x_axis["value_min"] + x_pixel_ratio * (
        x_axis["value_max"] - x_axis["value_min"]
    )
    y_value = y_axis["value_min"] + y_pixel_ratio * (
        y_axis["value_max"] - y_axis["value_min"]
    )
    if method == "minmax":
        normalized_x = (x_value - x_axis["value_min"]) / (
            x_axis["value_max"] - x_axis["value_min"]
        )
        normalized_y = (y_value - y_axis["value_min"]) / (
            y_axis["value_max"] - y_axis["value_min"]
        )
    else:
        normalized_x = x_pixel_ratio
        normalized_y = y_pixel_ratio
    return {
        "normalized": {
            "x": round(normalized_x, 6),
            "y": round(normalized_y, 6),
        },
        "value": {"x": round(x_value, 6), "y": round(y_value, 6)},
        "raw_pixel_ratio": {
            "x": round(x_pixel_ratio, 6),
            "y": round(y_pixel_ratio, 6),
        },
    }


def _point_cache_key(
    chart_region_content_sha256: str,
    chart_id: str,
    index: int,
    series: str,
    pixel: Sequence[int],
    chart: Mapping[str, Any],
    config: ParseConfig,
    axis_refs: Mapping[str, Any],
) -> str:
    payload = {
        "version": _POINT_CACHE_VERSION,
        "chart_region_content_sha256": chart_region_content_sha256,
        "chart_id": chart_id,
        "index": index,
        "series": series,
        "pixel": list(pixel),
        "region": chart["region"],
        "plot_area": chart["plot_area"],
        "axis": chart["axis"],
        "params": config.stage_params("chart"),
        "axis_line_refs": axis_refs,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _extract_bar_points(
    binary: np.ndarray,
    chart: Mapping[str, Any],
    excluded_boxes: Sequence[Sequence[int]],
    config: ParseConfig,
) -> list[dict[str, Any]]:
    x_axis = chart["axis"]["x"]
    y_axis = chart["axis"]["y"]
    left, top, right, bottom = chart["plot_area"]
    data_bottom = min(bottom, int(round(y_axis["pixel_min"])) - 1)
    data_box = [left, top, right, data_bottom]
    mask = _roi_mask(binary, data_box, excluded_boxes)
    height = max(1, data_bottom - top + 1)
    minimum_column_count = max(4, int(height * 0.04))
    clusters = _bar_clusters(mask, data_box, minimum_column_count)
    points: list[dict[str, Any]] = []
    for ordinal, (cluster_start, cluster_end) in enumerate(clusters):
        width = cluster_end - cluster_start + 1
        if width < 3:
            continue
        sub = mask[top : data_bottom + 1, cluster_start : cluster_end + 1]
        ys, _ = np.nonzero(sub)
        if ys.size == 0:
            continue
        bar_top = int(ys.min()) + top
        bar_bottom = int(ys.max()) + top
        if bar_bottom - bar_top + 1 < max(6, int(height * 0.03)):
            continue
        center_x = (cluster_start + cluster_end) // 2
        mapped = _map_point(center_x, bar_top, chart, config.coordinate_normalization)
        points.append(
            {
                "sequence_position": ordinal,
                "pixel": [center_x, bar_top],
                "bar_extent": [cluster_start, cluster_end, bar_top, bar_bottom],
                **mapped,
            }
        )
    points.sort(key=lambda item: (item["pixel"][0], item["pixel"][1]))
    for index, point in enumerate(points):
        point["sequence_position"] = index
    return points


def _extract_line_points(
    binary: np.ndarray,
    chart: Mapping[str, Any],
    excluded_boxes: Sequence[Sequence[int]],
    config: ParseConfig,
) -> list[dict[str, Any]]:
    left, top, right, bottom = chart["plot_area"]
    data_bottom = min(bottom, int(round(chart["axis"]["y"]["pixel_min"])) - 1)
    data_box = [left, top, right, data_bottom]
    mask = _roi_mask(binary, data_box, excluded_boxes)
    eroded = _erode_3x3(mask)
    components = _connected_components(eroded)
    markers = [
        component
        for component in components
        if component["area"] >= 9
        and component["bbox"][2] - component["bbox"][0] >= 2
        and component["bbox"][3] - component["bbox"][1] >= 2
    ]
    points: list[dict[str, Any]] = []
    for ordinal, marker in enumerate(markers):
        pixel_x = int(round(marker["centroid"][0]))
        pixel_y = int(round(marker["centroid"][1]))
        mapped = _map_point(pixel_x, pixel_y, chart, config.coordinate_normalization)
        points.append(
            {
                "sequence_position": ordinal,
                "pixel": [pixel_x, pixel_y],
                "marker_bbox": list(marker["bbox"]),
                **mapped,
            }
        )
    points.sort(key=lambda item: (item["pixel"][0], item["pixel"][1]))
    for index, point in enumerate(points):
        point["sequence_position"] = index
    return points


def _previous_point_lookup(
    previous_charts: Mapping[str, Any] | None,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    lookup: dict[tuple[str, int], Mapping[str, Any]] = {}
    if not previous_charts:
        return lookup
    for chart in previous_charts.get("charts", []):
        for point in chart.get("points", []):
            lookup[(chart.get("chart_id"), point.get("sequence_position"))] = point
    return lookup


def _chart_cache_key(
    content_fingerprint: str,
    upstream_line_key: str,
    truth: Mapping[str, Any],
    config: ParseConfig,
    decisions: Mapping[str, Any],
) -> str:
    payload = {
        "stage": "chart",
        "upstream_line_key": upstream_line_key,
        "content": content_fingerprint,
        "params": config.stage_params("chart"),
        "truth": region_signature(truth, "chart"),
        "decisions": {
            chart_id: {
                "classified_role": decision.get("classified_role"),
                "table_grid_score": decision.get("table_grid_score"),
                "overlap": decision.get("overlap"),
                "excluded_boxes": decision.get("excluded_boxes"),
            }
            for chart_id, decision in sorted(decisions.items())
        },
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def structure_charts(
    gray: np.ndarray,
    truth: Mapping[str, Any],
    segments: Sequence[Mapping[str, Any]],
    chart_decisions: Mapping[str, Any],
    config: ParseConfig,
    content_fingerprint: str,
    line_key: str,
    previous_charts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    binary = gray < config.binary_threshold
    previous_points = _previous_point_lookup(previous_charts)
    charts: list[dict[str, Any]] = []
    reused_points = 0
    recomputed_points = 0
    warnings: list[dict[str, Any]] = []
    normalization = None
    remaining_quota = config.page_chart_point_quota

    for chart in truth["charts"]:
        chart_id = chart["chart_id"]
        truth_points = sorted(chart["points"], key=lambda item: item["index"])
        series_by_ordinal = {
            ordinal: point["series"] for ordinal, point in enumerate(truth_points)
        }
        decision = chart_decisions[chart_id]
        excluded_boxes = decision.get("excluded_boxes", [])
        region_left, region_top, region_right, region_bottom = chart["region"]
        chart_crop_sha = hashlib.sha256(
            np.ascontiguousarray(
                gray[region_top : region_bottom + 1, region_left : region_right + 1]
            ).tobytes()
        ).hexdigest()
        kind_info = detect_chart_kind(binary, chart, excluded_boxes)
        if kind_info["kind"] != chart["kind"]:
            raise PageInvalid(
                "E_CHART_KIND_MISMATCH",
                f"chart {chart_id} declared {chart['kind']} but image looks like {kind_info['kind']}",
            )
        axis_refs = _axis_refs(segments, chart)
        if chart["kind"] == "bar":
            extracted = _extract_bar_points(binary, chart, excluded_boxes, config)
        else:
            extracted = _extract_line_points(binary, chart, excluded_boxes, config)
        if normalization is None:
            normalization = _normalization_record(chart, config.coordinate_normalization)

        available = len(extracted)
        allocated = min(available, remaining_quota)
        truncated = available > allocated
        selected = extracted[:allocated]
        remaining_quota -= allocated
        if truncated:
            warnings.append(
                {
                    "code": "E_CHART_POINT_QUOTA_TRUNCATED",
                    "chart_id": chart_id,
                    "available": available,
                    "page_quota": config.page_chart_point_quota,
                    "kept": len(selected),
                }
            )

        points: list[dict[str, Any]] = []
        for point in selected:
            index = point["sequence_position"]
            series = series_by_ordinal.get(index, "s0")
            cache_key = _point_cache_key(
                chart_crop_sha,
                chart_id,
                index,
                series,
                point["pixel"],
                chart,
                config,
                axis_refs,
            )
            previous = previous_points.get((chart_id, index))
            reused = previous is not None and previous.get("cache_key") == cache_key
            if reused:
                reused_points += 1
                point_record = dict(previous)
                point_record["cache_status"] = "reused"
            else:
                recomputed_points += 1
                point_record = {
                    "point_id": f"{chart_id}_P{index + 1:04d}",
                    "sequence_position": index,
                    "series": series,
                    "truth_series_index": (
                        truth_points[index]["index"] if index < len(truth_points) else index
                    ),
                    "pixel": point["pixel"],
                    "normalized": point["normalized"],
                    "value": point["value"],
                    "raw_pixel_ratio": point["raw_pixel_ratio"],
                    "normalization": normalization,
                    "axis_source": {
                        "calibration": "page_truth_axis_calibration",
                        "x_lines": axis_refs["vertical"],
                        "y_lines": axis_refs["horizontal"],
                    },
                    "bar_extent": point.get("bar_extent"),
                    "marker_bbox": point.get("marker_bbox"),
                    "cache_key": cache_key,
                    "cache_status": "recomputed",
                }
            points.append(point_record)

        charts.append(
            {
                "chart_id": chart_id,
                "kind": chart["kind"],
                "region": list(chart["region"]),
                "plot_area": list(chart["plot_area"]),
                "classification": decision,
                "kind_detection": kind_info,
                "axis": dict(chart["axis"]),
                "axis_line_refs": axis_refs,
                "normalization": normalization,
                "quota": {
                    "page_limit": config.page_chart_point_quota,
                    "page_remaining_before": remaining_quota + allocated,
                    "page_remaining_after": remaining_quota,
                    "allocated": allocated,
                    "available": available,
                    "used": len(points),
                    "truncated": truncated,
                },
                "points": points,
            }
        )

    chart_key = _chart_cache_key(
        content_fingerprint, line_key, truth, config, chart_decisions
    )
    return {
        "charts": charts,
        "warnings": warnings,
        "cache_key": chart_key,
        "point_cache": {
            "reused": reused_points,
            "recomputed": recomputed_points,
        },
    }


def chart_cache_key(
    content_fingerprint: str,
    line_key: str,
    truth: Mapping[str, Any],
    config: ParseConfig,
    decisions: Mapping[str, Any],
) -> str:
    return _chart_cache_key(content_fingerprint, line_key, truth, config, decisions)
