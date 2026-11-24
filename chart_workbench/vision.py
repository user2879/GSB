"""Deterministic pixel-level chart primitives.

This module deliberately receives only a grayscale page and calibration
parameters. Ground-truth JSON is never imported here.
"""

from __future__ import annotations

import re
from collections import deque
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from .font import ink_templates


DARK_THRESHOLD = 70
GRAPHIC_MIN = 80
GRAPHIC_MAX = 140
NUMERIC_LABEL = re.compile(r"^-?\d+(?:\.\d+)?$")
_TEMPLATES = ink_templates()


class StageError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


def _runs(bits: np.ndarray) -> List[Tuple[int, int, int]]:
    indices = np.flatnonzero(bits)
    if indices.size == 0:
        return []
    runs: List[Tuple[int, int, int]] = []
    start = int(indices[0])
    previous = int(indices[0])
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index == previous + 1:
            previous = index
            continue
        runs.append((start, previous, previous - start + 1))
        start = index
        previous = index
    runs.append((start, previous, previous - start + 1))
    return runs


def _group_line_candidates(candidates: List[Dict[str, Any]], max_gap: int) -> List[List[Dict[str, Any]]]:
    groups: List[List[Dict[str, Any]]] = []
    for candidate in sorted(candidates, key=lambda item: (item["index"], item["source"])):
        if not groups or candidate["index"] - groups[-1][-1]["index"] > max_gap:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)
    return groups


def _line_groups(mask: np.ndarray, min_length: int, orientation: str, max_gap: int) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []
    source = 0
    cross_count = mask.shape[0] if orientation == "horizontal" else mask.shape[1]
    for index in range(cross_count):
        bits = mask[index, :] if orientation == "horizontal" else mask[:, index]
        for start, end, length in _runs(bits):
            if length < min_length:
                source += 1
                continue
            candidates.append(
                {
                    "orientation": orientation,
                    "index": index,
                    "start": start,
                    "end": end,
                    "length": length,
                    "source": source,
                }
            )
            source += 1

    groups = _group_line_candidates(candidates, max_gap=max_gap)
    result: List[Dict[str, Any]] = []
    for group in groups:
        longest = max(group, key=lambda item: (item["length"], -item["source"]))
        center = int(round(sum(item["index"] for item in group) / float(len(group))))
        result.append(
            {
                "orientation": orientation,
                "position": center,
                "start": min(item["start"] for item in group),
                "end": max(item["end"] for item in group),
                "length": longest["length"],
                "thickness": len(group),
                "sources": [item["source"] for item in sorted(group, key=lambda item: item["index"])],
            }
        )
    result.sort(key=lambda item: (-item["length"], item["position"], item["sources"][0]))
    return result


def detect_axes(gray: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    threshold = float(config["axis_detection_threshold"])
    tolerance = int(config["line_width_tolerance"])
    height, width = gray.shape
    ink = gray < DARK_THRESHOLD
    min_horizontal = max(3, int(round(threshold * width)))
    min_vertical = max(3, int(round(threshold * height)))
    horizontal = _line_groups(ink, min_horizontal, "horizontal", max_gap=max(1, tolerance))
    vertical = _line_groups(ink, min_vertical, "vertical", max_gap=max(1, tolerance))
    if not horizontal or not vertical:
        raise StageError("E_AXIS_NOT_FOUND", "could not find both horizontal and vertical axes")

    horizontal_line = horizontal[0]
    vertical_line = vertical[0]
    x0 = int(vertical_line["position"])
    x1 = int(horizontal_line["end"])
    y0 = int(vertical_line["start"])
    y1 = int(horizontal_line["position"])
    if not (0 <= x0 < x1 < width and 0 <= y0 < y1 < height):
        raise StageError("E_AXIS_NOT_FOUND", "detected axis geometry is not a lower-left plot frame")

    return {
        "plot_box": {"x0": x0, "y0": y0, "x1": x1, "y1": y1},
        "x_axis": horizontal_line,
        "y_axis": vertical_line,
        "candidate_counts": {"horizontal": len(horizontal), "vertical": len(vertical)},
    }


def _tick_groups(
    candidates: List[Dict[str, Any]],
    axis: str,
    tolerance: int,
) -> List[Dict[str, Any]]:
    groups: List[List[Dict[str, Any]]] = []
    max_gap = max(1, tolerance)
    for candidate in candidates:
        if not groups or candidate["index"] - groups[-1][-1]["index"] > max_gap:
            groups.append([candidate])
        else:
            groups[-1].append(candidate)

    merged: List[Dict[str, Any]] = []
    max_thickness = max(2, tolerance + 2)
    for group in groups:
        thickness = group[-1]["index"] - group[0]["index"] + 1
        if thickness > max_thickness:
            continue
        position = int(round(sum(item["index"] for item in group) / float(len(group))))
        merged.append(
            {
                "position": position,
                "length": max(item["length"] for item in group),
                "thickness": thickness,
                "sources": [item["source"] for item in sorted(group, key=lambda item: item["index"])],
            }
        )
    merged.sort(key=lambda item: (item["position"], item["sources"][0] if item["sources"] else 0))
    for index, tick in enumerate(merged):
        tick["id"] = f"{axis}:tick:{index:04d}:{tick['position']}"
    return merged


def _extract_y_tick_candidates(ink: np.ndarray, plot: Dict[str, int], min_length: int) -> List[Dict[str, Any]]:
    x_end = plot["x0"] - 2
    scan_left = max(0, x_end - min_length - 10)
    candidates: List[Dict[str, Any]] = []
    source = 0
    for y in range(plot["y0"], plot["y1"] + 1):
        if x_end < 0 or not ink[y, x_end]:
            continue
        start = x_end
        while start - 1 >= scan_left and ink[y, start - 1]:
            start -= 1
        length = x_end - start + 1
        candidates.append(
            {
                "orientation": "vertical",
                "index": y,
                "start": start,
                "end": x_end,
                "length": length,
                "source": source,
            }
        )
        source += 1
    return [candidate for candidate in candidates if candidate["length"] >= min_length]


def _extract_x_tick_candidates(ink: np.ndarray, plot: Dict[str, int], min_length: int) -> List[Dict[str, Any]]:
    y_start = plot["y1"] + 2
    scan_bottom = min(ink.shape[0] - 1, y_start + min_length + 10)
    candidates: List[Dict[str, Any]] = []
    source = 0
    for x in range(plot["x0"], plot["x1"] + 1):
        if y_start >= ink.shape[0] or not ink[y_start, x]:
            continue
        end = y_start
        while end + 1 <= scan_bottom and ink[end + 1, x]:
            end += 1
        length = end - y_start + 1
        candidates.append(
            {
                "orientation": "horizontal",
                "index": x,
                "start": y_start,
                "end": end,
                "length": length,
                "source": source,
            }
        )
        source += 1
    return [candidate for candidate in candidates if candidate["length"] >= min_length]


def _components(mask: np.ndarray, roi: Tuple[int, int, int, int]) -> List[Dict[str, Any]]:
    x0, y0, x1, y1 = roi
    x0 = max(0, x0)
    y0 = max(0, y0)
    x1 = min(mask.shape[1], x1)
    y1 = min(mask.shape[0], y1)
    if x0 >= x1 or y0 >= y1:
        return []
    crop = mask[y0:y1, x0:x1]
    visited = np.zeros(crop.shape, dtype=bool)
    height, width = crop.shape
    result: List[Dict[str, Any]] = []
    for seed_y in range(height):
        for seed_x in range(width):
            if not crop[seed_y, seed_x] or visited[seed_y, seed_x]:
                continue
            queue = deque([(seed_x, seed_y)])
            visited[seed_y, seed_x] = True
            points: List[Tuple[int, int]] = []
            while queue:
                cx, cy = queue.popleft()
                points.append((cx, cy))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dx == 0 and dy == 0:
                            continue
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height and crop[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            queue.append((nx, ny))
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            bx0, bx1 = min(xs) + x0, max(xs) + x0
            by0, by1 = min(ys) + y0, max(ys) + y0
            result.append(
                {
                    "bbox": [bx0, by0, bx1, by1],
                    "area": len(points),
                    "pixels": [(x + x0, y + y0) for x, y in points],
                }
            )
    result.sort(key=lambda item: (item["bbox"][1], item["bbox"][0], item["area"]))
    for index, component in enumerate(result):
        component["id"] = index
    return result


def _union_find_blocks(components: Sequence[Dict[str, Any]], gap: int = 6) -> List[List[Dict[str, Any]]]:
    count = len(components)
    parents = list(range(count))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    for i in range(count):
        ax0, ay0, ax1, ay1 = components[i]["bbox"]
        for j in range(i + 1, count):
            bx0, by0, bx1, by1 = components[j]["bbox"]
            horizontal_gap = max(0, max(ax0, bx0) - min(ax1, bx1) - 1)
            vertical_gap = max(0, max(ay0, by0) - min(ay1, by1) - 1)
            vertical_reach = max(2, min(ay1 - ay0 + 1, by1 - by0 + 1) // 2 + 1)
            if horizontal_gap <= gap and vertical_gap <= vertical_reach:
                union(i, j)

    groups: Dict[int, List[Dict[str, Any]]] = {}
    for index, component in enumerate(components):
        groups.setdefault(find(index), []).append(component)
    return [groups[key] for key in sorted(groups)]


def _recognize_component(component: Dict[str, Any], gray: np.ndarray) -> Tuple[str, float]:
    x0, y0, x1, y1 = component["bbox"]
    crop = (gray[y0 : y1 + 1, x0 : x1 + 1] < DARK_THRESHOLD).astype(np.uint8) * 255
    source = Image.fromarray(crop, mode="L")
    best_char = ""
    best_score = 2.0
    for char, template in _TEMPLATES:
        target_height = len(template)
        target_width = len(template[0])
        resized = source.resize((target_width, target_height), Image.Resampling.NEAREST)
        bits = np.asarray(resized, dtype=np.uint8) >= 128
        expected = np.asarray(template, dtype=bool)
        score = float(np.mean(bits != expected))
        if score < best_score - 1e-12:
            best_score = score
            best_char = char
    return best_char, best_score


def _ocr_block(block_components: List[Dict[str, Any]], gray: np.ndarray) -> Optional[str]:
    ordered = sorted(
        block_components,
        key=lambda component: (
            component["bbox"][0],
            component["bbox"][1],
            component["bbox"][2],
        ),
    )
    chars: List[str] = []
    for component in ordered:
        char, score = _recognize_component(component, gray)
        if not char or score > 0.26:
            return None
        chars.append(char)
    text = "".join(chars)
    if not NUMERIC_LABEL.match(text):
        return None
    return text


def extract_labels(
    gray: np.ndarray,
    plot: Dict[str, int],
    axis: str,
    tick_max_length: int,
) -> List[Dict[str, Any]]:
    ink = gray < DARK_THRESHOLD
    if axis == "y":
        roi = (
            max(0, plot["x0"] - 220),
            max(0, plot["y0"] - 30),
            max(1, plot["x0"] - tick_max_length - 4),
            min(gray.shape[0], plot["y1"] + 30),
        )
    else:
        roi = (
            max(0, plot["x0"] - 24),
            min(gray.shape[0] - 1, plot["y1"] + tick_max_length + 5),
            min(gray.shape[1], plot["x1"] + 24),
            min(gray.shape[0], plot["y1"] + 110),
        )

    components = []
    for component in _components(ink, roi):
        x0, y0, x1, y1 = component["bbox"]
        width, height = x1 - x0 + 1, y1 - y0 + 1
        if width <= 40 and height <= 30 and component["area"] >= 1:
            components.append(component)

    labels: List[Dict[str, Any]] = []
    for block in _union_find_blocks(components, gap=6):
        text = _ocr_block(block, gray)
        if text is None:
            continue
        bx0 = min(component["bbox"][0] for component in block)
        by0 = min(component["bbox"][1] for component in block)
        bx1 = max(component["bbox"][2] for component in block)
        by1 = max(component["bbox"][3] for component in block)
        labels.append(
            {
                "text": text,
                "value": float(text),
                "bbox": [bx0, by0, bx1, by1],
                "anchor": [int(round((bx0 + bx1) / 2.0)), int(round((by0 + by1) / 2.0))],
                "component_ids": [component["id"] for component in sorted(block, key=lambda item: item["id"])],
            }
        )

    if axis == "y":
        labels.sort(key=lambda item: (item["anchor"][1], item["bbox"][0], item["bbox"][1]))
    else:
        labels.sort(key=lambda item: (item["anchor"][0], item["bbox"][1], item["bbox"][0]))
    for index, label in enumerate(labels):
        label["id"] = f"{axis}:label:{index:04d}"
    return labels


def associate_labels(
    ticks: Sequence[Dict[str, Any]],
    labels: Sequence[Dict[str, Any]],
    axis: str,
) -> List[Dict[str, Any]]:
    coordinate = 0 if axis == "x" else 1
    used: set[str] = set()
    associations: List[Dict[str, Any]] = []
    for tick in ticks:
        candidates = []
        for label in labels:
            if label["id"] in used:
                continue
            distance = abs(int(label["anchor"][coordinate]) - int(tick["position"]))
            candidates.append(
                (
                    distance,
                    int(label["anchor"][coordinate]),
                    int(label["bbox"][1]),
                    int(label["bbox"][0]),
                    label["id"],
                    label,
                )
            )
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[:5])
        best = candidates[0]
        if best[0] > 28:
            continue
        label = best[5]
        used.add(label["id"])
        associations.append(
            {
                "tick_id": tick["id"],
                "label_id": label["id"],
                "distance": int(best[0]),
                "value": float(label["value"]),
                "text": label["text"],
            }
        )
    return associations


def detect_ticks(gray: np.ndarray, config: Dict[str, Any]) -> Dict[str, Any]:
    axes = detect_axes(gray, config)
    plot = axes["plot_box"]
    ink = gray < DARK_THRESHOLD
    min_length = int(config["tick_min_length"])
    tolerance = int(config["line_width_tolerance"])
    y_candidates = _extract_y_tick_candidates(ink, plot, min_length)
    x_candidates = _extract_x_tick_candidates(ink, plot, min_length)
    y_ticks = _tick_groups(y_candidates, "y", tolerance)
    x_ticks = _tick_groups(x_candidates, "x", tolerance)
    if not y_ticks or not x_ticks:
        raise StageError("E_TICKS_NOT_FOUND", "tick extraction returned an empty axis")

    y_max_length = max(tick["length"] for tick in y_ticks)
    x_max_length = max(tick["length"] for tick in x_ticks)
    y_labels = extract_labels(gray, plot, "y", y_max_length)
    x_labels = extract_labels(gray, plot, "x", x_max_length)
    return {
        "plot_box": plot,
        "axis_detection": axes,
        "axes": {
            "x": {
                "ticks": x_ticks,
                "labels": x_labels,
                "associations": associate_labels(x_ticks, x_labels, "x"),
            },
            "y": {
                "ticks": y_ticks,
                "labels": y_labels,
                "associations": associate_labels(y_ticks, y_labels, "y"),
            },
        },
        "parameters": {
            "tick_min_length": min_length,
            "line_width_tolerance": tolerance,
            "axis_detection_threshold": float(config["axis_detection_threshold"]),
        },
    }


def _deduplicate_candidates(candidates: List[Dict[str, Any]], tolerance: int) -> List[Dict[str, Any]]:
    candidates.sort(key=lambda item: (item["pixel"][0], item["pixel"][1], item["component_id"]))
    result: List[Dict[str, Any]] = []
    for candidate in candidates:
        if result and abs(candidate["pixel"][0] - result[-1]["pixel"][0]) <= tolerance:
            continue
        result.append(candidate)
    return result


def detect_graphic_candidates(gray: np.ndarray, plot: Dict[str, int], line_width_tolerance: int) -> Dict[str, Any]:
    graphic = (gray >= GRAPHIC_MIN) & (gray <= GRAPHIC_MAX)
    roi = (plot["x0"] + 2, plot["y0"] + 2, plot["x1"] - 2, plot["y1"] - 2)
    components = [component for component in _components(graphic, roi) if component["area"] >= 20]

    bar_candidates: List[Dict[str, Any]] = []
    marker_candidates: List[Dict[str, Any]] = []
    for component in components:
        x0, y0, x1, y1 = component["bbox"]
        width, height = x1 - x0 + 1, y1 - y0 + 1
        area_ratio = component["area"] / float(max(1, width * height))
        if width >= 12 and height >= 8 and y1 >= plot["y1"] - 10 and area_ratio >= 0.45:
            bar_candidates.append(
                {
                    "component_id": component["id"],
                    "bbox": component["bbox"],
                    "pixel": [int(round((x0 + x1) / 2.0)), int(y0)],
                    "area": component["area"],
                }
            )
        elif 3 <= width <= 18 and 3 <= height <= 18:
            marker_candidates.append(
                {
                    "component_id": component["id"],
                    "bbox": component["bbox"],
                    "pixel": [int(round((x0 + x1) / 2.0)), int(round((y0 + y1) / 2.0))],
                    "area": component["area"],
                }
            )

    tie_tolerance = max(1, line_width_tolerance)
    if bar_candidates:
        candidates = _deduplicate_candidates(bar_candidates, tie_tolerance)
        chart_type = "bar"
    else:
        candidates = _deduplicate_candidates(marker_candidates, tie_tolerance)
        chart_type = "line"
    return {
        "chart_type": chart_type,
        "candidates": candidates,
        "component_count": len(components),
    }
