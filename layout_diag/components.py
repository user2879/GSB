"""Connected-component split/merge and candidate-region association."""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .config import EffectiveConfig


def run_component_stage(
    image_name: str,
    grayscale: np.ndarray,
    effective: EffectiveConfig,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    ink = grayscale < effective.cc.ink_threshold
    raw = _label_raw_components(ink)
    raw.sort(key=lambda item: (item["y0"], item["x0"], item["y1"], item["x1"], item["area"]))
    for index, component in enumerate(raw):
        component["raw_id"] = f"r{index:06d}"

    fragments: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    median_height = _median([component["y1"] - component["y0"] + 1 for component in raw if component["area"] >= effective.cc.min_area])
    for component in raw:
        pieces = _maybe_split(component, effective, median_height)
        if len(pieces) > 1:
            split_records.append(
                {
                    "source_component_id": component["raw_id"],
                    "result_ids": [piece["fragment_id"] for piece in pieces],
                    "basis": "horizontal_projection_valley",
                    "source_bbox": [component["x0"], component["y0"], component["x1"], component["y1"]],
                    "threshold": round(effective.cc.split_ratio, 6),
                }
            )
        fragments.extend(pieces)

    fragments.sort(key=lambda item: (item["y0"], item["x0"], item["y1"], item["x1"], item["fragment_id"]))
    parent = list(range(len(fragments)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> bool:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            return False
        if left_root < right_root:
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root
        return True

    pairs: list[tuple[int, int, int, int, int]] = []
    for left_index in range(len(fragments)):
        left = fragments[left_index]
        for right_index in range(left_index + 1, len(fragments)):
            right = fragments[right_index]
            gap_x = max(0, left["x0"] - right["x1"] - 1, right["x0"] - left["x1"] - 1)
            gap_y = max(0, left["y0"] - right["y1"] - 1, right["y0"] - left["y1"] - 1)
            distance = max(gap_x, gap_y)
            if distance <= effective.cc.merge_gap:
                pairs.append((distance, left_index, right_index, gap_x, gap_y))
    pairs.sort(key=lambda item: (item[0], fragments[item[1]]["fragment_id"], fragments[item[2]]["fragment_id"]))

    merge_records: list[dict[str, Any]] = []
    for distance, left_index, right_index, gap_x, gap_y in pairs:
        if not union(left_index, right_index):
            continue
        left, right = fragments[left_index], fragments[right_index]
        merge_records.append(
            {
                "fragment_ids": sorted([left["fragment_id"], right["fragment_id"]]),
                "result_group_id": "",
                "basis": [
                    f"chebyshev_bbox_gap={distance}<=merge_gap={effective.cc.merge_gap}",
                    f"horizontal_gap={gap_x}",
                    f"vertical_gap={gap_y}",
                ],
            }
        )

    grouped: dict[int, list[dict[str, Any]]] = {}
    for index, fragment in enumerate(fragments):
        grouped.setdefault(find(index), []).append(fragment)
    groups = [_combine_fragments(members) for members in grouped.values()]
    groups.sort(key=lambda item: (item["y0"], item["x0"], item["y1"], item["x1"], item["fragment_ids"]))

    accepted: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for group in groups:
        reasons: list[str] = []
        if group["area"] < effective.cc.min_area:
            reasons.append(f"area={group['area']}<min_area={effective.cc.min_area}")
        width = group["x1"] - group["x0"] + 1
        height = group["y1"] - group["y0"] + 1
        aspect = max(width, height) / float(max(1, min(width, height)))
        if aspect > effective.cc.max_aspect:
            reasons.append(f"aspect={aspect:.6f}>max_aspect={effective.cc.max_aspect:.6f}")
        if reasons:
            filtered.append(
                {
                    "fragment_ids": group["fragment_ids"],
                    "bbox": [group["x0"], group["y0"], group["x1"], group["y1"]],
                    "area": group["area"],
                    "reasons": reasons,
                }
            )
            continue
        accepted.append(group)

    associated: list[dict[str, Any]] = []
    for group in accepted:
        candidate_id, window_id, overlap = _associate_candidate(group, candidates)
        if not candidate_id or overlap <= 0:
            filtered.append(
                {
                    "fragment_ids": group["fragment_ids"],
                    "bbox": [group["x0"], group["y0"], group["x1"], group["y1"]],
                    "area": group["area"],
                    "reasons": ["no_overlap_with_coarse_candidate"],
                }
            )
            continue
        group["candidate_id"] = candidate_id
        group["window_id"] = window_id
        group["candidate_overlap"] = overlap
        associated.append(group)

    associated.sort(key=lambda item: (item["y0"], item["x0"], item["y1"], item["x1"], item["fragment_ids"]))
    for index, group in enumerate(associated):
        group["component_id"] = f"ccm{index:06d}"
        group["aspect"] = round(
            max(group["x1"] - group["x0"] + 1, group["y1"] - group["y0"] + 1)
            / float(max(1, min(group["x1"] - group["x0"] + 1, group["y1"] - group["y0"] + 1))),
            6,
        )

    group_id_by_fragment = {
        fragment_id: group["component_id"]
        for group in associated
        for fragment_id in group["fragment_ids"]
    }
    for record in merge_records:
        record["result_group_id"] = group_id_by_fragment.get(record["fragment_ids"][0], "filtered")

    return {
        "image": image_name,
        "merge_gap": effective.cc.merge_gap,
        "min_area": effective.cc.min_area,
        "max_aspect": effective.cc.max_aspect,
        "ink_threshold": effective.cc.ink_threshold,
        "raw_component_count": len(raw),
        "fragment_count": len(fragments),
        "component_count": len(associated),
        "components": [
            {
                key: value
                for key, value in group.items()
                if key not in {"pixel_pairs"}
            }
            | {"pixels": group["pixel_pairs"]}
            for group in associated
        ],
        "filtered": filtered,
        "split_records": split_records,
        "merge_records": merge_records,
    }


def _label_raw_components(ink: np.ndarray) -> list[dict[str, Any]]:
    height, width = ink.shape
    visited = np.zeros_like(ink, dtype=bool)
    components: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            if not ink[y, x] or visited[y, x]:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pixels: list[list[int]] = []
            while queue:
                current_y, current_x = queue.popleft()
                pixels.append([current_x, current_y])
                for delta_y in (-1, 0, 1):
                    for delta_x in (-1, 0, 1):
                        if delta_x == 0 and delta_y == 0:
                            continue
                        next_y = current_y + delta_y
                        next_x = current_x + delta_x
                        if 0 <= next_y < height and 0 <= next_x < width and ink[next_y, next_x] and not visited[next_y, next_x]:
                            visited[next_y, next_x] = True
                            queue.append((next_y, next_x))
            xs = [point[0] for point in pixels]
            ys = [point[1] for point in pixels]
            components.append(
                {
                    "x0": min(xs),
                    "y0": min(ys),
                    "x1": max(xs),
                    "y1": max(ys),
                    "area": len(pixels),
                    "centroid_x": sum(xs) / len(xs),
                    "centroid_y": sum(ys) / len(ys),
                    "pixel_pairs": pixels,
                }
            )
    return components


def _median(values: list[int]) -> float:
    if not values:
        return 1.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _maybe_split(component: dict[str, Any], effective: EffectiveConfig, median_height: float) -> list[dict[str, Any]]:
    height = component["y1"] - component["y0"] + 1
    if height < 2 or height <= effective.cc.split_ratio * median_height:
        return [_fragment_from_pixels(component, component["pixel_pairs"], f"{component['raw_id']}s0")]

    row_counts: dict[int, int] = {}
    for _, y in component["pixel_pairs"]:
        row_counts[y] = row_counts.get(y, 0) + 1
    low = component["y0"] + max(1, height // 4)
    high = component["y1"] - max(1, height // 4)
    candidates = [(row_counts.get(y, 0), y) for y in range(low, high + 1)]
    if not candidates:
        return [_fragment_from_pixels(component, component["pixel_pairs"], f"{component['raw_id']}s0")]
    count, split_y = min(candidates)
    average = component["area"] / float(height)
    if count > max(1.0, average * 0.35):
        return [_fragment_from_pixels(component, component["pixel_pairs"], f"{component['raw_id']}s0")]
    upper = [point for point in component["pixel_pairs"] if point[1] <= split_y]
    lower = [point for point in component["pixel_pairs"] if point[1] > split_y]
    if not upper or not lower:
        return [_fragment_from_pixels(component, component["pixel_pairs"], f"{component['raw_id']}s0")]
    return [
        _fragment_from_pixels(component, upper, f"{component['raw_id']}s0"),
        _fragment_from_pixels(component, lower, f"{component['raw_id']}s1"),
    ]


def _fragment_from_pixels(component: dict[str, Any], pixels: list[list[int]], fragment_id: str) -> dict[str, Any]:
    xs = [point[0] for point in pixels]
    ys = [point[1] for point in pixels]
    return {
        "fragment_id": fragment_id,
        "x0": min(xs),
        "y0": min(ys),
        "x1": max(xs),
        "y1": max(ys),
        "area": len(pixels),
        "centroid_x": sum(xs) / len(xs),
        "centroid_y": sum(ys) / len(ys),
        "pixel_pairs": pixels,
        "source_raw_id": component["raw_id"],
    }


def _combine_fragments(fragments: list[dict[str, Any]]) -> dict[str, Any]:
    pixels: list[list[int]] = []
    for fragment in fragments:
        pixels.extend(fragment["pixel_pairs"])
    xs = [point[0] for point in pixels]
    ys = [point[1] for point in pixels]
    return {
        "fragment_ids": sorted(fragment["fragment_id"] for fragment in fragments),
        "x0": min(xs),
        "y0": min(ys),
        "x1": max(xs),
        "y1": max(ys),
        "area": len(pixels),
        "centroid_x": sum(xs) / len(xs),
        "centroid_y": sum(ys) / len(ys),
        "pixel_pairs": pixels,
    }


def _associate_candidate(group: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[str, str, float]:
    group_area = (group["x1"] - group["x0"] + 1) * (group["y1"] - group["y0"] + 1)
    best: tuple[float, str, dict[str, Any]] | None = None
    center_x = (group["x0"] + group["x1"]) / 2.0
    center_y = (group["y0"] + group["y1"]) / 2.0
    for candidate in candidates:
        x0 = max(group["x0"], candidate["x0"])
        y0 = max(group["y0"], candidate["y0"])
        x1 = min(group["x1"], candidate["x1"])
        y1 = min(group["y1"], candidate["y1"])
        intersection = max(0, x1 - x0 + 1) * max(0, y1 - y0 + 1)
        containment = intersection / float(max(1, group_area))
        if best is None or containment > best[0] or (containment == best[0] and candidate["candidate_id"] < best[1]):
            best = (containment, candidate["candidate_id"], candidate)
    if best is not None and best[0] > 0:
        return best[1], best[2]["window_id"], round(best[0], 6)
    if not candidates:
        return "", "", 0.0
    nearest = min(
        candidates,
        key=lambda candidate: (
            (center_x - (candidate["x0"] + candidate["x1"]) / 2.0) ** 2
            + (center_y - (candidate["y0"] + candidate["y1"]) / 2.0) ** 2,
            candidate["candidate_id"],
        ),
    )
    return nearest["candidate_id"], nearest["window_id"], 0.0
