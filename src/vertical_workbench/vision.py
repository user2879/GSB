from __future__ import annotations

from collections import deque
from typing import Any, Iterable

import numpy as np

from .config import WorkbenchConfig
from .hashing import sha256_bytes


def window_rects(width: int, height: int, size: int, stride: int) -> list[dict[str, int]]:
    xs = list(range(0, max(1, width - size + 1), stride))
    ys = list(range(0, max(1, height - size + 1), stride))
    if xs[-1] != width - size:
        xs.append(width - size)
    if ys[-1] != height - size:
        ys.append(height - size)
    rects = [{"x": x, "y": y, "w": size, "h": size} for y in ys for x in xs]
    rects.sort(key=lambda r: (r["y"], r["x"]))
    return rects


def _bp(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        return 0
    return int(numerator) * 10_000 // int(denominator)


def compute_window(
    image_id: str,
    pixels: np.ndarray,
    rect: dict[str, int],
    config: WorkbenchConfig,
) -> dict[str, Any]:
    """Read exactly this window's covered pixel rectangle and score it."""
    x, y, size = rect["x"], rect["y"], rect["w"]
    patch = pixels[y : y + size, x : x + size]
    binary = patch < config.ink_threshold
    ink_count = int(binary.sum())
    area = size * size
    density_bp = _bp(ink_count, area)

    col_counts = binary.sum(axis=0).astype(np.int64)
    row_counts = binary.sum(axis=1).astype(np.int64)
    mean_col = int(col_counts.sum() // size)
    valley_columns = int((col_counts <= max(1, mean_col // 3)).sum()) if ink_count else size
    valley_bp = _bp(valley_columns, size)

    gy = np.abs(np.diff(patch.astype(np.int16), axis=0)).sum()
    gx = np.abs(np.diff(patch.astype(np.int16), axis=1)).sum()
    gradient_bp = min(10_000, _bp(int(gx + gy), area) * 20)

    local_fingerprint = sha256_bytes(patch.tobytes() + f"{x},{y},{size}".encode())
    seed_int = int(local_fingerprint[:16], 16)
    projection = 0
    max_projection = 0
    for index, count in enumerate(col_counts):
        coefficient = 1 + ((seed_int >> ((index * 7) % 60)) & 15)
        projection += int(count) * coefficient
        max_projection += max(1, int(col_counts.max())) * coefficient
    mapping_bp = _bp(projection, max_projection)

    score = (
        4_000 * density_bp
        + 2_000 * valley_bp
        + 2_000 * gradient_bp
        + 2_000 * mapping_bp
    ) // 10_000
    score = max(0, min(10_000, int(score)))
    equal_boundary = score == config.candidate_score_threshold
    candidate = ink_count >= config.min_component_area and score >= config.candidate_score_threshold
    return {
        "window_id": f"{image_id}:w{x:04d}:y{y:04d}:s{size}",
        "image_id": image_id,
        **rect,
        "ink_count": ink_count,
        "density_bp": density_bp,
        "valley_bp": valley_bp,
        "gradient_bp": gradient_bp,
        "mapping_bp": mapping_bp,
        "score": score,
        "score_threshold": config.candidate_score_threshold,
        "score_boundary_equal": equal_boundary,
        "boundary_flag": equal_boundary,
        "candidate": candidate,
        "local_fingerprint": local_fingerprint,
    }


def connected_components(mask: np.ndarray) -> list[dict[str, Any]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict[str, Any]] = []
    for start_y in range(height):
        for start_x in range(width):
            if not mask[start_y, start_x] or seen[start_y, start_x]:
                continue
            queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
            seen[start_y, start_x] = True
            coords: list[tuple[int, int]] = []
            sum_x = sum_y = 0
            x_min = width
            x_max = -1
            y_min = height
            y_max = -1
            while queue:
                py, px = queue.popleft()
                coords.append((py, px))
                sum_x += px
                sum_y += py
                x_min = min(x_min, px)
                x_max = max(x_max, px)
                y_min = min(y_min, py)
                y_max = max(y_max, py)
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = py + dy, px + dx
                        if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not seen[ny, nx]:
                            seen[ny, nx] = True
                            queue.append((ny, nx))
            area = len(coords)
            components.append(
                {
                    "component_id": f"raw-{len(components) + 1:03d}",
                    "coords": coords,
                    "area": area,
                    "x_min": int(x_min),
                    "x_max": int(x_max + 1),
                    "y_min": int(y_min),
                    "y_max": int(y_max + 1),
                    "center_x_q": sum_x << 8 // area if False else sum_x,
                    "sum_x": int(sum_x),
                    "sum_y": int(sum_y),
                }
            )
    components.sort(key=lambda c: (c["y_min"], c["x_min"]))
    for index, component in enumerate(components, start=1):
        component["component_id"] = f"raw-{index:03d}"
    return components


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def _gap(a: dict[str, Any], b: dict[str, Any]) -> tuple[int, int]:
    horizontal = max(0, max(a["x_min"], b["x_min"]) - min(a["x_max"], b["x_max"]))
    vertical = max(0, max(a["y_min"], b["y_min"]) - min(a["y_max"], b["y_max"]))
    return horizontal, vertical


def merge_components(
    components: list[dict[str, Any]],
    candidate_window_ids: Iterable[str],
    config: WorkbenchConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Compare every raw pair once and merge same-color split fragments."""
    window_set = set(candidate_window_ids)
    uf = UnionFind(len(components))
    records: list[dict[str, Any]] = []
    comparisons = 0
    merge_number = 0
    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            comparisons += 1
            a, b = components[i], components[j]
            horizontal_gap, vertical_gap = _gap(a, b)
            small_fragment = min(a["area"], b["area"]) <= config.area_boundary_threshold
            close_enough = max(horizontal_gap, vertical_gap) <= config.merge_gap
            vertical_stroke = (
                horizontal_gap <= 6
                and vertical_gap <= config.merge_gap
                and a["x_min"] < b["x_max"]
                and b["x_min"] < a["x_max"] + config.merge_gap
            )
            occluded_fragment = small_fragment and close_enough
            window_anchor = bool(window_set) and close_enough and vertical_gap <= config.merge_gap

            reason = None
            basis = None
            if vertical_stroke:
                reason = "MERGE_VERTICAL_STROKE"
                basis = {"horizontal_gap": horizontal_gap, "vertical_gap": vertical_gap}
            elif occluded_fragment:
                reason = "MERGE_OCCLUDED_SAME_COLOR_FRAGMENT"
                basis = {
                    "horizontal_gap": horizontal_gap,
                    "vertical_gap": vertical_gap,
                    "smaller_area": min(a["area"], b["area"]),
                    "area_threshold": config.area_boundary_threshold,
                    "tie_rule": "equal_area_is_small_fragment",
                }
            elif window_anchor:
                reason = "MERGE_WINDOW_ANCHOR"
                basis = {"horizontal_gap": horizontal_gap, "vertical_gap": vertical_gap}

            if reason:
                uf.union(i, j)
                merge_number += 1
                records.append(
                    {
                        "merge_id": f"merge-{merge_number:03d}",
                        "raw_component_ids": [a["component_id"], b["component_id"]],
                        "reason": reason,
                        "basis": basis,
                        "same_color": True,
                        "candidate_window_count": len(window_set),
                    }
                )

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, component in enumerate(components):
        groups.setdefault(uf.find(index), []).append(component)

    domains: list[dict[str, Any]] = []
    for group in groups.values():
        coords: list[tuple[int, int]] = []
        source_ids = []
        for component in group:
            coords.extend(component["coords"])
            source_ids.append(component["component_id"])
        ys = [p[0] for p in coords]
        xs = [p[1] for p in coords]
        x_min, x_max = min(xs), max(xs) + 1
        y_min, y_max = min(ys), max(ys) + 1
        area = len(coords)
        domains.append(
            {
                "source_component_ids": source_ids,
                "coords": coords,
                "area": area,
                "x_min": int(x_min),
                "x_max": int(x_max),
                "y_min": int(y_min),
                "y_max": int(y_max),
                "merge_records": [
                    r for r in records if any(cid in r["raw_component_ids"] for cid in source_ids)
                ],
                "split_records": [],
                "area_boundary_equal": area == config.area_boundary_threshold,
                "boundary_flag": area == config.area_boundary_threshold,
            }
        )
    domains.sort(key=lambda d: (d["y_min"], d["x_min"]))
    return domains, records, comparisons


def split_domain_by_valley(domain: dict[str, Any], config: WorkbenchConfig) -> list[dict[str, Any]]:
    height = domain["y_max"] - domain["y_min"]
    if height <= config.split_height:
        return [domain]
    x0, x1 = domain["x_min"], domain["x_max"]
    y0, y1 = domain["y_min"], domain["y_max"]
    row_counts = np.zeros(height, dtype=np.int64)
    for py, _px in domain["coords"]:
        if y0 <= py < y1:
            row_counts[py - y0] += 1
    middle = height // 2
    search0 = max(0, middle - height // 4)
    search1 = min(height, middle + height // 4)
    valley = search0 + int(np.argmin(row_counts[search0:search1]))
    if int(row_counts[valley]) > max(2, config.area_boundary_threshold // 3):
        return [domain]

    upper_coords = [p for p in domain["coords"] if p[0] < y0 + valley]
    lower_coords = [p for p in domain["coords"] if p[0] >= y0 + valley]
    if not upper_coords or not lower_coords:
        return [domain]
    split_records = domain["split_records"] + [
        {
            "reason": "SPLIT_VERTICAL_PROJECTION_VALLEY",
            "at_relative_row": int(valley),
            "valley_ink": int(row_counts[valley]),
            "old_bbox": [x0, y0, x1, y1],
        }
    ]
    result = []
    for coords in (upper_coords, lower_coords):
        ys = [p[0] for p in coords]
        xs = [p[1] for p in coords]
        item = dict(domain)
        item.update(
            {
                "coords": coords,
                "area": len(coords),
                "x_min": min(xs),
                "x_max": max(xs) + 1,
                "y_min": min(ys),
                "y_max": max(ys) + 1,
                "split_records": split_records,
            }
        )
        result.append(item)
    return result
