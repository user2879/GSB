"""Deterministic projection-window block detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .errors import E_BLOCK_BOUNDS, WorkbenchError
from .models import BBox, Block, round_metric


DETECT_STAGE_VERSION = 1


@dataclass
class _Component:
    x0: int
    y0: int
    x1: int
    y1: int
    area: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        parent = self.parent
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        if self.rank[root_left] < self.rank[root_right]:
            root_left, root_right = root_right, root_left
        self.parent[root_right] = root_left
        if self.rank[root_left] == self.rank[root_right]:
            self.rank[root_left] += 1


def _row_runs(row: np.ndarray) -> Iterable[tuple[int, int]]:
    indices = np.flatnonzero(row)
    if indices.size == 0:
        return
    start = int(indices[0])
    previous = start
    for raw_index in indices[1:]:
        index = int(raw_index)
        if index != previous + 1:
            yield start, previous
            start = index
        previous = index
    yield start, previous


def _connected_components(mask: np.ndarray) -> list[_Component]:
    """Eight-connected run union with deterministic row scanning."""

    parent: list[int] = []
    component_area: list[int] = []
    component_bounds: list[tuple[int, int, int, int]] = []
    union_pairs: list[tuple[int, int]] = []
    previous_runs: list[tuple[int, int, int]] = []

    for y in range(mask.shape[0]):
        current_runs: list[tuple[int, int, int]] = []
        for start, end in _row_runs(mask[y]):
            component_id = len(parent)
            parent.append(component_id)
            component_area.append(end - start + 1)
            component_bounds.append((start, y, end + 1, y + 1))
            current_runs.append((start, end, component_id))

        for start, end, component_id in current_runs:
            for prev_start, prev_end, prev_id in previous_runs:
                if start <= prev_end + 1 and end >= prev_start - 1:
                    union_pairs.append((component_id, prev_id))
        previous_runs = current_runs

    # Replay accumulated row unions into one disjoint set.
    components = _DisjointSet(len(parent))
    for left, right in union_pairs:
        components.union(left, right)

    aggregated: dict[int, list[int]] = {}
    for index in range(len(parent)):
        root = components.find(index)
        x0, y0, x1, y1 = component_bounds[index]
        if root not in aggregated:
            aggregated[root] = [x0, y0, x1, y1, component_area[index]]
        else:
            bounds = aggregated[root]
            bounds[0] = min(bounds[0], x0)
            bounds[1] = min(bounds[1], y0)
            bounds[2] = max(bounds[2], x1)
            bounds[3] = max(bounds[3], y1)
            bounds[4] += component_area[index]

    return [
        _Component(bounds[0], bounds[1], bounds[2], bounds[3], bounds[4])
        for _, bounds in sorted(aggregated.items())
    ]


def _contains(outer: _Component, inner: _Component) -> bool:
    return (
        outer.x0 <= inner.x0
        and outer.y0 <= inner.y0
        and outer.x1 >= inner.x1
        and outer.y1 >= inner.y1
        and outer.area >= inner.area
    )


def _overlap_length(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0))


def _cluster_components(
    components: list[_Component], horizontal_window: int, vertical_window: int
) -> list[_Component]:
    if not components:
        return []
    ds = _DisjointSet(len(components))

    # Containment attaches glyph fragments and image texture to their enclosing
    # block before projection-window merging is attempted.
    for child_index, child in enumerate(components):
        best: int | None = None
        best_area: int | None = None
        for outer_index, outer in enumerate(components):
            if outer_index == child_index or not _contains(outer, child):
                continue
            if outer.area <= child.area:
                continue
            if best_area is None or outer.area < best_area:
                best = outer_index
                best_area = outer.area
        if best is not None:
            ds.union(child_index, best)

    # Projection windows bridge detached bars/characters and line fragments.
    for _ in range(6):
        groups = _aggregate_groups(components, ds)
        before = len(groups)
        for left_index in range(len(groups)):
            for right_index in range(left_index + 1, len(groups)):
                if ds.find(groups[left_index][0]) == ds.find(groups[right_index][0]):
                    continue
                left = groups[left_index][1]
                right = groups[right_index][1]
                overlap_y = _overlap_length(left.y0, left.y1, right.y0, right.y1)
                min_height = max(1, min(left.height, right.height))
                vertical_ratio = overlap_y / min_height
                horizontal_gap = max(0, max(left.x0, right.x0) - min(left.x1, right.x1))
                overlap_x = _overlap_length(left.x0, left.x1, right.x0, right.x1)
                min_width = max(1, min(left.width, right.width))
                horizontal_ratio = overlap_x / min_width
                vertical_gap = max(0, max(left.y0, right.y0) - min(left.y1, right.y1))
                if (
                    vertical_ratio >= 0.5
                    and horizontal_gap <= horizontal_window
                ) or (
                    horizontal_ratio >= 0.35
                    and vertical_gap <= vertical_window
                ):
                    ds.union(groups[left_index][0], groups[right_index][0])
        groups = _aggregate_groups(components, ds)
        if len(groups) == before:
            break

    return [group[1] for group in _aggregate_groups(components, ds)]


def _aggregate_groups(
    components: list[_Component], ds: _DisjointSet
) -> list[tuple[int, _Component]]:
    aggregates: dict[int, list[int]] = {}
    for index, component in enumerate(components):
        root = ds.find(index)
        if root not in aggregates:
            aggregates[root] = [
                component.x0,
                component.y0,
                component.x1,
                component.y1,
                component.area,
            ]
        else:
            aggregate = aggregates[root]
            aggregate[0] = min(aggregate[0], component.x0)
            aggregate[1] = min(aggregate[1], component.y0)
            aggregate[2] = max(aggregate[2], component.x1)
            aggregate[3] = max(aggregate[3], component.y1)
            aggregate[4] += component.area
    groups = [
        (root, _Component(values[0], values[1], values[2], values[3], values[4]))
        for root, values in aggregates.items()
    ]
    groups.sort(key=lambda item: (item[1].y0, item[1].x0, item[1].x1, item[1].y1))
    return groups


def _projection_periodicity(mask: np.ndarray, bbox: BBox) -> float:
    inset = 2
    x0 = bbox.x0 + inset
    y0 = bbox.y0 + inset
    x1 = bbox.x1 - inset
    y1 = bbox.y1 - inset
    if x1 <= x0 or y1 <= y0:
        x0, y0, x1, y1 = bbox.x0, bbox.y0, bbox.x1, bbox.y1
    crop = mask[y0:y1, x0:x1]
    if crop.size == 0:
        return 0.0
    row_ink = crop.sum(axis=1).astype(np.float64)
    active = row_ink > max(1.0, float(crop.shape[1]) * 0.02)
    transitions = int(np.count_nonzero(active[1:] != active[:-1]))
    return round_metric(transitions / max(1, 2 * crop.shape[0]))


def detect_blocks(
    gray: np.ndarray, config: dict[str, Any]
) -> tuple[list[Block], dict[str, Any]]:
    threshold = int(config["binarization_threshold"])
    minimum = config["block_min_size"]
    window = config["projection_window"]
    mask = gray < threshold

    raw_components = _connected_components(mask)
    candidate_components = _cluster_components(
        raw_components,
        int(window["horizontal"]),
        int(window["vertical"]),
    )

    image_height, image_width = mask.shape
    filtered: list[tuple[BBox, float, float]] = []
    rejected_too_small = 0
    rejected_out_of_bounds = 0
    for component in candidate_components:
        bbox = BBox(component.x0, component.y0, component.x1, component.y1)
        if (
            bbox.x0 < 0
            or bbox.y0 < 0
            or bbox.x1 > image_width
            or bbox.y1 > image_height
            or bbox.x1 <= bbox.x0
            or bbox.y1 <= bbox.y0
        ):
            rejected_out_of_bounds += 1
            continue
        if bbox.width < int(minimum["width"]) or bbox.height < int(minimum["height"]):
            rejected_too_small += 1
            continue
        crop = mask[bbox.y0 : bbox.y1, bbox.x0 : bbox.x1]
        density = round_metric(float(crop.sum()) / float(bbox.area))
        periodicity = _projection_periodicity(mask, bbox)
        filtered.append((bbox, density, periodicity))

    filtered.sort(key=lambda item: (item[0].y0, item[0].x0, item[0].x1, item[0].y1))
    blocks = [
        Block(
            block_id=f"b{index:04d}",
            bbox=bbox,
            ink_density=density,
            projection_periodicity=periodicity,
        )
        for index, (bbox, density, periodicity) in enumerate(filtered, start=1)
    ]
    diagnostics = {
        "threshold": threshold,
        "projection_window": {
            "horizontal": int(window["horizontal"]),
            "vertical": int(window["vertical"]),
        },
        "raw_component_count": len(raw_components),
        "clustered_component_count": len(candidate_components),
        "detected_block_count": len(blocks),
        "rejected_too_small": rejected_too_small,
        "rejected_out_of_bounds": rejected_out_of_bounds,
    }
    return blocks, diagnostics


def validate_block_bounds(blocks: list[Block], width: int, height: int, stage: str) -> None:
    seen: set[tuple[int, int, int, int]] = set()
    for block in blocks:
        bbox = block.bbox
        if (
            bbox.x0 < 0
            or bbox.y0 < 0
            or bbox.x1 > width
            or bbox.y1 > height
            or bbox.x1 <= bbox.x0
            or bbox.y1 <= bbox.y0
        ):
            raise WorkbenchError(
                E_BLOCK_BOUNDS,
                "block bounding box is outside the image or has non-positive area",
                stage,
                details={"block_id": block.block_id, "bbox": bbox.to_list()},
            )
        key = (bbox.x0, bbox.y0, bbox.x1, bbox.y1)
        if key in seen:
            raise WorkbenchError(
                E_BLOCK_BOUNDS,
                "duplicate block bounding box",
                stage,
                details={"block_id": block.block_id, "bbox": bbox.to_list()},
            )
        seen.add(key)
