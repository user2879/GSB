"""Pair-region fusion, global pixel fusion and connected-area postprocessing."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .classify import TileResult
from .inputs import FusionStrategy, IndexData, SegmentationConfig, TileSpec
from .util import canonical_hash, rounded_float


@dataclass(slots=True)
class RegionResult:
    pair_id: str
    tile_ids: tuple[str, str]
    overlap_bounds: tuple[int, int, int, int]
    fingerprint: str
    region_version: str
    fusion_version: str
    votes: list[dict[str, Any]]
    winner_histogram: dict[str, int]

    def manifest_entry(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "tile_ids": list(self.tile_ids),
            "overlap_bounds": list(self.overlap_bounds),
            "fingerprint": self.fingerprint,
            "region_version": self.region_version,
            "fusion_version": self.fusion_version,
        }


@dataclass(slots=True)
class GlobalResult:
    bounds: tuple[int, int, int, int]
    classes: np.ndarray
    class_histogram: dict[str, int]
    overlap_trace: list[dict[str, Any]]
    postprocess_records: list[dict[str, Any]]
    postprocess_components: list[dict[str, Any]]
    fusion_version: str

    @property
    def summary(self) -> dict[str, Any]:
        height, width = self.classes.shape
        return {
            "bounds": list(self.bounds),
            "shape": [height, width],
            "class_histogram": self.class_histogram,
            "overlap_pixel_count": len(self.overlap_trace),
            "postprocess_backfill_count": len(self.postprocess_records),
        }


def overlap_bounds(spec_a: TileSpec, spec_b: TileSpec) -> tuple[int, int, int, int] | None:
    x0 = max(spec_a.bounds[0], spec_b.bounds[0])
    y0 = max(spec_a.bounds[1], spec_b.bounds[1])
    x1 = min(spec_a.bounds[2], spec_b.bounds[2])
    y1 = min(spec_a.bounds[3], spec_b.bounds[3])
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)


def iter_adjacent_pairs(index: IndexData) -> list[tuple[TileSpec, TileSpec, tuple[int, int, int, int]]]:
    by_id = index.by_id
    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[TileSpec, TileSpec, tuple[int, int, int, int]]] = []
    for spec in sorted(index.tiles, key=lambda item: item.order_key):
        for neighbor_id in spec.neighbors:
            key = tuple(sorted((spec.tile_id, neighbor_id)))
            if key in seen:
                continue
            seen.add(key)
            other = by_id[neighbor_id]
            bounds = overlap_bounds(spec, other)
            if bounds is None:
                continue
            first, second = (spec, other) if spec.order_key <= other.order_key else (other, spec)
            pairs.append((first, second, bounds))
    pairs.sort(key=lambda item: (item[0].row, item[0].col, item[1].row, item[1].col, item[0].tile_id))
    return pairs


def pair_id(spec_a: TileSpec, spec_b: TileSpec) -> str:
    ordered = sorted((spec_a, spec_b), key=lambda spec: spec.order_key)
    return f"{ordered[0].tile_id}|{ordered[1].tile_id}"


def region_fingerprint(
    tile_a: TileResult,
    tile_b: TileResult,
    bounds: tuple[int, int, int, int],
    strategy: FusionStrategy,
) -> str:
    return canonical_hash(
        {
            "tiles": {
                tile_a.tile_id: tile_a.cache_key,
                tile_b.tile_id: tile_b.cache_key,
            },
            "overlap_bounds": list(bounds),
            "fusion_version": strategy.version,
            "weight_power": strategy.weight_power,
            "scale": strategy.scale,
            "tie_break": strategy.tie_break,
        },
        prefix="fusion-region-v1|",
    )


def _pixel_weight(local_x: int, local_y: int, width: int, height: int, strategy: FusionStrategy) -> float:
    distance = min(local_x + 1, local_y + 1, width - local_x, height - local_y)
    return float(strategy.scale * (float(distance) ** strategy.weight_power))


def _vote(
    sources: Iterable[tuple[tuple[int, int, str], int, float]],
    *,
    votes_digits: int = 12,
) -> tuple[int, dict[str, float], list[dict[str, Any]]]:
    source_list = sorted(sources, key=lambda item: item[0])
    votes: dict[int, float] = {}
    for _, class_code, weight in source_list:
        votes[class_code] = votes.get(class_code, 0.0) + weight
    max_vote = max(votes.values())
    tied_classes = [code for code, value in votes.items() if value == max_vote]
    support_order: dict[int, tuple[int, int, str]] = {}
    for order_key, class_code, _ in source_list:
        if class_code in tied_classes:
            support_order[class_code] = min(support_order.get(class_code, order_key), order_key)
    winner = min(tied_classes, key=lambda code: (support_order[code], code))
    vote_payload = {str(code): rounded_float(votes[code], votes_digits) for code in sorted(votes)}
    source_payload = [
        {
            "tile_id": order_key[2],
            "row": order_key[0],
            "col": order_key[1],
            "class": int(class_code),
            "weight": rounded_float(weight, votes_digits),
        }
        for order_key, class_code, weight in source_list
    ]
    return winner, vote_payload, source_payload


def compute_region(
    tile_a: TileResult,
    spec_a: TileSpec,
    tile_b: TileResult,
    spec_b: TileSpec,
    bounds: tuple[int, int, int, int],
    strategy: FusionStrategy,
) -> RegionResult:
    x0, y0, x1, y1 = bounds
    votes: list[dict[str, Any]] = []
    winners: Counter[int] = Counter()
    for y in range(y0, y1):
        for x in range(x0, x1):
            sources = []
            for tile, spec in ((tile_a, spec_a), (tile_b, spec_b)):
                local_x = x - spec.bounds[0]
                local_y = y - spec.bounds[1]
                class_code = int(tile.classes[local_y, local_x])
                weight = _pixel_weight(local_x, local_y, spec.width, spec.height, strategy)
                sources.append(((spec.row, spec.col, spec.tile_id), class_code, weight))
            winner, vote_dist, source_payload = _vote(sources)
            winners[winner] += 1
            votes.append(
                {
                    "x": x,
                    "y": y,
                    "sources": source_payload,
                    "votes": vote_dist,
                    "winner": winner,
                }
            )
    fingerprint = region_fingerprint(tile_a, tile_b, bounds, strategy)
    return RegionResult(
        pair_id=pair_id(spec_a, spec_b),
        tile_ids=tuple(sorted((tile_a.tile_id, tile_b.tile_id))),
        overlap_bounds=bounds,
        fingerprint=fingerprint,
        region_version=fingerprint,
        fusion_version=strategy.version,
        votes=votes,
        winner_histogram={str(code): int(winners[code]) for code in sorted(winners)},
    )


def _covering_tiles(
    x: int,
    y: int,
    specs: Iterable[TileSpec],
) -> list[TileSpec]:
    return [
        spec
        for spec in specs
        if spec.bounds[0] <= x < spec.bounds[2] and spec.bounds[1] <= y < spec.bounds[3]
    ]


def fuse_global(
    tiles: dict[str, TileResult],
    index: IndexData,
    config: SegmentationConfig,
    strategy: FusionStrategy,
) -> GlobalResult:
    valid_specs = [spec for spec in index.tiles if spec.tile_id in tiles]
    if not valid_specs:
        return GlobalResult(
            bounds=(0, 0, 0, 0),
            classes=np.full((0, 0), 255, dtype=np.uint16),
            class_histogram={str(code): 0 for code in config.class_codes},
            overlap_trace=[],
            postprocess_records=[],
            postprocess_components=[],
            fusion_version=strategy.version,
        )
    x0 = min(spec.bounds[0] for spec in valid_specs)
    y0 = min(spec.bounds[1] for spec in valid_specs)
    x1 = max(spec.bounds[2] for spec in valid_specs)
    y1 = max(spec.bounds[3] for spec in valid_specs)
    grid = np.full((y1 - y0, x1 - x0), 255, dtype=np.uint16)
    overlap_trace: list[dict[str, Any]] = []
    sorted_specs = sorted(valid_specs, key=lambda spec: spec.order_key)
    for y in range(y0, y1):
        for x in range(x0, x1):
            covering = _covering_tiles(x, y, sorted_specs)
            if not covering:
                continue
            if len(covering) == 1:
                spec = covering[0]
                tile = tiles[spec.tile_id]
                grid[y - y0, x - x0] = int(tile.classes[y - spec.bounds[1], x - spec.bounds[0]])
                continue
            sources = []
            for spec in covering:
                tile = tiles[spec.tile_id]
                local_x = x - spec.bounds[0]
                local_y = y - spec.bounds[1]
                class_code = int(tile.classes[local_y, local_x])
                weight = _pixel_weight(local_x, local_y, spec.width, spec.height, strategy)
                sources.append(((spec.row, spec.col, spec.tile_id), class_code, weight))
            winner, vote_dist, source_payload = _vote(sources)
            grid[y - y0, x - x0] = winner
            overlap_trace.append(
                {
                    "x": x,
                    "y": y,
                    "sources": source_payload,
                    "votes": vote_dist,
                    "winner": winner,
                }
            )

    classes, backfills, components = postprocess_connected_areas(grid, config)
    for record in overlap_trace:
        record["final_class"] = int(classes[record["y"] - y0, record["x"] - x0])
    histogram_array = np.bincount(classes[classes != 255].ravel(), minlength=len(config.class_codes))
    histogram = {str(code): int(histogram_array[code]) for code in config.class_codes}
    return GlobalResult(
        bounds=(x0, y0, x1, y1),
        classes=classes,
        class_histogram=histogram,
        overlap_trace=overlap_trace,
        postprocess_records=backfills,
        postprocess_components=components,
        fusion_version=strategy.version,
    )


def _components_for_class(grid: np.ndarray, class_code: int) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    visited = np.zeros((height, width), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for y in range(height):
        for x in range(width):
            if visited[y, x] or int(grid[y, x]) != class_code:
                continue
            queue: deque[tuple[int, int]] = deque([(y, x)])
            visited[y, x] = True
            pixels: list[tuple[int, int]] = []
            while queue:
                cy, cx = queue.popleft()
                pixels.append((cy, cx))
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx]:
                        if int(grid[ny, nx]) == class_code:
                            visited[ny, nx] = True
                            queue.append((ny, nx))
            pixels.sort()
            components.append(pixels)
    return components


def postprocess_connected_areas(
    source_grid: np.ndarray, config: SegmentationConfig
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    grid = source_grid.copy()
    small_mask = np.zeros(grid.shape, dtype=bool)
    components_payload: list[dict[str, Any]] = []
    component_info: list[tuple[int, list[tuple[int, int]], bool]] = []
    for code in config.class_codes:
        for pixels in _components_for_class(grid, code):
            removed = len(pixels) < config.min_connected_area
            component_info.append((code, pixels, removed))
            components_payload.append(
                {
                    "class": code,
                    "area": len(pixels),
                    "removed": removed,
                    "bounds": [
                        min(x for _, x in pixels),
                        min(y for y, _ in pixels),
                        max(x for _, x in pixels) + 1,
                        max(y for y, _ in pixels) + 1,
                    ],
                }
            )
            if removed:
                for y, x in pixels:
                    small_mask[y, x] = True

    stable_area: Counter[int] = Counter()
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            class_code = int(grid[y, x])
            if class_code != 255 and not small_mask[y, x]:
                stable_area[class_code] += 1
    if stable_area:
        global_main_class = min(
            stable_area,
            key=lambda code: (-stable_area[code], code),
        )
    else:
        global_main_class = config.class_codes[0]

    records: list[dict[str, Any]] = []
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if not small_mask[y, x]:
                continue
            original = int(grid[y, x])
            counts: Counter[int] = Counter()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < grid.shape[0] and 0 <= nx < grid.shape[1]:
                    neighbor = int(grid[ny, nx])
                    if neighbor != 255 and neighbor != original and not small_mask[ny, nx]:
                        counts[neighbor] += 1
            if counts:
                replacement = min(
                    counts,
                    key=lambda code: (-counts[code], -stable_area.get(code, 0), code),
                )
                reason = "adjacent_dominant_stable_class"
            else:
                replacement = global_main_class
                reason = "global_dominant_stable_class_fallback"
            grid[y, x] = replacement
            records.append(
                {
                    "x": x,
                    "y": y,
                    "from_class": original,
                    "to_class": int(replacement),
                    "reason": reason,
                    "neighbor_counts": {str(code): int(counts[code]) for code in sorted(counts)},
                }
            )
    components_payload.sort(key=lambda item: (item["bounds"][1], item["bounds"][0], item["class"]))
    return grid, records, components_payload
