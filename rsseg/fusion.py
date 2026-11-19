"""Cross-tile overlap fusion and connected-area class post-processing.

Fusion is deterministic:

* ``distance_weighted`` votes with weights derived from the pixel distance to
  each participating tile boundary; the lexicographically smaller tile key
  wins exact ties.
* ``first_wins`` keeps the smallest tile key directly (still deterministic).

Post-processing removes connected components below the configured minimum
area and fills them with the dominant neighboring class.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CLASS_CODES, SegmentationConfig
from .tiles import TileIndex, TileRecord


@dataclass(frozen=True)
class FusionRegion:
    """One overlap between two tiles; ``a`` is always the lexicographic key."""

    a: tuple[int, int]
    b: tuple[int, int]
    global_x0: int
    global_y0: int
    global_x1: int
    global_y1: int
    local_a_box: tuple[int, int, int, int]
    local_b_box: tuple[int, int, int, int]

    @property
    def key(self) -> tuple[tuple[int, int], tuple[int, int]]:
        return (self.a, self.b)

    @property
    def name(self) -> str:
        return f"fusion_{self.a[0]}_{self.a[1]}__{self.b[0]}_{self.b[1]}"


def build_fusion_regions(index: TileIndex) -> list[FusionRegion]:
    """Intersect spatial bounds of adjacent tiles; non-overlapping pairs are skipped."""

    tile_map = index.tile_map()
    regions: list[FusionRegion] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for tile in index.tiles:
        for neighbor_key in tile.neighbors:
            pair = frozenset((tile.key, neighbor_key))
            if pair in seen:
                continue
            seen.add(pair)
            neighbor = tile_map[neighbor_key]
            region = _intersect(tile, neighbor)
            if region is not None:
                regions.append(region)
    return sorted(regions, key=lambda r: (r.a, r.b))


def _intersect(tile_a: TileRecord, tile_b: TileRecord) -> FusionRegion | None:
    ax0, ay0, ax1, ay1 = tile_a.bounds
    bx0, by0, bx1, by1 = tile_b.bounds
    gx0, gy0 = max(ax0, bx0), max(ay0, by0)
    gx1, gy1 = min(ax1, bx1), min(ay1, by1)
    if gx0 >= gx1 or gy0 >= gy1:
        return None
    first, second = sorted((tile_a, tile_b), key=lambda t: t.key)
    return FusionRegion(
        a=first.key,
        b=second.key,
        global_x0=gx0,
        global_y0=gy0,
        global_x1=gx1,
        global_y1=gy1,
        local_a_box=(gx0 - first.bounds[0], gy0 - first.bounds[1],
                     gx1 - first.bounds[0], gy1 - first.bounds[1]),
        local_b_box=(gx0 - second.bounds[0], gy0 - second.bounds[1],
                     gx1 - second.bounds[0], gy1 - second.bounds[1]),
    )


def _distance_weights(region: FusionRegion,
                      index: TileIndex) -> tuple[np.ndarray, np.ndarray]:
    tile_map = index.tile_map()
    tile_a = tile_map[region.a]
    tile_b = tile_map[region.b]
    h = region.global_y1 - region.global_y0
    w = region.global_x1 - region.global_x0
    ax0, ay0, ax1, ay1 = tile_a.bounds
    bx0, by0, bx1, by1 = tile_b.bounds

    def plane(bounds: tuple[int, int, int, int]) -> np.ndarray:
        x0, y0, x1, y1 = bounds
        xs = np.arange(region.global_x0, region.global_x1)
        ys = np.arange(region.global_y0, region.global_y1)
        xx, yy = np.meshgrid(xs, ys)
        dist_x = np.minimum(np.maximum(xx - x0, 0), np.maximum(x1 - 1 - xx, 0))
        dist_y = np.minimum(np.maximum(yy - y0, 0), np.maximum(y1 - 1 - yy, 0))
        dist = dist_x + dist_y
        return (dist + 1).astype(np.int64)

    wa = plane(tile_a.bounds)
    wb = plane(tile_b.bounds)
    return wa, wb


def assemble_global(
    index: TileIndex,
    class_maps: dict[tuple[int, int], np.ndarray],
    cfg: SegmentationConfig,
) -> np.ndarray:
    """Paste tile maps then resolve every declared overlap deterministically."""

    x0, y0, x1, y1 = index.bounds
    canvas = np.zeros((y1 - y0, x1 - x0), dtype=np.uint8)
    regions = build_fusion_regions(index)
    covered = np.zeros_like(canvas, dtype=bool)

    for tile in index.tiles:
        local = class_maps[tile.key]
        tx0, ty0, tx1, ty1 = tile.bounds
        canvas[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] = local
        covered[ty0 - y0:ty1 - y0, tx0 - x0:tx1 - x0] = True

    for region in regions:
        map_a = class_maps[region.a]
        map_b = class_maps[region.b]
        ax0, ay0, ax1, ay1 = region.local_a_box
        bx0, by0, bx1, by1 = region.local_b_box
        patch_a = map_a[ay0:ay1, ax0:ax1]
        patch_b = map_b[by0:by1, bx0:bx1]
        if cfg.fusion_strategy == "first_wins":
            fused = np.where(region.a < region.b, patch_a, patch_b)
        else:
            wa, wb = _distance_weights(region, index)
            fused = _weighted_vote(patch_a, patch_b, wa, wb, region.a, region.b)
        gy0 = region.global_y0 - y0
        gx0 = region.global_x0 - x0
        canvas[gy0:gy0 + fused.shape[0], gx0:gx0 + fused.shape[1]] = fused

    if not np.all(covered):  # pragma: no cover - index validation prevents gaps
        raise RuntimeError("global canvas contains uncovered pixels")
    return canvas


def _weighted_vote(patch_a: np.ndarray, patch_b: np.ndarray,
                   weight_a: np.ndarray, weight_b: np.ndarray,
                   key_a: tuple[int, int], key_b: tuple[int, int]) -> np.ndarray:
    result = np.zeros_like(patch_a)
    for code in CLASS_CODES:
        result[(patch_a == code) & (patch_b == code)] = code
    conflict = patch_a != patch_b
    a_wins = weight_a > weight_b
    tie = conflict & (weight_a == weight_b)
    if key_a < key_b:
        a_wins = a_wins | tie
    else:  # pragma: no cover - region construction guarantees a < b
        a_wins = a_wins & ~tie
    result[conflict & a_wins] = patch_a[conflict & a_wins]
    result[conflict & ~a_wins] = patch_b[conflict & ~a_wins]
    return result


def connected_components(label_plane: np.ndarray) -> tuple[np.ndarray, int]:
    """Vectorized BFS 4-connectivity labeling; raster scan gives deterministic ids."""

    h, w = label_plane.shape
    labels = np.zeros((h, w), dtype=np.int32)
    component_id = 0
    visited = np.zeros((h, w), dtype=bool)
    for start_y in range(h):
        for start_x in range(w):
            if visited[start_y, start_x]:
                continue
            component_id += 1
            frontier = [(start_y, start_x)]
            target = label_plane[start_y, start_x]
            visited[start_y, start_x] = True
            labels[start_y, start_x] = component_id
            while frontier:
                ys = np.array([p[0] for p in frontier], dtype=np.intp)
                xs = np.array([p[1] for p in frontier], dtype=np.intp)
                next_ys: list[int] = []
                next_xs: list[int] = []
                up_y, up_x = ys - 1, xs
                down_y, down_x = ys + 1, xs
                left_y, left_x = ys, xs - 1
                right_y, right_x = ys, xs + 1
                for ny, nx in ((up_y, up_x), (down_y, down_x),
                               (left_y, left_x), (right_y, right_x)):
                    valid = (ny >= 0) & (ny < h) & (nx >= 0) & (nx < w)
                    if not np.any(valid):
                        continue
                    ny, nx = ny[valid], nx[valid]
                    accept = valid.copy()
                    ok = (~visited[ny, nx]) & (label_plane[ny, nx] == target)
                    chosen_y, chosen_x = ny[ok], nx[ok]
                    visited[chosen_y, chosen_x] = True
                    labels[chosen_y, chosen_x] = component_id
                    next_ys.extend(chosen_y.tolist())
                    next_xs.extend(chosen_x.tolist())
                frontier = list(zip(next_ys, next_xs))
    return labels, component_id


def postprocess(global_grid: np.ndarray, min_area: int) -> tuple[np.ndarray, list[dict]]:
    """Replace sub-threshold components with the dominant 4-neighbor class."""

    grid = global_grid.copy()
    component_labels, component_count = connected_components(grid)
    changes: list[dict] = []
    h, w = grid.shape
    for component_id in range(1, component_count + 1):
        mask = component_labels == component_id
        area = int(mask.sum())
        if area >= min_area:
            continue
        pixels = np.argwhere(mask)
        original = int(grid[mask][0])
        neighbor_votes: dict[int, int] = {}
        for y, x in pixels:
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < h and 0 <= nx < w and not mask[ny, nx]:
                    code = int(grid[ny, nx])
                    neighbor_votes[code] = neighbor_votes.get(code, 0) + 1
        if not neighbor_votes:
            continue
        fill = sorted(neighbor_votes.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        grid[mask] = fill
        changes.append({
            "component_id": int(component_id),
            "original_class": original,
            "fill_class": int(fill),
            "area": area,
            "pixels": [[int(y), int(x)] for y, x in pixels.tolist()],
        })
    changes.sort(key=lambda item: (item["original_class"], item["component_id"]))
    return grid, changes
