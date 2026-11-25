"""Window-to-pixel projection, weighted border voting, and raster cleanup."""

from __future__ import annotations

from collections import Counter, deque
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .tiles import TileData
from .util import sha256_json
from .windows import WindowSpec


def build_tile_pixel_maps(
    tiles: dict[str, TileData],
    windows: list[WindowSpec],
    scores: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    tile_size = int(config["tile"]["size"])
    score_by_window = {record["window_id"]: record for record in scores}
    decision_by_window = {record["window_id"]: record for record in decisions}
    maps: dict[str, dict[str, Any]] = {}
    for tile_id, tile in tiles.items():
        maps[tile_id] = {
            "score_micro": np.full((tile_size, tile_size), -1, dtype=np.int64),
            "window_id": np.full((tile_size, tile_size), "", dtype=object),
        }
    for window in windows:
        tile = tiles[window.owner_tile_id]
        top, left, bottom, right = window.core_rect
        tile_top = tile.spec.row * tile_size
        tile_left = tile.spec.col * tile_size
        local_top = max(0, top - tile_top)
        local_left = max(0, left - tile_left)
        local_bottom = min(tile_size, bottom - tile_top)
        local_right = min(tile_size, right - tile_left)
        if local_top >= local_bottom or local_left >= local_right:
            continue
        score = score_by_window[window.window_id]
        score_map = maps[window.owner_tile_id]["score_micro"]
        window_map = maps[window.owner_tile_id]["window_id"]
        score_map[local_top:local_bottom, local_left:local_right] = int(
            score["score_micro"]
        )
        window_map[local_top:local_bottom, local_left:local_right] = window.window_id

    for tile_id, map_data in maps.items():
        decision_map = np.zeros(map_data["score_micro"].shape, dtype=np.int8)
        decision_map[map_data["score_micro"] < 0] = -1
        valid = map_data["score_micro"] >= 0
        threshold_micro = int(config["detector"]["threshold_permille"]) * 1000
        decision_map[valid] = (map_data["score_micro"][valid] >= threshold_micro).astype(
            np.int8
        )
        map_data["decision"] = decision_map
        map_data["tile_id"] = tile_id
    return maps


def _tile_order(tile: TileData) -> tuple[int, int, str]:
    return (tile.spec.row, tile.spec.col, tile.spec.tile_id)


def adjacent_tile_pairs(tiles: dict[str, TileData]) -> list[tuple[str, str, str]]:
    by_position = {
        (tile.spec.row, tile.spec.col): tile_id for tile_id, tile in tiles.items()
    }
    pairs = []
    for tile_id, tile in sorted(tiles.items(), key=lambda item: _tile_order(item[1])):
        right = by_position.get((tile.spec.row, tile.spec.col + 1))
        if right and right in tile.spec.neighbors:
            pairs.append((tile_id, right, "vertical"))
        below = by_position.get((tile.spec.row + 1, tile.spec.col))
        if below and below in tile.spec.neighbors:
            pairs.append((tile_id, below, "horizontal"))
    return pairs


def _band_geometry(
    tile_a: TileData,
    tile_b: TileData,
    orientation: str,
    tile_size: int,
    band_width: int,
) -> dict[str, Any]:
    if orientation == "vertical":
        boundary = tile_b.spec.col * tile_size
        top = tile_a.spec.row * tile_size
        bottom = top + tile_size
        left = max(tile_a.spec.col * tile_size, boundary - band_width)
        right = min((tile_b.spec.col + 1) * tile_size, boundary + band_width)
        return {
            "orientation": orientation,
            "boundary": boundary,
            "top": top,
            "bottom": bottom,
            "left": left,
            "right": right,
            "a_side": "left",
            "b_side": "right",
        }
    boundary = tile_b.spec.row * tile_size
    left = tile_a.spec.col * tile_size
    right = left + tile_size
    top = max(tile_a.spec.row * tile_size, boundary - band_width)
    bottom = min((tile_b.spec.row + 1) * tile_size, boundary + band_width)
    return {
        "orientation": orientation,
        "boundary": boundary,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "a_side": "top",
        "b_side": "bottom",
    }


def _windows_in_band(
    windows_by_tile: dict[str, list[WindowSpec]],
    tile_id: str,
    tile: TileData,
    geometry: dict[str, Any],
    tile_size: int,
) -> list[str]:
    found = []
    tile_top = tile.spec.row * tile_size
    tile_left = tile.spec.col * tile_size
    geometry_top = int(geometry["top"]) - tile_top
    geometry_bottom = int(geometry["bottom"]) - tile_top
    geometry_left = int(geometry["left"]) - tile_left
    geometry_right = int(geometry["right"]) - tile_left
    for window in windows_by_tile.get(tile_id, []):
        top, left, bottom, right = window.core_rect
        local_top = top - tile_top
        local_left = left - tile_left
        local_bottom = bottom - tile_top
        local_right = right - tile_left
        if (
            local_top < geometry_bottom
            and local_bottom > geometry_top
            and local_left < geometry_right
            and local_right > geometry_left
        ):
            found.append(window.window_id)
    return sorted(found)


def _pixel_side_values(
    map_data: dict[str, Any],
    row: int,
    col: int,
    tile: TileData,
    tile_size: int,
) -> tuple[int, int] | None:
    local_row = row - tile.spec.row * tile_size
    local_col = col - tile.spec.col * tile_size
    if not (0 <= local_row < tile_size and 0 <= local_col < tile_size):
        return None
    score = int(map_data["score_micro"][local_row, local_col])
    decision = int(map_data["decision"][local_row, local_col])
    if score < 0 or decision < 0:
        return None
    return score, decision


def plan_fusion_bands(
    tiles: dict[str, TileData],
    windows: list[WindowSpec],
    scores: list[dict[str, Any]],
    summaries: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tile_size = int(config["tile"]["size"])
    band_width = int(config["fusion"]["overlap_width"])
    windows_by_tile: dict[str, list[WindowSpec]] = {}
    for window in windows:
        windows_by_tile.setdefault(window.owner_tile_id, []).append(window)
    bands = []
    for tile_a_id, tile_b_id, orientation in adjacent_tile_pairs(tiles):
        tile_a = tiles[tile_a_id]
        tile_b = tiles[tile_b_id]
        if tile_a_id not in summaries or tile_b_id not in summaries:
            continue
        geometry = _band_geometry(tile_a, tile_b, orientation, tile_size, band_width)
        side_a_windows = _windows_in_band(
            windows_by_tile, tile_a_id, tile_a, geometry, tile_size
        )
        side_b_windows = _windows_in_band(
            windows_by_tile, tile_b_id, tile_b, geometry, tile_size
        )
        if not side_a_windows or not side_b_windows:
            continue
        score_by_window = {record["window_id"]: record for record in scores}
        score_keys = [
            score_by_window[window_id]["score_key"]
            for window_id in side_a_windows + side_b_windows
        ]
        fusion_key = sha256_json(
            {
                "fusion_strategy_version": config["fusion"]["strategy_version"],
                "tile_a_summary_key": summaries[tile_a_id]["summary_key"],
                "tile_b_summary_key": summaries[tile_b_id]["summary_key"],
                "side_a_score_keys": [
                    score_by_window[window_id]["score_key"] for window_id in side_a_windows
                ],
                "side_b_score_keys": [
                    score_by_window[window_id]["score_key"] for window_id in side_b_windows
                ],
                "geometry": geometry,
                "band_width": band_width,
            }
        )
        bands.append(
            {
                "band_id": f"{tile_a_id}__{tile_b_id}__{orientation}",
                "tile_a_id": tile_a_id,
                "tile_b_id": tile_b_id,
                "tile_a_cache_key": summaries[tile_a_id]["summary_key"],
                "tile_b_cache_key": summaries[tile_b_id]["summary_key"],
                "tile_a_content_key": tiles[tile_a_id].content_key,
                "tile_b_content_key": tiles[tile_b_id].content_key,
                "side_a_score_keys": score_keys[: len(side_a_windows)],
                "side_b_score_keys": score_keys[len(side_a_windows) :],
                "geometry": geometry,
                "fusion_key": fusion_key,
                "fusion_strategy_version": config["fusion"]["strategy_version"],
            }
        )
    bands.sort(
        key=lambda band: (
            tiles[band["tile_a_id"]].spec.row,
            tiles[band["tile_a_id"]].spec.col,
            band["geometry"]["orientation"],
        )
    )
    return bands


def compute_fusion_band(
    plan: dict[str, Any],
    tiles: dict[str, TileData],
    pixel_maps: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    tile_size = int(config["tile"]["size"])
    tile_a = tiles[plan["tile_a_id"]]
    tile_b = tiles[plan["tile_b_id"]]
    geometry = plan["geometry"]
    width = int(config["fusion"]["overlap_width"])
    update_rows = []
    update_cols = []
    update_values = []
    update_sources = []
    tie_pixels = []
    orientation = geometry["orientation"]
    for row in range(int(geometry["top"]), int(geometry["bottom"])):
        for col in range(int(geometry["left"]), int(geometry["right"])):
            if orientation == "vertical":
                boundary = int(geometry["boundary"])
                if col < boundary:
                    a_row, a_col = row, col
                    b_row, b_col = row, 2 * boundary - 1 - col
                else:
                    a_row, a_col = row, 2 * boundary - 1 - col
                    b_row, b_col = row, col
            else:
                boundary = int(geometry["boundary"])
                if row < boundary:
                    a_row, a_col = row, col
                    b_row, b_col = 2 * boundary - 1 - row, col
                else:
                    a_row, a_col = 2 * boundary - 1 - row, col
                    b_row, b_col = row, col
            values_a = _pixel_side_values(
                pixel_maps[plan["tile_a_id"]], a_row, a_col, tile_a, tile_size
            )
            values_b = _pixel_side_values(
                pixel_maps[plan["tile_b_id"]], b_row, b_col, tile_b, tile_size
            )
            if values_a is None or values_b is None:
                continue
            score_a, decision_a = values_a
            score_b, decision_b = values_b
            if orientation == "vertical":
                boundary = int(geometry["boundary"])
                distance_a = boundary - a_col
                distance_b = b_col - boundary + 1
            else:
                boundary = int(geometry["boundary"])
                distance_a = boundary - a_row
                distance_b = b_row - boundary + 1
            proximity_a = max(1, width - distance_a + 1)
            proximity_b = max(1, width - distance_b + 1)
            score_a_level = score_a // 1000
            score_b_level = score_b // 1000
            anomaly_support = (
                proximity_a * score_a_level + proximity_b * score_b_level
            )
            normal_support = proximity_a * (1000 - score_a_level) + proximity_b * (
                1000 - score_b_level
            )
            if anomaly_support > normal_support:
                final_value = 1
                source = "weighted_vote_anomaly"
            elif anomaly_support < normal_support:
                final_value = 0
                source = "weighted_vote_normal"
            else:
                order_a = (tile_a.spec.row, tile_a.spec.col, tile_a.spec.tile_id)
                order_b = (tile_b.spec.row, tile_b.spec.col, tile_b.spec.tile_id)
                lexical = min(order_a, order_b)
                lexical_tile = order_a if lexical == order_a else order_b
                lexical_decision = decision_a if lexical_tile == order_a else decision_b
                final_value = int(lexical_decision)
                source = "tie_lexicographic_tile_decision"
                tie_pixels.append(
                    [row, col, "equal_weighted_support", list(lexical_tile)]
                )
            update_rows.append(row)
            update_cols.append(col)
            update_values.append(final_value)
            update_sources.append(
                {
                    "row": row,
                    "col": col,
                    "value": final_value,
                    "source": source,
                    "side_a": {
                        "score_micro": score_a,
                        "decision": bool(decision_a),
                        "anomaly_weight": proximity_a * score_a_level,
                        "normal_weight": proximity_a * (1000 - score_a_level),
                        "proximity": proximity_a,
                    },
                    "side_b": {
                        "score_micro": score_b,
                        "decision": bool(decision_b),
                        "anomaly_weight": proximity_b * score_b_level,
                        "normal_weight": proximity_b * (1000 - score_b_level),
                        "proximity": proximity_b,
                    },
                }
            )
    return {
        "band_id": plan["band_id"],
        "fusion_key": plan["fusion_key"],
        "fusion_strategy_version": plan["fusion_strategy_version"],
        "tile_a_id": plan["tile_a_id"],
        "tile_b_id": plan["tile_b_id"],
        "tile_a_cache_key": plan["tile_a_cache_key"],
        "tile_b_cache_key": plan["tile_b_cache_key"],
        "tile_a_content_key": plan["tile_a_content_key"],
        "tile_b_content_key": plan["tile_b_content_key"],
        "side_a_score_keys": plan["side_a_score_keys"],
        "side_b_score_keys": plan["side_b_score_keys"],
        "overlap_geometry": geometry,
        "overlap_pixel_count": len(update_rows),
        "tie_pixel_count": len(tie_pixels),
        "tie_pixels": tie_pixels,
        "tie_reason": (
            "equal weighted anomaly and normal support; selected lexicographically smaller tile row,column,id"
            if tie_pixels
            else None
        ),
        "update_rows": update_rows,
        "update_cols": update_cols,
        "update_values": update_values,
        "update_sources": update_sources,
        "weight_rule": "proximity_to_boundary_times_window_score_split_between_anomaly_and_normal_support",
    }


def apply_fused_bands(
    pixel_maps: dict[str, dict[str, Any]],
    fusion_results: list[dict[str, Any]],
    tiles: dict[str, TileData],
    config: dict[str, Any],
    grid_shape: tuple[int, int] | None = None,
) -> dict[str, Any]:
    tile_size = int(config["tile"]["size"])
    if grid_shape is None:
        max_row = max(tile.spec.row for tile in tiles.values())
        max_col = max(tile.spec.col for tile in tiles.values())
        height = (max_row + 1) * tile_size
        width = (max_col + 1) * tile_size
    else:
        height, width = grid_shape
    fused = np.full((height, width), -1, dtype=np.int8)
    sources = np.full((height, width), "", dtype=object)
    for tile_id, tile in tiles.items():
        map_data = pixel_maps[tile_id]
        top = tile.spec.row * tile_size
        left = tile.spec.col * tile_size
        fused[top : top + tile_size, left : left + tile_size] = map_data["decision"]
        sources[top : top + tile_size, left : left + tile_size] = "window_core"
    for result in sorted(fusion_results, key=lambda item: item["band_id"]):
        for row, col, value in zip(
            result["update_rows"], result["update_cols"], result["update_values"]
        ):
            fused[int(row), int(col)] = int(value)
            sources[int(row), int(col)] = result["band_id"]
    pre_post = fused.copy()
    component_info = remove_small_components(
        fused, int(config["fusion"]["min_component_pixels"])
    )
    pixel_key = sha256_json(
        {
            "pre_post_rows": [
                "".join("1" if value == 1 else "0" if value == 0 else "u" for value in row)
                for row in pre_post.tolist()
            ],
            "post_rows": [
                "".join("1" if value == 1 else "0" if value == 0 else "u" for value in row)
                for row in fused.tolist()
            ],
            "fusion_keys": [result["fusion_key"] for result in fusion_results],
            "postprocess_version": "small-component-backfill-v1",
            "min_component_pixels": int(config["fusion"]["min_component_pixels"]),
        }
    )
    return {
        "pixel_key": pixel_key,
        "shape": [height, width],
        "anomaly": fused,
        "pre_post": pre_post,
        "sources": sources,
        "postprocess": component_info,
    }


def remove_small_components(anomaly: np.ndarray, minimum_size: int) -> dict[str, Any]:
    height, width = anomaly.shape
    visited = np.zeros((height, width), dtype=bool)
    small_components = []
    backfills = []
    for row in range(height):
        for col in range(width):
            if visited[row, col] or anomaly[row, col] != 1:
                continue
            queue = deque([(row, col)])
            visited[row, col] = True
            component = []
            while queue:
                current_row, current_col = queue.popleft()
                component.append((current_row, current_col))
                for delta_row, delta_col in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_row = current_row + delta_row
                    next_col = current_col + delta_col
                    if (
                        0 <= next_row < height
                        and 0 <= next_col < width
                        and not visited[next_row, next_col]
                        and anomaly[next_row, next_col] == 1
                    ):
                        visited[next_row, next_col] = True
                        queue.append((next_row, next_col))
            if len(component) < minimum_size:
                small_components.append(component)
    component_pixels = {
        pixel for component in small_components for pixel in component
    }
    for component in small_components:
        for row, col in component:
            neighbor_labels = []
            for delta_row in (-1, 0, 1):
                for delta_col in (-1, 0, 1):
                    if delta_row == 0 and delta_col == 0:
                        continue
                    next_row = row + delta_row
                    next_col = col + delta_col
                    if not (0 <= next_row < height and 0 <= next_col < width):
                        continue
                    if (next_row, next_col) in component_pixels:
                        continue
                    label = int(anomaly[next_row, next_col])
                    if label >= 0:
                        neighbor_labels.append(label)
            counts = Counter(neighbor_labels)
            if counts:
                majority_count = max(counts.values())
                majority = min(
                    label for label, count in counts.items() if count == majority_count
                )
            else:
                majority = 0
            anomaly[row, col] = majority
            backfills.append(
                {
                    "row": row,
                    "col": col,
                    "component_size": len(component),
                    "backfill_class": majority,
                    "neighbor_counts": dict(sorted(counts.items())),
                }
            )
    return {
        "minimum_component_pixels": minimum_size,
        "removed_component_count": len(small_components),
        "removed_pixel_count": len(component_pixels),
        "components": [
            {"size": len(component), "pixels": [list(pixel) for pixel in component]}
            for component in small_components
        ],
        "backfills": backfills,
        "rule": "remove_four_connected_anomaly_components_below_minimum_then_backfill_eight_neighbor_majority",
    }


def write_pixel_raster(path: Path, anomaly: np.ndarray) -> None:
    image = np.full(anomaly.shape, 0, dtype=np.uint8)
    image[anomaly == 1] = 255
    image[anomaly < 0] = 128
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image, mode="L").save(path, optimize=False)


def pixel_raster_bytes(anomaly: np.ndarray) -> bytes:
    image = np.full(anomaly.shape, 0, dtype=np.uint8)
    image[anomaly == 1] = 255
    image[anomaly < 0] = 128
    buffer = BytesIO()
    Image.fromarray(image, mode="L").save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()
