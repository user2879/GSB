"""View-scale adaptive strategy table."""

from __future__ import annotations

from typing import Any


STRATEGY_TABLE: list[dict[str, Any]] = [
    {
        "name": "compact",
        "view_limit": 2,
        "pair_limit": 1,
        "block_size": 8,
        "max_match_distance": 36,
        "base_depth": 1,
        "split_order": [2, 1, 0],
        "descriptor_shift": 1,
    },
    {
        "name": "balanced",
        "view_limit": 3,
        "pair_limit": 3,
        "block_size": 10,
        "max_match_distance": 52,
        "base_depth": 2,
        "split_order": [1, 2, 0],
        "descriptor_shift": 2,
    },
    {
        "name": "wide",
        "view_limit": 64,
        "pair_limit": 4096,
        "block_size": 12,
        "max_match_distance": 72,
        "base_depth": 3,
        "split_order": [0, 1, 2],
        "descriptor_shift": 3,
    },
]


def select_strategy(
    view_count: int,
    pair_count: int,
    quantization_level: int,
    root_voxel_count: int,
) -> dict[str, Any]:
    """Select a strategy using view/pair scale and estimated child voxels.

    The estimate is intentionally an integer, monotone proxy rather than a
    display-only label.  It feeds the same table that determines feature block
    size, matching radius, split depth and split order.
    """

    estimated_children = root_voxel_count * max(1, quantization_level) * (1 + min(3, view_count))
    chosen = STRATEGY_TABLE[-1]
    for candidate in STRATEGY_TABLE:
        if view_count <= candidate["view_limit"] and pair_count <= candidate["pair_limit"]:
            chosen = candidate
            break
    result = dict(chosen)
    result["view_count"] = view_count
    result["pair_count"] = pair_count
    result["quantization_level"] = quantization_level
    result["root_voxel_count"] = root_voxel_count
    result["estimated_children"] = estimated_children
    result["effective_max_depth"] = max(
        int(result["base_depth"]),
        int(result["base_depth"]) + (1 if estimated_children > 160 else 0),
    )
    return result


def strategy_table() -> list[dict[str, Any]]:
    return [dict(item) for item in STRATEGY_TABLE]
