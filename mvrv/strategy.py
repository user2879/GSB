from __future__ import annotations

import math
from typing import Any

from .hashing import sha256_object


STRATEGY_TABLE: list[dict[str, Any]] = [
    {
        "id": "coarse",
        "max_views": 2,
        "max_pairs": 1,
        "max_estimated_children": 24,
        "block_size": 8,
        "feature_threshold": 45,
        "descriptor_max_distance": 260,
        "search_radius": 48,
        "child_order": [0, 4, 2, 6, 1, 5, 3, 7],
        "order_axis": "zyx",
    },
    {
        "id": "balanced",
        "max_views": 4,
        "max_pairs": 6,
        "max_estimated_children": 256,
        "block_size": 4,
        "feature_threshold": 80,
        "descriptor_max_distance": 220,
        "search_radius": 40,
        "child_order": [0, 1, 2, 3, 4, 5, 6, 7],
        "order_axis": "xyz",
    },
    {
        "id": "fine",
        "max_views": 10_000,
        "max_pairs": 10_000,
        "max_estimated_children": 10_000_000,
        "block_size": 2,
        "feature_threshold": 120,
        "descriptor_max_distance": 190,
        "search_radius": 36,
        "child_order": [0, 2, 1, 3, 4, 6, 5, 7],
        "order_axis": "yxz",
    },
]


def estimate_subvoxels(config: dict[str, Any], view_count: int, pair_count: int) -> int:
    bounds = config["world_bounds"]
    size = int(config["voxel_size"])
    root_counts = [
        math.ceil((bounds["max"][i] - bounds["min"][i]) / size)
        for i in range(3)
    ]
    root_total = math.prod(root_counts)
    estimated_occupied = min(root_total, max(1, pair_count * 4, view_count * 6))
    return estimated_occupied * 8


def select_strategy(view_count: int, pair_count: int, estimated_children: int) -> dict[str, Any]:
    for candidate in STRATEGY_TABLE:
        if (
            view_count <= candidate["max_views"]
            and pair_count <= candidate["max_pairs"]
            and estimated_children <= candidate["max_estimated_children"]
        ):
            strategy = dict(candidate)
            strategy["estimated_children"] = estimated_children
            strategy["fingerprint"] = sha256_object(
                {key: value for key, value in strategy.items() if key != "fingerprint"}
            )
            return strategy
    raise RuntimeError("no reconstruction strategy selected")
