"""View-scale adaptive strategy selection with auditable work estimates."""

from __future__ import annotations

from typing import Any

from .constants import STRATEGY_TABLE
from .models import ReconstructionConfig


def estimated_subvoxels(config: ReconstructionConfig) -> int:
    x0, y0, z0, x1, y1, z1 = config.world_bounds
    roots = 1
    for low, high in ((x0, x1), (y0, y1), (z0, z1)):
        roots *= max(1, (high - low + config.voxel_size - 1) // config.voxel_size)
    grade_factor = 16 // config.quant_level
    per_root = 1 + max(0, config.max_subdiv_depth) * 8
    return roots * per_root * grade_factor // 8


def select_strategy(
    config: ReconstructionConfig,
    view_count: int,
    pair_count: int,
) -> dict[str, Any]:
    estimate = estimated_subvoxels(config)
    if config.strategy_override is not None:
        selected = next(
            entry for entry in STRATEGY_TABLE if entry["name"] == config.strategy_override
        )
        basis = "override"
    else:
        selected = STRATEGY_TABLE[-1]
        basis = "capacity_ceiling"
        for entry in STRATEGY_TABLE:
            if (
                view_count <= entry["max_views"]
                and pair_count <= entry["max_pairs"]
                and estimate <= entry["max_estimated_subvoxels"]
            ):
                selected = entry
                basis = "first_fitting_policy_row"
                break
    image_width = 64
    image_height = 48
    block = int(selected["feature_block"])
    blocks = (image_width // block) * (image_height // block)
    return {
        "name": selected["name"],
        "basis": basis,
        "view_count": view_count,
        "pair_count": pair_count,
        "estimated_subvoxels": estimate,
        "estimated_feature_candidates_per_view": blocks,
        "feature_block": block,
        "descriptor_chroma_floor": int(selected["descriptor_chroma_floor"]),
        "match_radius": int(selected["match_radius"]),
        "max_matches": int(selected["max_matches"]),
        "subdivision_order": selected["subdivision_order"],
        "estimated_scan_blocks": blocks * view_count,
        "estimated_multiply_adds": blocks * view_count * block * block * 6,
    }


def resolution_specific_strategy(
    config: ReconstructionConfig,
    view_count: int,
    pair_count: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    record = select_strategy(config, view_count, pair_count)
    block = record["feature_block"]
    blocks = (width // block) * (height // block)
    record["image_width"] = width
    record["image_height"] = height
    record["estimated_feature_candidates_per_view"] = blocks
    record["estimated_scan_blocks"] = blocks * view_count
    record["estimated_multiply_adds"] = blocks * view_count * block * block * 6
    return record

