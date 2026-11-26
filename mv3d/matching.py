"""Stable, declared-pair-only feature matching."""

from __future__ import annotations

from typing import Any

from .constants import ENGINE_VERSION
from .fixed import stable_hash
from .models import ReconstructionConfig


def pair_identifier(first: int, second: int) -> str:
    low, high = sorted((first, second))
    return f"p:{low:04d}:{high:04d}"


def match_key(
    first: dict[str, Any],
    second: dict[str, Any],
    config: ReconstructionConfig,
    strategy: dict[str, Any],
) -> str:
    return stable_hash(
        {
            "engine_version": ENGINE_VERSION,
            "stage": "matches",
            "feature_keys": sorted((first["key"], second["key"])),
            "bits": config.bits,
            "quant_level": config.quant_level,
            "min_disparity": config.min_disparity,
            "match_radius": strategy["match_radius"],
            "max_matches": strategy["max_matches"],
        }
    )


def descriptor_distance(first: list[int], second: list[int]) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1]) + abs(first[2] - second[2])


def _disparity(first: dict[str, Any], second: dict[str, Any]) -> int:
    return max(abs(first["x"] - second["x"]), abs(first["y"] - second["y"]))


def match_pair(
    first_view: dict[str, Any],
    second_view: dict[str, Any],
    config: ReconstructionConfig,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    left_points = sorted(first_view["points"], key=lambda item: (item["x"], item["y"], item["id"]))
    right_points = sorted(second_view["points"], key=lambda item: (item["x"], item["y"], item["id"]))
    radius = int(strategy["match_radius"])
    maximum = int(strategy["max_matches"])
    used_right: set[str] = set()
    inliers: list[dict[str, Any]] = []
    for left in left_points:
        options = []
        for right in right_points:
            if right["id"] in used_right:
                continue
            color_distance = descriptor_distance(left["descriptor"], right["descriptor"])
            if color_distance > radius:
                continue
            if _disparity(left, right) < config.min_disparity:
                continue
            options.append(
                (
                    color_distance,
                    right["x"],
                    right["y"],
                    right["id"],
                    right,
                )
            )
        if not options:
            continue
        color_distance, _, _, _, right = min(options)
        used_right.add(right["id"])
        inliers.append(
            {
                "left_id": left["id"],
                "right_id": right["id"],
                "left": [left["x"], left["y"]],
                "right": [right["x"], right["y"]],
                "descriptor_distance": color_distance,
                "disparity": _disparity(left, right),
            }
        )
        if len(inliers) >= maximum:
            break
    inliers.sort(
        key=lambda item: (
            item["left"][0],
            item["left"][1],
            item["right"][0],
            item["right"][1],
            item["left_id"],
            item["right_id"],
        )
    )
    key = match_key(first_view, second_view, config, strategy)
    return {
        "pair_id": pair_identifier(first_view["view_id"], second_view["view_id"]),
        "view_ids": [first_view["view_id"], second_view["view_id"]],
        "key": key,
        "feature_keys": [first_view["key"], second_view["key"]],
        "inliers": inliers,
        "inlier_count": len(inliers),
        "left_candidates": len(left_points),
        "right_candidates": len(right_points),
        "multiply_adds": len(left_points) * len(right_points) * 5,
        "match_radius": radius,
        "min_disparity": config.min_disparity,
    }


def unavailable_pair(first: int, second: int, reason: str) -> dict[str, Any]:
    pair_id = pair_identifier(first, second)
    return {
        "pair_id": pair_id,
        "view_ids": sorted((first, second)),
        "key": stable_hash(
            {
                "engine_version": ENGINE_VERSION,
                "stage": "matches_unavailable",
                "pair_id": pair_id,
                "reason": reason,
            }
        ),
        "inliers": [],
        "inlier_count": 0,
        "status": "unavailable",
        "reason": reason,
        "multiply_adds": 0,
    }

