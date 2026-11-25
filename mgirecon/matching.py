"""Deterministic cross-view feature matching."""

from __future__ import annotations

import math
from itertools import combinations
from typing import Any

from .fixed import digest


def pair_id(left: str, right: str) -> str:
    return f"{left}-{right}"


def candidate_pairs(view_ids: list[str]) -> list[tuple[str, str]]:
    ordered = sorted(view_ids)
    return list(combinations(ordered, 2))


def _feature_cost(left: dict[str, Any], right: dict[str, Any]) -> int:
    left_desc = left["descriptor"]
    right_desc = right["descriptor"]
    color_cost = sum(abs(int(a) - int(b)) for a, b in zip(left_desc["rgb"], right_desc["rgb"]))
    gradient_cost = abs(int(left_desc["gx"]) - int(right_desc["gx"])) + abs(
        int(left_desc["gy"]) - int(right_desc["gy"])
    )
    return color_cost * 4 + gradient_cost


def match_params_key(
    *,
    left_map: dict[str, Any],
    right_map: dict[str, Any],
    min_disparity: int,
    residual_threshold: int,
    max_match_distance: int,
    quantization_level: int,
    min_inliers: int,
) -> str:
    return digest(
        {
            "stage": "match",
            "pair_id": pair_id(left_map["view_id"], right_map["view_id"]),
            "left_feature_key": left_map["cache_key"],
            "right_feature_key": right_map["cache_key"],
            "min_disparity": min_disparity,
            "residual_threshold": residual_threshold,
            "max_match_distance": max_match_distance,
            "quantization_level": quantization_level,
            "min_inliers": min_inliers,
        }
    )


def match_pair(
    left_map: dict[str, Any],
    right_map: dict[str, Any],
    *,
    min_disparity: int,
    residual_threshold: int,
    max_match_distance: int,
    quantization_level: int,
    min_inliers: int,
) -> dict[str, Any]:
    left_view = left_map["view_id"]
    right_view = right_map["view_id"]
    identifier = pair_id(left_view, right_view)
    left_features = sorted(left_map["features"], key=lambda item: (item["coord"][0], item["coord"][1], item["id"]))
    right_features = sorted(right_map["features"], key=lambda item: (item["coord"][0], item["coord"][1], item["id"]))
    unused = {feature["id"]: feature for feature in right_features}
    matches: list[dict[str, Any]] = []
    minimum_sq = min_disparity * min_disparity
    maximum_sq = max_match_distance * max_match_distance
    match_cost_limit = max(1, residual_threshold)
    for left_feature in left_features:
        lx, ly = left_feature["coord"]
        best: tuple[int, str, dict[str, Any]] | None = None
        for right_feature in right_features:
            if right_feature["id"] not in unused:
                continue
            rx, ry = right_feature["coord"]
            disparity_sq = (rx - lx) * (rx - lx) + (ry - ly) * (ry - ly)
            if disparity_sq < minimum_sq or disparity_sq > maximum_sq:
                continue
            cost = _feature_cost(left_feature, right_feature)
            if cost > match_cost_limit:
                continue
            candidate = (cost, right_feature["id"], right_feature)
            if best is None or candidate[:2] < best[:2]:
                best = candidate
        if best is None:
            continue
        cost, right_id, right_feature = best
        del unused[right_id]
        rx, ry = right_feature["coord"]
        disparity_sq = (rx - lx) * (rx - lx) + (ry - ly) * (ry - ly)
        matches.append(
            {
                "left_feature": left_feature["id"],
                "right_feature": right_feature["id"],
                "left_coord": [lx, ly],
                "right_coord": [rx, ry],
                "cost": cost,
                "disparity": math.isqrt(disparity_sq),
                "disparity_sq": disparity_sq,
            }
        )
    matches.sort(key=lambda item: (item["left_coord"][0], item["left_coord"][1], item["right_coord"], item["right_feature"]))
    params_key = match_params_key(
        left_map=left_map,
        right_map=right_map,
        min_disparity=min_disparity,
        residual_threshold=residual_threshold,
        max_match_distance=max_match_distance,
        quantization_level=quantization_level,
        min_inliers=min_inliers,
    )
    cache_key = digest({"params_key": params_key, "matches": matches})
    return {
        "pair_id": identifier,
        "left": left_view,
        "right": right_view,
        "cache_key": cache_key,
        "params_key": params_key,
        "min_disparity": min_disparity,
        "residual_threshold": residual_threshold,
        "max_match_distance": max_match_distance,
        "quantization_level": quantization_level,
        "inlier_count": len(matches),
        "matches": matches,
        "selected": False,
        "tied": False,
        "tie_winner": False,
        "decision_rank": -1,
    }


def select_matching_pairs(pair_results: list[dict[str, Any]], *, min_inliers: int, max_pairs: int) -> list[dict[str, Any]]:
    eligible = [pair for pair in pair_results if pair["inlier_count"] >= min_inliers]
    eligible.sort(key=lambda pair: (-pair["inlier_count"], pair["pair_id"]))
    winners = eligible[:max_pairs]
    winner_ids = {pair["pair_id"] for pair in winners}
    count_occurrences: dict[int, int] = {}
    for pair in eligible:
        count_occurrences[pair["inlier_count"]] = count_occurrences.get(pair["inlier_count"], 0) + 1
    best_count_by_left: dict[str, int] = {}
    for pair in pair_results:
        left = pair["left"]
        best_count_by_left[left] = max(best_count_by_left.get(left, -1), pair["inlier_count"])
    best_ids_by_left: dict[str, list[str]] = {}
    for pair in pair_results:
        if pair["inlier_count"] == best_count_by_left[pair["left"]] and pair["inlier_count"] >= min_inliers:
            best_ids_by_left.setdefault(pair["left"], []).append(pair["pair_id"])
    tie_winners = {min(ids) for ids in best_ids_by_left.values() if len(ids) > 1}
    for rank, pair in enumerate(eligible):
        pair["selected"] = pair["pair_id"] in winner_ids
        pair["decision_rank"] = rank
        pair["tied"] = count_occurrences.get(pair["inlier_count"], 0) > 1
        pair["tie_winner"] = pair["pair_id"] in tie_winners
    return sorted(pair_results, key=lambda pair: pair["pair_id"])
