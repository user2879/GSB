from __future__ import annotations

from typing import Any

from .hashing import sha256_object


def pair_cache_key(left_set: dict[str, Any], right_set: dict[str, Any], strategy: dict[str, Any]) -> str:
    return sha256_object(
        {
            "stage": "match_pair",
            "pair": [left_set["view_index"], right_set["view_index"]],
            "left_cache_key": left_set["cache_key"],
            "right_cache_key": right_set["cache_key"],
            "strategy_fingerprint": strategy["fingerprint"],
            "matcher": "mutual_descriptor_lex_v1",
        }
    )


def _descriptor_distance(left: list[int], right: list[int]) -> int:
    weights = [3, 4, 2, 2, 2, 1, 1]
    return sum(weight * abs(a - b) for weight, a, b in zip(weights, left, right))


def match_pair(
    left_set: dict[str, Any],
    right_set: dict[str, Any],
    strategy: dict[str, Any],
) -> dict[str, Any]:
    left_features = left_set["features"]
    right_features = right_set["features"]
    radius = int(strategy["search_radius"])
    max_distance = int(strategy["descriptor_max_distance"])

    nearest_right: dict[tuple[int, int], tuple[int, int, int]] = {}
    nearest_left: dict[tuple[int, int], tuple[int, int, int]] = {}
    distances: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}

    for left in left_features:
        lx, ly = left["xy"]
        best: tuple[int, int, int] | None = None
        for right in right_features:
            rx, ry = right["xy"]
            if abs(rx - lx) > radius or abs(ry - ly) > max(2, radius // 6):
                continue
            distance = _descriptor_distance(left["descriptor"], right["descriptor"])
            distances[((lx, ly), (rx, ry))] = distance
            if distance > max_distance:
                continue
            if best is None or distance < best[0] or (distance == best[0] and (rx, ry) < best[1:]):
                best = (distance, rx, ry)
        if best is not None:
            nearest_right[(lx, ly)] = best

    for right in right_features:
        rx, ry = right["xy"]
        best = None
        for left in left_features:
            lx, ly = left["xy"]
            if abs(rx - lx) > radius or abs(ry - ly) > max(2, radius // 6):
                continue
            distance = distances.get(((lx, ly), (rx, ry)))
            if distance is None:
                distance = _descriptor_distance(left["descriptor"], right["descriptor"])
            if distance > max_distance:
                continue
            if best is None or distance < best[0] or (distance == best[0] and (lx, ly) < best[1:]):
                best = (distance, lx, ly)
        if best is not None:
            nearest_left[(rx, ry)] = best

    right_by_xy = {(f["xy"][0], f["xy"][1]): f for f in right_features}
    matches = []
    rank = 0
    for left in left_features:
        lx, ly = left["xy"]
        if (lx, ly) not in nearest_right:
            continue
        distance, rx, ry = nearest_right[(lx, ly)]
        reverse = nearest_left.get((rx, ry))
        if reverse is None or (reverse[1], reverse[2]) != (lx, ly):
            continue
        right = right_by_xy[(rx, ry)]
        rank += 1
        match_fingerprint = sha256_object(
            {
                "left_view": left_set["view_index"],
                "right_view": right_set["view_index"],
                "left_xy": [lx, ly],
                "right_xy": [rx, ry],
                "left_feature": left["content_fingerprint"],
                "right_feature": right["content_fingerprint"],
                "distance": distance,
            }
        )
        matches.append(
            {
                "match_id": f"M:{left_set['view_index']:02d}-{right_set['view_index']:02d}:{lx:02d},{ly:02d}",
                "pair": [left_set["view_index"], right_set["view_index"]],
                "left_xy": [lx, ly],
                "right_xy": [rx, ry],
                "left_feature_key": left["content_fingerprint"],
                "right_feature_key": right["content_fingerprint"],
                "descriptor_distance": distance,
                "inlier_rank": rank,
                "pair_cumulative_inliers": rank,
                "match_fingerprint": match_fingerprint,
                "sort_key": [left_set["view_index"], right_set["view_index"], lx, ly],
            }
        )

    cache_key = pair_cache_key(left_set, right_set, strategy)
    return {
        "pair": [left_set["view_index"], right_set["view_index"]],
        "pair_id": f"P:{left_set['view_index']:02d}-{right_set['view_index']:02d}",
        "matches": matches,
        "inlier_count": len(matches),
        "cache_key": cache_key,
        "upstream": {
            "left_cache_key": left_set["cache_key"],
            "right_cache_key": right_set["cache_key"],
            "strategy_fingerprint": strategy["fingerprint"],
        },
        "sort_key": [left_set["view_index"], right_set["view_index"]],
    }
