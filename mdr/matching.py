from __future__ import annotations

from typing import Any

from .utils import fingerprint


def descriptor_distance(left: dict[str, Any], right: dict[str, Any]) -> int:
    a = left["descriptor"]
    b = right["descriptor"]
    if len(a) != len(b):
        return 1 << 30
    return sum(abs(int(x) - int(y)) for x, y in zip(a, b))


def match_param_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "matcher_version": 1,
            "max_distance": config["descriptor_max_distance"],
            "order": "pair_sequence_then_xy_lex",
            "frac_bits": config["frac_bits"],
            "quant_level": config["quant_level"],
        }
    )


def match_cache_key(
    pair: tuple[str, str],
    left_set: dict[str, Any],
    right_set: dict[str, Any],
    config: dict[str, Any],
) -> str:
    return fingerprint(
        {
            "stage": "matches",
            "pair": [pair[0], pair[1]],
            "left_feature_cache_key": left_set["cache_key"],
            "right_feature_cache_key": right_set["cache_key"],
            "param_fingerprint": match_param_fingerprint(config),
        }
    )


def match_pair(
    pair: tuple[str, str],
    feature_sets: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    left_id, right_id = pair
    left_set = feature_sets[left_id]
    right_set = feature_sets[right_id]
    max_distance = int(config["descriptor_max_distance"])

    candidates: list[dict[str, Any]] = []
    for left in sorted(left_set["features"], key=lambda f: (int(f["x"]), int(f["y"]))):
        for right in sorted(right_set["features"], key=lambda f: (int(f["x"]), int(f["y"]))):
            distance = descriptor_distance(left, right)
            if distance <= max_distance:
                candidates.append(
                    {
                        "distance": int(distance),
                        "left_x": int(left["x"]),
                        "left_y": int(left["y"]),
                        "right_x": int(right["x"]),
                        "right_y": int(right["y"]),
                        "left": left,
                        "right": right,
                    }
                )

    candidates.sort(
        key=lambda item: (
            item["distance"],
            item["left_x"],
            item["left_y"],
            item["right_x"],
            item["right_y"],
            item["left"]["feature_id"],
            item["right"]["feature_id"],
        )
    )

    used_left: set[str] = set()
    used_right: set[str] = set()
    accepted_raw: list[dict[str, Any]] = []
    for candidate in candidates:
        left = candidate["left"]
        right = candidate["right"]
        if left["feature_id"] in used_left or right["feature_id"] in used_right:
            continue
        used_left.add(left["feature_id"])
        used_right.add(right["feature_id"])
        accepted_raw.append(candidate)

    accepted_raw.sort(
        key=lambda item: (
            item["left_x"],
            item["left_y"],
            item["right_x"],
            item["right_y"],
        )
    )

    matches: list[dict[str, Any]] = []
    for index, candidate in enumerate(accepted_raw):
        left = candidate["left"]
        right = candidate["right"]
        match_id = f"{left_id}__{right_id}__m{index:03d}"
        match_key = fingerprint(
            {
                "stage": "match-item",
                "pair": [left_id, right_id],
                "left_feature_key": left["feature_cache_key"],
                "right_feature_key": right["feature_cache_key"],
                "left_xy": [candidate["left_x"], candidate["left_y"]],
                "right_xy": [candidate["right_x"], candidate["right_y"]],
                "distance": candidate["distance"],
            }
        )
        matches.append(
            {
                "match_id": match_id,
                "match_index": index,
                "inlier_index": index,
                "cumulative_inliers": index + 1,
                "left_view": left_id,
                "right_view": right_id,
                "left_feature_id": left["feature_id"],
                "right_feature_id": right["feature_id"],
                "left_xy": [candidate["left_x"], candidate["left_y"]],
                "right_xy": [candidate["right_x"], candidate["right_y"]],
                "distance": candidate["distance"],
                "left_content_fingerprint": left["content_fingerprint"],
                "right_content_fingerprint": right["content_fingerprint"],
                "match_cache_key": match_key,
            }
        )

    param_fingerprint = match_param_fingerprint(config)
    cache_key = match_cache_key(pair, left_set, right_set, config)
    return {
        "stage": "matches",
        "pair": [left_id, right_id],
        "left_view": left_id,
        "right_view": right_id,
        "cache_key": cache_key,
        "param_fingerprint": param_fingerprint,
        "upstream_cache_keys": [left_set["cache_key"], right_set["cache_key"]],
        "match_count": len(matches),
        "inlier_count": len(matches),
        "matches": matches,
    }
