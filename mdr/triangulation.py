from __future__ import annotations

from typing import Any

from .geometry import quantize_point, reprojection_residual_q, triangulate_rays
from .utils import fingerprint


def point_cache_key(match: dict[str, Any], cameras: dict[str, dict[str, Any]], config: dict[str, Any]) -> str:
    left_view = match["left_view"]
    right_view = match["right_view"]
    return fingerprint(
        {
            "stage": "triangulated-point",
            "match_key": match["match_cache_key"],
            "camera_fingerprints": {
                left_view: fingerprint(cameras[left_view]),
                right_view: fingerprint(cameras[right_view]),
            },
            "min_parallax": int(config["min_parallax"]),
            "frac_bits": int(config["frac_bits"]),
            "quant_level": str(config["quant_level"]),
            "quant_shift": int(config["quant_shifts"][config["quant_level"]]),
        }
    )


def triangulate_match(
    match: dict[str, Any],
    cameras: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    left_view = match["left_view"]
    right_view = match["right_view"]
    lx, ly = (int(v) for v in match["left_xy"])
    rx, ry = (int(v) for v in match["right_xy"])
    parallax = max(abs(lx - rx), abs(ly - ry))
    min_parallax = int(config["min_parallax"])
    bits = int(config["frac_bits"])
    scale = 1 << bits
    shift = int(config["quant_shifts"][config["quant_level"]])
    cache_key = point_cache_key(match, cameras, config)

    base = {
        "point_id": "pt-" + str(match["match_cache_key"])[:16],
        "match_id": match["match_id"],
        "match_cache_key": match["match_cache_key"],
        "pair": [left_view, right_view],
        "observations": {left_view: [lx, ly], right_view: [rx, ry]},
        "parallax_pixels": parallax,
        "cache_key": cache_key,
    }

    if parallax < min_parallax:
        return {
            **base,
            "status": "rejected",
            "reject_reason": "MIN_PARALLAX",
            "reject_detail": f"parallax {parallax} < min_parallax {min_parallax}",
        }

    solved = triangulate_rays(cameras[left_view], (lx, ly), cameras[right_view], (rx, ry), scale)
    if not solved["ok"]:
        return {
            **base,
            "status": "rejected",
            "reject_reason": solved["reason"],
        }

    raw_point_q = solved["point_q"]
    point_q, coord_dirs = quantize_point(raw_point_q, shift)
    observations = {left_view: (lx, ly), right_view: (rx, ry)}
    residual = reprojection_residual_q(point_q, observations, cameras, scale, shift)
    if not residual["ok"]:
        return {
            **base,
            "status": "rejected",
            "reject_reason": residual["reason"],
        }

    camera_fingerprints = {view_id: fingerprint(cameras[view_id]) for view_id in (left_view, right_view)}
    return {
        **base,
        "status": "accepted",
        "point_q": point_q,
        "raw_point_q": raw_point_q,
        "coord_round_dirs": coord_dirs,
        "solve_round_dirs": solved["solve_round_dirs"],
        "residual_q": residual["residual_q"],
        "raw_residual_q": residual["residual_q"],
        "view_residual_q": residual["view_residual_q"],
        "residual_round_dirs": residual["round_dirs"],
        "mean_residual_round_dir": residual["mean_round_dir"],
        "camera_fingerprints": camera_fingerprints,
        "cache_key": cache_key,
    }


def triangulate_all(
    matches_by_pair: dict[tuple[str, str], dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    points: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for pair in sorted(matches_by_pair, key=lambda p: (p[0], p[1])):
        bundle = matches_by_pair[pair]
        for match in sorted(bundle["matches"], key=lambda m: (m["left_x"], m["left_y"], m["right_x"], m["right_y"])):
            record = triangulate_match(match, cameras, config)
            if record["status"] == "accepted":
                points.append(record)
            else:
                rejected.append(record)
    points.sort(key=lambda p: (p["pair"][0], p["pair"][1], min(p["observations"].values()), p["point_id"]))
    rejected.sort(key=lambda p: (p["pair"][0], p["pair"][1], p["point_id"]))
    cache_key = fingerprint(
        {
            "stage": "triangulation",
            "point_keys": [p["cache_key"] for p in points],
            "rejected_keys": [p["cache_key"] for p in rejected],
            "config": {
                "min_parallax": config["min_parallax"],
                "frac_bits": config["frac_bits"],
                "quant_level": config["quant_level"],
            },
        }
    )
    return {
        "stage": "triangulation",
        "cache_key": cache_key,
        "point_count": len(points),
        "rejected_count": len(rejected),
        "points": points,
        "rejected": rejected,
    }
