from __future__ import annotations

from typing import Any

from .geometry import project_point_q
from .utils import fingerprint, quantize_fixed


ROUND_CODE = {"exact": 0, "up": 1, "down": 2}
CODE_ROUND = {str(value): key for key, value in ROUND_CODE.items()}


def _pixel_center(point: dict[str, Any], view_id: str, camera: dict[str, Any], scale: int) -> tuple[int, int, int] | None:
    observations = point.get("observations", {})
    if view_id in observations:
        return int(observations[view_id][0]), int(observations[view_id][1]), int(point["raw_point_q"][2])
    projected = project_point_q(point["raw_point_q"], camera, scale)
    if not projected["ok"]:
        return None
    from .utils import round_div

    x, _ = round_div(projected["u_q"], scale)
    y, _ = round_div(projected["v_q"], scale)
    return x, y, int(projected["z_q"])


def build_depth_map(
    view_id: str,
    view: dict[str, Any],
    accepted_points: list[dict[str, Any]],
    cameras: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    width = int(config["width"])
    height = int(config["height"])
    radius = int(config["depth_radius"])
    bits = int(config["frac_bits"])
    scale = 1 << bits
    shift = int(config["quant_shifts"][config["quant_level"]])
    step = 1 << shift
    camera = cameras[view_id]

    candidates: dict[tuple[int, int], list[tuple[int, str, str]]] = {}
    referenced_points = depth_dependencies(view_id, accepted_points, camera, config)
    for point in sorted(accepted_points, key=lambda p: p["point_id"]):
        center = _pixel_center(point, view_id, camera, scale)
        if center is None:
            continue
        u, v, _ = center
        if u < -radius or u >= width + radius or v < -radius or v >= height + radius:
            continue
        for y in range(max(0, v - radius), min(height, v + radius + 1)):
            for x in range(max(0, u - radius), min(width, u + radius + 1)):
                projected = project_point_q(point["raw_point_q"], camera, scale)
                if not projected["ok"]:
                    continue
                candidates.setdefault((x, y), []).append((int(projected["z_q"]), point["point_id"], point["cache_key"]))

    pixels: list[list[Any]] = []
    conflict_pixels: list[dict[str, Any]] = []
    for y in range(height):
        for x in range(width):
            refs = candidates.get((x, y), [])
            flag = 0
            source = "background"
            if refs:
                refs_sorted = sorted(refs, key=lambda item: (item[0], item[1]))
                raw_depth, source, _ = refs_sorted[0]
                spread = refs_sorted[-1][0] - refs_sorted[0][0]
                if len(refs_sorted) > 1:
                    if 0 < spread <= max(1, step // 2):
                        flag = 1
                        conflict_pixels.append(
                            {
                                "pixel_ref": f"{view_id}:{x}:{y}",
                                "x": x,
                                "y": y,
                                "raw_spread_q": spread,
                                "half_step_q": max(1, step // 2),
                                "rule": config["tie_rules"]["depth_conflict"],
                                "chosen_source": source,
                            }
                        )
                    elif spread > max(1, step // 2):
                        flag = 2
            else:
                block_offset = ((x // 12) + (y // 12)) * 2
                raw_depth = (int(config["depth_background_z"]) + block_offset) * scale

            depth_q, direction = quantize_fixed(int(raw_depth), shift)
            pixels.append([x, y, depth_q, ROUND_CODE[direction], flag, source])

    referenced_points.sort(key=lambda item: item["point_id"])
    param_fingerprint = depth_param_fingerprint(config)
    cache_key = depth_cache_key(view_id, camera, referenced_points, config)
    pixel_hash = fingerprint(pixels)
    return {
        "stage": "depth_map",
        "view_id": view_id,
        "width": width,
        "height": height,
        "cache_key": cache_key,
        "pixel_hash": pixel_hash,
        "param_fingerprint": param_fingerprint,
        "upstream_point_keys": referenced_points,
        "pixel_fields": ["x", "y", "depth_q", "round_code", "flag_1_quant_conflict_2_disagreement", "source_point_id"],
        "round_code_map": CODE_ROUND,
        "pixel_count": len(pixels),
        "conflict_count": len(conflict_pixels),
        "conflict_pixels": conflict_pixels,
        "pixels": pixels,
}


def depth_param_fingerprint(config: dict[str, Any]) -> str:
    return fingerprint(
        {
            "stage": "depth-params",
            "radius": config["depth_radius"],
            "background_z": config["depth_background_z"],
            "width": config["width"],
            "height": config["height"],
            "frac_bits": config["frac_bits"],
            "quant_level": config["quant_level"],
            "quant_shift": config["quant_shifts"][config["quant_level"]],
            "tie_rule": config["tie_rules"]["depth_conflict"],
        }
    )


def depth_dependencies(
    view_id: str,
    accepted_points: list[dict[str, Any]],
    camera: dict[str, Any],
    config: dict[str, Any],
) -> list[dict[str, str]]:
    bits = int(config["frac_bits"])
    scale = 1 << bits
    radius = int(config["depth_radius"])
    width = int(config["width"])
    height = int(config["height"])
    referenced: list[dict[str, str]] = []
    for point in sorted(accepted_points, key=lambda p: p["point_id"]):
        center = _pixel_center(point, view_id, camera, scale)
        if center is None:
            continue
        u, v, _ = center
        if u < -radius or u >= width + radius or v < -radius or v >= height + radius:
            continue
        referenced.append({"point_id": point["point_id"], "cache_key": point["cache_key"]})
    referenced.sort(key=lambda item: item["point_id"])
    return referenced


def depth_cache_key(
    view_id: str,
    camera: dict[str, Any],
    referenced_points: list[dict[str, str]],
    config: dict[str, Any],
) -> str:
    return fingerprint(
        {
            "stage": "depth-map",
            "view_id": view_id,
            "camera_fingerprint": fingerprint(camera),
            "referenced_points": referenced_points,
            "param_fingerprint": depth_param_fingerprint(config),
        }
    )


def backproject_pixel_q(
    view_id: str,
    x: int,
    y: int,
    depth_q: int,
    camera: dict[str, Any],
    config: dict[str, Any],
) -> list[int] | None:
    from .geometry import int_camera
    from .utils import round_div

    cam = int_camera(camera)
    k = cam["K"]
    r = cam["R"]
    t = cam["t"]
    fx, skew, cx = k[0]
    _, fy, cy = k[1]
    if depth_q <= 0 or fx <= 0 or fy <= 0:
        return None
    x_cam_q, _ = round_div(depth_q * (fy * (x - cx) - skew * (y - cy)), fx * fy)
    y_cam_q, _ = round_div(depth_q * (y - cy), fy)
    z_cam_q = depth_q
    centered = [x_cam_q, y_cam_q, z_cam_q]
    world_cam = [centered[i] - t[i] * (1 << int(config["frac_bits"])) for i in range(3)]
    return [sum(r[j][i] * world_cam[j] for j in range(3)) for i in range(3)]


def depth_record_map(depth_map: dict[str, Any]) -> dict[tuple[int, int], list[Any]]:
    return {(int(row[0]), int(row[1])): row for row in depth_map["pixels"]}
