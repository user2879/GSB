from __future__ import annotations

from typing import Any

from .utils import quantize_fixed, round_div


def int_camera(camera: dict[str, Any]) -> dict[str, list[list[int]] | list[int]]:
    """Convert validated camera numbers to exact integer prototype values."""
    k = [[int(round(float(v))) for v in row] for row in camera["K"]]
    r = [[int(round(float(v))) for v in row] for row in camera["R"]]
    t = [int(round(float(v))) for v in camera["t"]]
    return {"K": k, "R": r, "t": t}


def camera_center(camera: dict[str, Any]) -> list[int]:
    cam = int_camera(camera)
    r = cam["R"]
    t = cam["t"]
    return [-sum(r[j][i] * t[j] for j in range(3)) for i in range(3)]


def pixel_ray(camera: dict[str, Any], u: int, v: int) -> list[int]:
    cam = int_camera(camera)
    k = cam["K"]
    r = cam["R"]
    fx, _, cx = k[0]
    skew = k[0][1]
    _, fy, cy = k[1]
    raw = [
        fy * (u - cx) - skew * (v - cy),
        fx * (v - cy),
        fx * fy,
    ]
    return [sum(r[j][i] * raw[j] for j in range(3)) for i in range(3)]


def dot(a: list[int], b: list[int]) -> int:
    return sum(x * y for x, y in zip(a, b))


def triangulate_rays(
    camera0: dict[str, Any],
    xy0: tuple[int, int],
    camera1: dict[str, Any],
    xy1: tuple[int, int],
    scale: int,
) -> dict[str, Any]:
    c0 = [value * scale for value in camera_center(camera0)]
    c1 = [value * scale for value in camera_center(camera1)]
    d0 = pixel_ray(camera0, xy0[0], xy0[1])
    d1 = pixel_ray(camera1, xy1[0], xy1[1])
    delta = [c1[i] - c0[i] for i in range(3)]

    a00 = dot(d0, d0)
    a01 = dot(d0, d1)
    a11 = dot(d1, d1)
    denominator = a00 * a11 - a01 * a01
    if denominator <= 0:
        return {"ok": False, "reason": "DEGENERATE_RAY_DENOMINATOR"}

    e0 = dot(d0, delta)
    e1 = dot(d1, delta)
    a_num = a11 * e0 - a01 * e1
    b_num = a01 * e0 - a00 * e1

    point0: list[int] = []
    point1: list[int] = []
    directions: list[str] = []
    for axis in range(3):
        value0, dir0 = round_div(a_num * d0[axis], denominator)
        value1, dir1 = round_div(b_num * d1[axis], denominator)
        p0 = c0[axis] + value0
        p1 = c1[axis] + value1
        mid, mid_dir = round_div(p0 + p1, 2)
        point0.append(mid)
        directions.append(f"{dir0},{dir1},{mid_dir}")

    for camera, point in ((camera0, point0), (camera1, point0)):
        projected = project_point_q(point, camera, scale)
        if not projected["ok"]:
            return {"ok": False, "reason": "NEGATIVE_OR_INVALID_DEPTH"}

    return {"ok": True, "point_q": point0, "solve_round_dirs": directions, "denominator": denominator}


def project_point_q(point_q: list[int], camera: dict[str, Any], scale: int) -> dict[str, Any]:
    cam = int_camera(camera)
    k = cam["K"]
    r = cam["R"]
    t = cam["t"]
    fx, skew, cx = k[0]
    _, fy, cy = k[1]

    xc_q = [sum(r[i][j] * point_q[j] for j in range(3)) + t[i] * scale for i in range(3)]
    z_q = xc_q[2]
    if z_q <= 0:
        return {"ok": False, "reason": "NEGATIVE_DEPTH", "camera_q": xc_q}

    u_q, u_dir = round_div(fx * scale * xc_q[0] + skew * scale * xc_q[1], z_q)
    v_q, v_dir = round_div(fy * scale * xc_q[1], z_q)
    u_q += cx * scale
    v_q += cy * scale
    return {
        "ok": True,
        "u_q": u_q,
        "v_q": v_q,
        "z_q": z_q,
        "round_dirs": [u_dir, v_dir],
        "camera_q": xc_q,
    }


def reprojection_residual_q(
    point_q: list[int],
    observations: dict[str, tuple[int, int]],
    cameras: dict[str, dict[str, Any]],
    scale: int,
    quant_shift: int = 0,
) -> dict[str, Any]:
    view_residuals: dict[str, int] = {}
    directions: dict[str, str] = {}
    total = 0
    for view_id, (u, v) in observations.items():
        projected = project_point_q(point_q, cameras[view_id], scale)
        if not projected["ok"]:
            return {"ok": False, "reason": projected["reason"]}
        eu = projected["u_q"] - u * scale
        ev = projected["v_q"] - v * scale
        squared = eu * eu + ev * ev
        residual_q, direction = round_div(squared, scale)
        residual_q, quant_dir = quantize_fixed(residual_q, quant_shift)
        view_residuals[view_id] = residual_q
        directions[view_id] = f"{direction},{quant_dir}"
        total += residual_q
    mean_q, mean_dir = round_div(total, max(1, len(view_residuals)))
    return {
        "ok": True,
        "residual_q": mean_q,
        "view_residual_q": view_residuals,
        "round_dirs": directions,
        "mean_round_dir": mean_dir,
    }


def quantize_point(point_q: list[int], shift: int) -> tuple[list[int], list[str]]:
    quantized: list[int] = []
    directions: list[str] = []
    for value in point_q:
        q_value, direction = quantize_fixed(value, shift)
        quantized.append(q_value)
        directions.append(direction)
    return quantized, directions
