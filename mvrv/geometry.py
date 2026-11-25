from __future__ import annotations

from typing import Any

from .fixedpoint import SCALE, divide_round


def projection_matrix(camera: dict[str, Any], scale: int = SCALE) -> list[list[int]]:
    k = camera["K_q"]
    r = camera["R_q"]
    t = camera["t_q"]
    matrix: list[list[int]] = []
    for i in range(3):
        row = []
        for j in range(3):
            value = sum(k[i][a] * r[a][j] for a in range(3)) // scale
            row.append(value)
        translation = sum(k[i][a] * t[a] for a in range(3)) // scale
        row.append(translation)
        matrix.append(row)
    return matrix


def _det3(matrix: list[list[int]]) -> int:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)


def _replace_column(matrix: list[list[int]], column: int, values: list[int]) -> list[list[int]]:
    return [
        [values[r] if c == column else matrix[r][c] for c in range(3)]
        for r in range(3)
    ]


def triangulate_fixed(
    observations: list[tuple[int, int, dict[str, Any]]], scale: int = SCALE
) -> dict[str, Any]:
    normal = [[0, 0, 0] for _ in range(3)]
    rhs = [0, 0, 0]
    projections = []

    for x, y, camera in observations:
        p = projection_matrix(camera, scale)
        projections.append(p)
        u_q = x * scale
        v_q = y * scale
        for axis, coordinate_q in ((0, u_q), (1, v_q)):
            row: list[int] = []
            for j in range(3):
                value = (coordinate_q * p[2][j]) // scale - p[axis][j]
                row.append(value)
            const = (coordinate_q * p[2][3]) // scale - p[axis][3]
            b_value = -scale * const
            for i in range(3):
                for j in range(3):
                    normal[i][j] += row[i] * row[j]
                rhs[i] += row[i] * b_value

    determinant = _det3(normal)
    if determinant == 0:
        return {"ok": False, "error": "singular_triangulation"}

    coords: list[int] = []
    directions: list[str] = []
    for axis in range(3):
        numerator = _det3(_replace_column(normal, axis, rhs))
        value, direction = divide_round(numerator, determinant)
        coords.append(value)
        directions.append(direction)

    residual_sum = 0
    residual_count = 0
    reprojections = []
    for (x, y, _camera), p in zip(observations, projections):
        projected = project_fixed(p, coords[0], coords[1], coords[2], scale)
        if not projected["ok"]:
            return {"ok": False, "error": projected["error"]}
        uq, vq = projected["u_q"], projected["v_q"]
        homogeneous = [coords[0], coords[1], coords[2], scale]
        nums = [sum(p[row][j] * homogeneous[j] for j in range(4)) for row in range(3)]
        den = nums[2]
        safe_den = abs(den)
        u_error, _ = divide_round(abs(scale * nums[0] - x * scale * den), safe_den)
        v_error, _ = divide_round(abs(scale * nums[1] - y * scale * den), safe_den)
        residual_sum += u_error + v_error
        residual_count += 2
        reprojections.append([uq, vq])

    residual_q = residual_sum // max(1, residual_count)
    depths = []
    for _x, _y, camera in observations:
        depth_q = camera_depth(camera, coords[0], coords[1], coords[2], scale)
        depths.append(depth_q)

    return {
        "ok": True,
        "coords_q": coords,
        "rounding": {"x": directions[0], "y": directions[1], "z": directions[2]},
        "residual_q": residual_q,
        "depth_q": sum(depths) // len(depths),
        "view_depth_q": depths,
        "reprojections_q": reprojections,
        "normal_det": determinant,
    }


def project_fixed(
    projection: list[list[int]], x_q: int, y_q: int, z_q: int, scale: int = SCALE
) -> dict[str, Any]:
    homogeneous = [x_q, y_q, z_q, scale]
    numerators = [
        sum(projection[row][j] * homogeneous[j] for j in range(4))
        for row in range(3)
    ]
    denominator = numerators[2]
    if denominator == 0:
        return {"ok": False, "error": "projection_at_infinity"}
    sign = 1 if denominator > 0 else -1
    safe_den = abs(denominator)
    u_q, u_dir = divide_round(sign * scale * numerators[0], safe_den)
    v_q, v_dir = divide_round(sign * scale * numerators[1], safe_den)
    return {
        "ok": True,
        "u_q": u_q,
        "v_q": v_q,
        "rounding": {"u": u_dir, "v": v_dir},
        "denominator": denominator,
    }


def camera_depth(
    camera: dict[str, Any], x_q: int, y_q: int, z_q: int, scale: int = SCALE
) -> int:
    r = camera["R_q"]
    t = camera["t_q"]
    point = [x_q, y_q, z_q]
    return sum(r[2][j] * point[j] for j in range(3)) // scale + t[2]
